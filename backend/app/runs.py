from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4
from .prescription import render_plan
from dataclasses import dataclass, field
import anyio

from .db import connection, now, one
from .model import InvocationContext, invocation
from .workflow import run_flow


@dataclass
class Run:
    id: str
    thread_id: str
    user_id: str
    mode: str
    context: InvocationContext = field(default_factory=InvocationContext)
    events: list[dict] = field(default_factory=list)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    started: float = field(default_factory=time.monotonic)
    terminal: dict | None = None
    phase: str = "preparation"
    phase_started: float = field(default_factory=time.monotonic)
    phases_ms: dict = field(default_factory=dict)
    node_started: dict = field(default_factory=dict)


    candidate_id: str | None = None
    candidate_status: str = ""
    candidate_content: str = ""
    sequence: int = 0

    def __post_init__(self):
        self.context.emit = self.emit

    def final_event(self, result, answer, trace):
        payload = {"done": True, "run_id": self.id, "thread_id": self.thread_id,
                   "elapsed_ms": trace["elapsedMs"], "response": result["response"],
                   "intent_type": result.get("intent_type"), "symptoms": result.get("symptoms", []),
                   "notes": result.get("consultation_notes", []), "message": answer, "trace": trace}
        return {**payload, "candidate_id": self.candidate_id, "sequence": self.sequence + 1,
                "type": "answer.final", "payload": payload}

    def emit(self, event_type, payload=None, *, force=False):
        if not force:
            self.context.check()
        payload = payload or {}
        if event_type == "candidate.started":
            self.candidate_id = str(uuid4())
            self.candidate_status = "generating"
            self.candidate_content = payload.get("text", "")
            with connection() as db:
                db.execute("INSERT INTO candidates(id,run_id,status,content,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                           (self.candidate_id, self.id, "generating", self.candidate_content, now(), now()))
        elif event_type == "candidate.delta":
            if self.candidate_status != "generating":
                return None
            if payload.get("channel") == "body":
                self.candidate_content += payload.get("text", "")
        elif event_type == "candidate.completed":
            self.candidate_status = "generated"
            self.candidate_content = payload.get("text", self.candidate_content)
        elif event_type == "review.started":
            self.candidate_status = "reviewing"
        elif event_type in {"review.rejected", "review.approved"}:
            self.candidate_status = "rejected" if event_type.endswith("rejected") else "approved"
        elif event_type in {"run.cancelled", "run.failed"}:
            self.candidate_status = "cancelled" if event_type.endswith("cancelled") else "failed"
        elif event_type == "answer.final":
            self.candidate_status = "final"
            self.candidate_content = payload["response"]
        if self.candidate_id:
            with connection() as db:
                db.execute("UPDATE candidates SET status=?,content=?,updated_at=? WHERE id=?",
                           (self.candidate_status, self.candidate_content, now(), self.candidate_id))
                if "plan" in payload:
                    db.execute("UPDATE candidates SET plan=? WHERE id=?",
                               (json.dumps(payload["plan"], ensure_ascii=False), self.candidate_id))
                if "review" in payload:
                    db.execute("UPDATE candidates SET review=? WHERE id=?",
                               (json.dumps(payload["review"], ensure_ascii=False), self.candidate_id))
        self.sequence += 1
        event = {"run_id": self.id, "candidate_id": self.candidate_id, "sequence": self.sequence,
                 "type": event_type, "payload": payload}
        # Existing REST completion fields and stage fields remain compatible during the protocol migration.
        if event_type in {"stage", "answer.final", "run.cancelled", "run.failed"}:
            event.update({key: value for key, value in payload.items() if key != "type"})
        with connection() as db:
            db.execute("INSERT INTO run_events(run_id,sequence,candidate_id,type,payload) VALUES(?,?,?,?,?)",
                       (self.id, self.sequence, self.candidate_id, event_type, json.dumps(payload, ensure_ascii=False)))
        self.queue.put_nowait(event)
        return event

    def close_nodes(self, status):
        for event in list(self.events):
            key = (event["stage"], event.get("attempt", 1))
            if event["status"] == "start" and not any(
                (item["stage"], item.get("attempt", 1)) == key and item["status"] != "start"
                for item in self.events
            ):
                self.events.append({**event, "status": status,
                                    "elapsed_ms": round((time.monotonic() - self.node_started.get(key, self.started)) * 1000)})

    def set_phase(self, phase: str):
        stamp = time.monotonic()
        if phase != self.phase:
            self.phases_ms[self.phase] = self.phases_ms.get(self.phase, 0) + round((stamp - self.phase_started) * 1000)
            self.phase, self.phase_started = phase, stamp

    def telemetry(self):
        return {"run_id": self.id, "events": self.events, "model_calls": self.context.calls,
                "elapsedMs": round((time.monotonic() - self.started) * 1000), "mode": self.mode,
                "phases_ms": self.phases_ms}


active: dict[str, Run] = {}


def persist_stop(run: Run, status: str):
    trace = {**run.telemetry(), "status": status}
    # A completed run is immutable: a disconnect while sending cannot erase its authoritative result.
    with connection() as db:
        changed = db.execute("UPDATE chat_runs SET status=?,telemetry=?,updated_at=? WHERE id=? AND status IN ('running','cancelling')",
                             (status, json.dumps(trace, ensure_ascii=False), now(), run.id)).rowcount
        if changed:
            from uuid import uuid4
            db.execute("INSERT INTO messages(id,thread_id,role,content,intent_type,trace,created_at) VALUES(?,?,?,?,?,?,?)",
                       (str(uuid4()), run.thread_id, "assistant",
                        "已停止，本轮没有生成最终答复。" if status == "cancelled" else "问答暂不可用，本轮未提供治疗建议，请稍后重试。",
                        status, json.dumps(trace, ensure_ascii=False), now()))


async def execute(run: Run, message: str, profile: dict, file_text: str, save):
    token = invocation.set(run.context)
    try:
        if run.candidate_id is None:
            run.emit("candidate.started", {})

        async for event in run_flow(run.thread_id, run.mode, message, profile, file_text, run.user_id, run.id):
            run.context.check()
            if event["type"] == "stage":
                if event["status"] == "start":
                    run.node_started[(event["stage"], event.get("attempt", 1))] = time.monotonic()
                    phase = ("generation" if event["stage"] in {"bianzheng", "formula", "herb", "synthesis", "revise_formula"}
                             else "review" if event["stage"] in {"reviewer", "safety"} else None)
                    if phase and phase != run.phase:
                        run.set_phase(phase)
                        run.emit("phase", {"phase": phase})
                run.events.append(event)
                with connection() as db:
                    db.execute("UPDATE chat_runs SET telemetry=?,updated_at=? WHERE id=?",
                               (json.dumps(run.telemetry(), ensure_ascii=False), now(), run.id))
                run.emit("stage", event)
            elif event["type"] == "result":
                # No await between the cancellation fence and atomic result + memory commit.
                run.set_phase("ready")
                trace = {**run.telemetry(), "status": "completed", "preflight": event["data"].get("preflight", {}),
                         "memory_retrieval": event["data"].get("memory_retrieval", {}),
                         "classics_retrieval": {"attempts": event["data"].get("retrieval_history", []),
                                                "final": event["data"].get("retrieval_trace", {})},
                         "inquiry": {"facts": event["data"].get("symptom_facts", []),
                                     "pending": event["data"].get("symptom_pending")},
                         "safety": {key: event["data"].get("safety_state", {}).get(key)
                                    for key in ("subject", "status", "pending", "version")}}

                if run.candidate_status != "approved":
                    if run.candidate_status == "rejected":
                        run.emit("candidate.started", {"restricted": True})
                    run.emit("candidate.completed", {"text": event["data"]["response"]})
                    run.emit("review.started", {"restricted": True})
                    run.emit("review.approved", {"review": {"approved": True, "restricted": True}, "plan": None})
                # The final event, candidate, answer and memory commit together.
                run.terminal = save(run, event["data"], trace)
                run.sequence = run.terminal["sequence"]
                run.candidate_status = "final"
                run.candidate_content = run.terminal["response"]
                run.queue.put_nowait(run.terminal)
    except asyncio.CancelledError:
        run.context.cancelled = True
        run.set_phase("cancelled")
        # Custom stream events may be interrupted before delivery; close all unfinished nodes explicitly.
        run.close_nodes("cancelled")
        persist_stop(run, "cancelled")
        run.terminal = {"cancelled": True, "run_id": run.id, "thread_id": run.thread_id}
        run.terminal = run.emit("run.cancelled", run.terminal, force=True)
        raise
    except Exception:
        run.set_phase("failed")
        run.close_nodes("failed")
        persist_stop(run, "failed")
        run.terminal = {"error": "问答暂不可用，本轮未提供治疗建议，请稍后重试。", "run_id": run.id}
        run.terminal = run.emit("run.failed", run.terminal, force=True)
    finally:
        invocation.reset(token)
        active.pop(run.id, None)


async def cancel(run_id: str):
    run = active.get(run_id)
    if not run:
        return
    row = one("SELECT status FROM chat_runs WHERE id=?", (run_id,))
    if not row or row["status"] not in {"running", "cancelling"}:
        return
    first_cancellation = not run.context.cancelled
    run.context.cancelled = True
    with connection() as db:
        db.execute("UPDATE chat_runs SET status='cancelling',updated_at=? WHERE id=? AND status='running'", (now(), run_id))
    if run.task:
        if first_cancellation:
            run.task.cancel()
        # Starlette's ASGI 2.3 disconnect listener cancels an AnyIO scope.
        # Protect finalizers from repeated cancellation while HTTP streams close.
        with anyio.CancelScope(shield=True):
            try:
                await asyncio.shield(run.task)
            except asyncio.CancelledError:
                pass
    # Handles cancellation before the task got its first execution opportunity.
    if run.terminal is None:
        persist_stop(run, "cancelled")
        run.terminal = {"cancelled": True, "run_id": run.id, "thread_id": run.thread_id}
        run.terminal = run.emit("run.cancelled", run.terminal, force=True)
        active.pop(run.id, None)


def recover_interrupted():
    rows = []
    with connection() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM chat_runs WHERE status IN ('running','cancelling')")]
    for row in rows:
        run = Run(row["id"], row["thread_id"], row["user_id"], "fixed")
        candidate = one("SELECT id FROM candidates WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run.id,))
        run.candidate_id = candidate["id"] if candidate else None
        run.sequence = one("SELECT COALESCE(MAX(sequence),0) AS value FROM run_events WHERE run_id=?", (run.id,))["value"]
        old = json.loads(row["telemetry"])
        run.events = old.get("events", [])
        run.context.calls = old.get("model_calls", [])
        for call in run.context.calls:
            if call.get("status") == "running":
                call["status"] = "cancelled"
        run.close_nodes("cancelled")
        persist_stop(run, "cancelled")
        run.emit("run.cancelled", {"cancelled": True, "recovered": True}, force=True)
