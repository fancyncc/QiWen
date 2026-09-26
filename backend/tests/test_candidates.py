import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app import model, workflow
from app.db import connection, one
from test_api import auth, register
from test_model import configure, output


def test_candidate_rejection_replacement_and_final_history():
    with TestClient(app) as client:
        token, user_id = register(client, "versions_" + uuid4().hex[:8])
        headers = auth(token)
        client.patch(f"/api/profiles/{user_id}", headers=headers, json={"allergies": "甘草"})
        with client.stream("POST", "/api/chat?stream=true", headers=headers,
                           json={"message": "最近失眠三天，白天疲劳", "mode": "multi"}) as response:
            events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        assert all(set(("run_id", "candidate_id", "sequence", "type", "payload")) <= event.keys() for event in events)
        assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
        rejected = [event for event in events if event["type"] == "review.rejected"]
        started = [event for event in events if event["type"] == "candidate.started"]
        assert len(rejected) == 3 and len(started) == 4
        assert len({event["candidate_id"] for event in started}) == 4
        final = events[-1]
        assert final["type"] == "answer.final"
        assert final["candidate_id"] not in {event["candidate_id"] for event in rejected}
        versions = client.get(f'/api/runs/{final["run_id"]}/candidates', headers=headers).json()
        assert [row["status"] for row in versions] == ["rejected", "rejected", "rejected", "final"]
        assert versions[-1]["content"] == final["payload"]["response"]
        history = client.get(f'/api/threads/{final["thread_id"]}/messages', headers=headers).json()
        assert len(history) == 2 and history[-1]["content"] == final["payload"]["response"]
        assert history[-1]["candidate_id"] == final["candidate_id"]
        assert history[-1]["prescription"] is None
        persisted = client.get(f'/api/runs/{final["run_id"]}/events', headers=headers).json()
        assert [row["type"] for row in persisted] == [event["type"] for event in events]


def test_real_sse_deltas_precede_completion_and_cancel_blocks_late_events(monkeypatch):
    entered, closed = threading.Event(), threading.Event()
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first-live-chunk"},"finish_reason":null}]}\n\n'
            entered.set()
            await asyncio.sleep(60)
            yield b'data: {"choices":[{"delta":{"content":"late-chunk"},"finish_reason":"stop"}]}\n\n'
        async def aclose(self):
            closed.set()
    def handler(request):
        role = json.loads(request.content)["model"]
        if role == "safety_preflight":
            return output('{"decision":"allow","risk_categories":[],"reason":"stub","required_questions":[]}')
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())
    configure(monkeypatch, handler)
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
    for role in model.ROLES:
        monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), role)
    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        token, _ = register(client, "live_delta_" + uuid4().hex[:8])
        headers = auth(token)
        run_id = str(uuid4())
        future = executor.submit(client.post, "/api/chat", headers=headers,
                                 json={"run_id": run_id, "message": "最近失眠三天，白天疲劳"})
        assert entered.wait(5)
        before = client.get(f"/api/runs/{run_id}/events", headers=headers).json()
        delta = next(event for event in before if event["type"] == "candidate.delta")
        assert delta["payload"]["text"] == "first-live-chunk"
        assert not any(event["type"] in {"candidate.completed", "review.started", "answer.final"} for event in before)
        client.post(f"/api/runs/{run_id}/cancel", headers=headers).raise_for_status()
        assert future.result(timeout=5).json()["type"] == "run.cancelled"
        assert closed.wait(2)
        after = client.get(f"/api/runs/{run_id}/events", headers=headers).json()
        assert after[-1]["type"] == "run.cancelled"
        assert not any("late-chunk" in json.dumps(event) or event["type"] == "answer.final" for event in after)


def test_final_event_failure_rolls_back_answer_and_memory():
    with TestClient(app) as client:
        token, _ = register(client, "atomic_" + uuid4().hex[:8])
        headers = auth(token)
        thread_id = client.post("/api/threads", headers=headers, json={}).json()["id"]
        with connection() as db:
            db.execute("CREATE TRIGGER fail_final_event BEFORE INSERT ON run_events WHEN NEW.type='answer.final' BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        try:
            result = client.post("/api/chat", headers=headers, json={"thread_id": thread_id, "message": "最近失眠三天，白天疲劳"})
            assert result.status_code == 503
            history = client.get(f"/api/threads/{thread_id}/messages", headers=headers).json()
            assert len(history) == 2 and history[-1]["intent_type"] == "failed"
            assert history[-1]["prescription"] is None
            assert one("SELECT * FROM profile_memories WHERE thread_id=?", (thread_id,)) is None
            run = one("SELECT id,status FROM chat_runs WHERE thread_id=?", (thread_id,))
            assert run["status"] == "failed"
            assert not one("SELECT * FROM run_events WHERE run_id=? AND type='answer.final'", (run["id"],))
        finally:
            with connection() as db:
                db.execute("DROP TRIGGER fail_final_event")
