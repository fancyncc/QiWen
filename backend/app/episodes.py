"""Evidence-based episode boundaries and deterministic rolling summaries."""
from __future__ import annotations

import copy
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from .db import now
from . import symptoms

QUESTION = "这次是继续上次的不适，还是开始咨询一个新的问题？请回答“继续上次”或“新的问题”。"


def elapsed_days(value: str) -> float:
    try:
        date = datetime.fromisoformat(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - date).total_seconds() / 86400
    except (ValueError, TypeError):
        return 0


def resolve(state: dict) -> dict:
    safe = copy.deepcopy(state["safety_state"])
    if state.get("identity_mode") != "clinical" or not safe.get("subject_confirmed"):
        return {"episode_id": "", "episode_pending": False}
    subject, text = safe["subject"], state.get("message", "")
    episodes = safe.setdefault("episodes", {})
    episode = episodes.get(subject)
    facts = safe.get("symptom_facts", [])
    if not episode:
        dates = [f.get("source", {}).get("collected_at", "") for f in facts if f.get("subject") == subject]
        episode = {"id": str(uuid4()), "started_at": None if dates else now(), "last_seen_at": max(dates) if dates else now(), "legacy": bool(dates)}
        episodes[subject] = episode
    explicit_new = bool(re.search(r"新的问题|新问题|新的不适|换个问题|另一个问题|重新开始问诊", text))
    continuing = bool(re.search(r"继续上次|还是上次|接着上次|同一个问题", text))
    stale = (episode.get("legacy") and not episode.get("last_seen_at")) or elapsed_days(episode.get("last_seen_at", "")) > max(1, int(os.getenv("QIWEN_EPISODE_RECHECK_DAYS", "14")))
    boundary = safe.get("episode_confirmation")
    if boundary and boundary.get("subject") != subject:
        boundary = None
        safe.pop("episode_confirmation", None)
    if explicit_new:
        previous_id = episode["id"]
        episode = {"id": str(uuid4()), "started_at": now(), "last_seen_at": now(), "legacy": False}
        episodes[subject] = episode
        for fact in facts:
            if fact.get("subject") == subject and not fact.get("superseded_by"):
                fact.setdefault("episode_id", previous_id)
                fact["archived_episode"] = True
        safe["episode"] = ""
        safe["clinical"][subject] = {"pending_context": "", "pending_question": "", "symptom_pending": None}
        safe["subject_epoch"] = safe.get("subject_epoch", 0) + 1
    if stale or explicit_new:
        safe["recheck_temporal"] = True
        for fact in safe["facts"]:
            if fact.get("subject") in {subject, "unknown"} and fact["category"] in {"medication", "pregnancy_lactation", "urgent"}:
                # Never expire risk into a denial; old observations require review.
                if fact.get("source", {}).get("id") != state.get("source_message_id"):
                    fact["temporal_scope"] = "past"
        for key in ("medication_details", "pregnancy_status", "urgent_details"):
            safe["answers"].pop(key, None)
        for saved in safe.get("safety_by_subject", {}).values():
            for key in ("medication_details", "pregnancy_status", "urgent_details"):
                saved.get("answers", {}).pop(key, None)
    if stale and not continuing and not explicit_new and not state.get("identity_urgent"):
        if not boundary:
            from .workflow import attachment_text
            buffered = symptoms.extract(text, subject=subject, message_id=state.get("source_message_id", ""), thread_id=state["thread_id"])
            buffered += symptoms.extract(attachment_text(state), subject=subject, message_id=state.get("source_message_id", ""), thread_id=state["thread_id"], kind="attachment")
            boundary = {"subject": subject, "message": text, "file_text": state.get("file_text", ""),
                        "source_message_id": state.get("source_message_id", ""), "facts": buffered}
        safe["episode_confirmation"] = boundary
        return {"safety_state": safe, "episode_id": episode["id"], "episode_pending": True,
                "pending_context": "", "pending_question": QUESTION, "symptom_pending": None}
    updates = {"episode_pending": False}
    if boundary and (continuing or explicit_new):
        buffered = boundary.get("facts", [])
        for fact in buffered:
            fact["episode_id"] = episode["id"]
        facts = symptoms.merge(facts, buffered)
        updates["pending_context"] = boundary["message"]
        updates["file_text"] = boundary.get("file_text", "")
        safe.pop("episode_confirmation", None)
        updates["symptom_facts"] = facts
    episode["last_seen_at"] = now()
    safe["symptom_facts"] = facts
    return {**updates, "safety_state": safe, "episode_id": episode["id"],
            "identity_scope": f"{state['thread_id']}:{subject}:{safe.get('subject_epoch', 0)}",
            **({"pending_context": boundary["message"] if boundary else "", "pending_question": "", "symptom_pending": None} if explicit_new else {})}


def summary(state: dict) -> tuple[str, dict]:
    safe = state["safety_state"]
    subject, episode_id = safe["subject"], state.get("episode_id", "")
    facts = symptoms.effective(state.get("symptom_facts", []), subject)
    # Only source-backed patient statements; no model diagnosis, plan or generated advice.
    lines = [symptoms.describe(facts, subject)] if facts else []
    if state.get("pending_question"):
        lines.append("待确认：" + state["pending_question"])
    data = {"episode_id": episode_id, "subject": subject, "facts": facts,
            "pending_question": state.get("pending_question", ""),
            "risk_facts": [copy.deepcopy(f) for f in safe.get("facts", []) if f.get("subject") == subject
                           and f.get("source", {}).get("kind") not in {"summary", "manual_profile", "confirmed_profile"}],
            "source_message_ids": list(dict.fromkeys(f.get("source", {}).get("message_id", "") for f in facts)),
            "last_run_id": state.get("run_id", ""),
            "started_at": safe.get("episodes", {}).get(subject, {}).get("started_at")}
    labels = {"present": "用户自述有", "absent": "用户明确否认", "uncertain": "待确认"}
    for fact in data["risk_facts"]:
        lines.append(f"安全自述：{fact['entity']}（{fact['category']}、{fact['temporal_scope']}、{labels.get(fact['assertion'], '待确认')}）")
    # Drop complete oldest facts from the derived text only; structured evidence stays intact.
    while len("\n".join(lines)) > 1800 and len(facts) > 1:
        facts = facts[1:]
        lines[0] = symptoms.describe(facts, subject)
    return "\n".join(lines), data
