"""Session-only clinical memory and legacy controls regression fixtures."""
import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db, memory, workflow, safety_store, model, symptoms
from app.context import clinical_context
from app.safety_context import empty_state, extract_facts, applicable_facts
from app.agents import tools, multi
from app.main import app
from test_api import auth, register


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memory.sqlite3")
    db.init_db()
    with db.connection() as conn:
        for user in ("alice", "bob"):
            conn.execute("INSERT INTO users VALUES(?,?,?,?)", (user,user,"unused",db.now()))
            conn.execute("INSERT INTO profiles(user_id,updated_at) VALUES(?,?)", (user,db.now()))
        for thread, user, summary in (
            ("current", "alice", "头痛"),
            ("head", "alice", "上个月头痛"),
            ("sleep", "alice", "之前没有失眠"),
            ("other-user", "bob", "头痛失眠"),
        ):
            conn.execute("INSERT INTO threads VALUES(?,?,?,'fixed',?,?)", (thread,user,thread,db.now(),db.now()))
            conn.execute("INSERT INTO profile_memories VALUES(?,?,?,?)", (thread,user,summary,"2026-09-01T00:00:00+00:00"))
    return db


def test_legacy_summary_controls_delete_without_losing_chat_or_other_users(memory_db):
    with db.connection() as conn:
        control = memory.bump_control(conn, "alice", "head", False)
    assert control == {"enabled": False, "revision": 1}
    assert not db.one("SELECT * FROM profile_memories WHERE thread_id='head'")
    assert db.one("SELECT * FROM threads WHERE id='head'")
    assert db.one("SELECT * FROM profile_memories WHERE thread_id='other-user'")
    db.init_db()
    assert memory.control("alice", "head")["enabled"] == 0
    assert db.one("SELECT * FROM profile_memories WHERE thread_id='sleep'")
    assert not db.one("SELECT * FROM memory_terms")


def test_old_context_and_summary_risks_never_enter_clinical_input():
    risks = extract_facts("正在服用华法林", subject="self", kind="summary", thread_id="old")
    state = {"message": "我今天头痛", "thread_id": "new", "memory_context": "foreign-marker",
             "memory_records": [{"summary": "foreign-marker", "thread_id": "old"}],
             "safety_state": {**empty_state(), "subject": "self", "subject_confirmed": True, "facts": risks}}
    text = clinical_context(state)
    assert "foreign-marker" not in text and "华法林" not in text
    assert not applicable_facts(state["safety_state"])
    assert not hasattr(tools, "MEMORY_TOOL") and not hasattr(tools, "search_memories")
    assert "memory_rerank" not in model.ROLES


def test_legacy_saved_state_and_foreign_symptoms_are_excluded(memory_db):
    state = {**empty_state(), "subject": "self", "subject_confirmed": True}
    state["facts"] = extract_facts("正在服用华法林", subject="self", kind="summary", thread_id="head")
    state["symptom_facts"] = symptoms.extract("我咳嗽三天", kind="summary", thread_id="head")
    with db.connection() as conn:
        safety_store.commit_state(conn, "alice", "current", state, 0)
    loaded = safety_store.load_state("alice", "current")
    assert loaded["facts"] == [] and loaded["symptom_facts"] == []
    foreign = symptoms.extract("我腹胀三天", thread_id="head")
    text = clinical_context({"thread_id": "current", "message": "我今天头痛", "safety_state": loaded,
                             "symptom_facts": foreign})
    assert "腹胀" not in text and "咳嗽" not in text and "华法林" not in text


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_new_thread_uses_only_profile_and_its_own_clinical_facts(monkeypatch, mode):
    captured = []
    original = workflow.profile_load
    async def load(state):
        result = await original(state)
        captured.append(clinical_context({**state, **result}))
        return result
    monkeypatch.setattr(workflow, "profile_load", load)
    with TestClient(app) as client:
        token, uid = register(client, "session_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{uid}", headers=h, json={"allergies": "甘草", "history": "画像背景标记"}).raise_for_status()
        old = client.post("/api/threads", headers=h, json={"mode": mode}).json()["id"]
        upload = client.post("/api/upload", headers=h, files={"file": ("old.txt", "附件来源隔离标记".encode(), "text/plain")})
        upload.raise_for_status()
        first = client.post("/api/chat", headers=h, json={"thread_id": old, "message": "我咳嗽三天，夜间明显，没有发热", "files": [upload.json()["id"]]})
        first.raise_for_status()
        with db.connection() as conn:
            conn.execute("INSERT INTO profile_memories VALUES(?,?,?,?)", (old, uid, "旧会话症状标记，咳嗽三天", db.now()))
        new = client.post("/api/threads", headers=h, json={"mode": mode}).json()["id"]
        captured.clear()
        reply = client.post("/api/chat", headers=h, json={"thread_id": new, "message": "我今天头痛三天，没有其他不适"})
        reply.raise_for_status()
        assert captured and all("甘草" in text and "画像背景标记" in text for text in captured)
        assert all("咳嗽" not in text and "夜间" not in text and "旧会话症状标记" not in text and "附件来源隔离标记" not in text for text in captured)
        state = safety_store.load_state(uid, new)
        assert all(f["symptom"] != "咳嗽" for f in state["symptom_facts"])
        assert not db.one("SELECT * FROM profile_memories WHERE thread_id=?", (new,))
        assert db.one("SELECT * FROM consultation_episodes WHERE thread_id=?", (new,))
        assert db.one("SELECT summary FROM profile_memories WHERE thread_id=?", (old,))["summary"].startswith("旧会话")
        assert "memories" not in client.get(f"/api/profiles/{uid}", headers=h).json()


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_same_thread_retains_duration_denials_and_sources(monkeypatch, mode):
    captured = []
    original = workflow.profile_load
    async def load(state):
        result = await original(state)
        captured.append(clinical_context({**state, **result}))
        return result
    monkeypatch.setattr(workflow, "profile_load", load)
    with TestClient(app) as client:
        token, uid = register(client, "continuity_" + uuid4().hex[:8])
        h = auth(token)
        t = client.post("/api/threads", headers=h, json={"mode": mode}).json()["id"]
        for text in ("我咳嗽三天", "夜间明显", "我没有发热，也没有其他不适"):
            client.post("/api/chat", headers=h, json={"thread_id": t, "message": text}).raise_for_status()
        assert "三天" in captured[-1] and "夜间" in captured[-1]
        summary = db.one("SELECT summary FROM consultation_episodes WHERE thread_id=? ORDER BY updated_at DESC LIMIT 1", (t,))["summary"]
        assert all(term in summary for term in ("三天", "夜间", "否认")), summary
        assert not db.one("SELECT * FROM profile_memories WHERE thread_id=?", (t,))


def test_bare_qualifier_keeps_original_duration_source_and_does_not_guess_between_complaints():
    facts = symptoms.extract("我咳嗽三天，夜间明显", message_id="initial", thread_id="t")
    bound = symptoms.bind_details(facts, "饭后加重", subject="self", message_id="followup", thread_id="t")
    result = symptoms.effective(symptoms.merge(facts, bound), "self")[0]
    assert result["duration_text"] == "三天" and result["timing_text"] == "饭后"
    assert result["evidence_refs"]["duration_text"]["message_id"] == "initial"
    assert result["evidence_refs"]["timing_text"]["message_id"] == "followup"
    multiple = symptoms.extract("我咳嗽三天，同时头痛", message_id="two", thread_id="t")
    assert not symptoms.bind_details(multiple, "夜间明显", subject="self", message_id="answer", thread_id="t")
    assert not symptoms.bind_details(facts, "母亲夜间明显", subject="self", message_id="answer", thread_id="t")


def test_retired_display_endpoints_and_legacy_delete_permissions():
    with TestClient(app) as client:
        token, uid = register(client, "legacy_" + uuid4().hex[:8])
        other, _ = register(client, "legacy_" + uuid4().hex[:8])
        h = auth(token)
        t = client.post("/api/threads", headers=h, json={}).json()["id"]
        assert client.get("/api/memories", headers=h).status_code == 404
        assert client.get(f"/api/threads/{t}/context", headers=h).status_code == 404
        with db.connection() as conn:
            conn.execute("INSERT INTO profile_memories VALUES(?,?,?,?)", (t, uid, "历史保留", db.now()))
        assert client.delete(f"/api/threads/{t}/memory", headers=auth(other)).status_code == 404
        assert client.delete(f"/api/profiles/{uid}/memories/{t}", headers=h).status_code == 204
        assert not db.one("SELECT * FROM profile_memories WHERE thread_id=?", (t,))
        assert client.get(f"/api/threads/{t}/messages", headers=h).status_code == 200
