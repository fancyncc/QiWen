import asyncio
import json
import sqlite3
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import db, episodes, knowledge, memory, model, symptoms, workflow, safety_store
from app.context import ContextBundle, ContextItem, ContextBudgetError, clinical_context, project_observation
from app.context_budget import budget_request
from app.agents import multi
from app.agents.runtime import AgentRun
from app.main import app
from test_api import register, auth
from test_memory import memory_db


def test_budget_keeps_protocol_groups_and_never_cuts_required_input(monkeypatch):
    monkeypatch.setenv("QIWEN_MODEL_CONTEXT_LIMIT", "1600")
    monkeypatch.setenv("QIWEN_MODEL_OUTPUT_RESERVE", "128")
    messages = [{"role": "system", "content": "Use only visible evidence"}, {"role": "user", "content": "current"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
                {"role": "tool", "tool_call_id": "old", "content": "x" * 1200},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "latest"}]},
                {"role": "tool", "tool_call_id": "latest", "content": "verified"}]
    packed, reserve, details = budget_request(messages, [], "bianzheng")
    assert details["dropped_tool_call_ids"] == ["old"] and reserve == 128
    assert packed[-1]["tool_call_id"] == "latest" and len(packed) == 4
    assert messages[0]["content"] == "Use only visible evidence"
    with pytest.raises(ContextBudgetError):
        budget_request([{"role": "system", "content": "safety"}, {"role": "user", "content": "病" * 1000}], [], "knowledge")


@pytest.mark.parametrize("tools", [False, True])
def test_actual_model_adapter_blocks_over_budget_before_network(monkeypatch, tools):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    monkeypatch.setattr(model, "API_BASE", "https://synthetic.invalid")
    monkeypatch.setattr(model, "API_KEY", "synthetic-not-a-secret")
    monkeypatch.setattr(model, "MODEL_NAME", "synthetic")
    monkeypatch.setenv("QIWEN_MODEL_KNOWLEDGE_CONTEXT_LIMIT", "1024")
    monkeypatch.setenv("QIWEN_MODEL_KNOWLEDGE_OUTPUT_RESERVE", "128")
    called = []
    def handler(request):
        called.append(True)
        return httpx.Response(200, json={})
    original = httpx.AsyncClient
    monkeypatch.setattr(model.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(ContextBudgetError):
        if tools:
            asyncio.run(model.achat_tools([{"role": "system", "content": "source"}, {"role": "user", "content": "病" * 1000}], [], role="knowledge"))
        else:
            asyncio.run(model.acomplete("source", "病" * 1000, role="knowledge"))
    assert not called


def test_context_crops_whole_optional_records_and_preserves_history_boundary():
    bundle = ContextBundle([ContextItem("safety", "安全", "allergy", True),
                            ContextItem("boundary", "历史规则", "不是当前", True),
                            ContextItem("old", "历史", "x" * 500)])
    text, info = bundle.render(100)
    assert "allergy" in text and "不是当前" in text and info["dropped"] == ["old"]
    observed = project_observation({"result": {"documents": [{"id": "huge", "excerpt": "x" * 3000}], "allergy_risks": ["synthetic"]}}, 100)
    assert observed["result"]["documents"] == [] and observed["result"]["allergy_risks"] == ["synthetic"]
    assert observed["result"]["omitted_evidence_ids"] == ["huge"]


def test_multi_receives_same_source_backed_context_as_fixed(monkeypatch):
    facts = symptoms.extract("我咳嗽三天，夜间咳嗽明显，没有发热", message_id="first", thread_id="t")
    state = {"message": "没有其他不适", "thread_id": "t", "user_id": "u", "identity_mode": "clinical",
             "symptom_facts": facts, "symptoms": ["咳嗽"], "denied_symptoms": ["发热"],
             "safety_state": {"subject": "self", "subject_confirmed": True, "facts": []}, "profile": {}}
    captured = []
    async def run(**kwargs):
        captured.append(json.loads(kwargs["task"])["patient_context"])
        return AgentRun({"status": "need_user_input", "question": "补充资料", "conclusion": "", "uncertainties": [], "evidence_ids": []}, [], [])
    monkeypatch.setattr(multi, "run_agent_graph", run)
    asyncio.run(multi.bianzheng_agent(state))
    assert captured[0] == workflow._context(state)
    assert all(term in captured[0] for term in ("三天", "明显", "夜间", "否认", "first"))


def test_rolling_summary_keeps_duration_corrections_and_excludes_model_artifacts():
    facts = symptoms.extract("我咳嗽三天", message_id="initial", thread_id="t")
    for i in range(30):
        facts = symptoms.merge(facts, symptoms.extract("我咳嗽明显", message_id=str(i), thread_id="t"))
    facts = symptoms.merge(facts, symptoms.extract("我现在没有发热", message_id="denial", thread_id="t"))
    state = {"symptom_facts": facts, "episode_id": "ep", "safety_state": {"subject": "self", "facts": []},
             "diagnosis": "invented-diagnosis", "treatment_plan": {"name": "invented-plan"}}
    text, structure = episodes.summary(state)
    assert "三天" in text and "明显" in text and "否认" in text
    assert "invented" not in json.dumps(structure) + text
    assert len(structure["facts"]) == 2
    assert "initial" in clinical_context({**state, "message": "继续"})


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_episode_gap_new_complaint_and_knowledge_interruption(mode):
    with TestClient(app) as client:
        token, uid = register(client, "episode_" + uuid4().hex[:8])
        h = auth(token)
        thread = client.post("/api/threads", headers=h, json={"mode": mode}).json()["id"]
        first = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": "我咳嗽三天，夜间明显，没有发热"})
        assert first.status_code == 200
        state = safety_store.load_state(uid, thread)
        original = state["episodes"]["self"]["id"]
        state["episodes"]["self"]["last_seen_at"] = "2025-01-01T00:00:00+00:00"
        with db.connection() as conn:
            conn.execute("UPDATE thread_safety_state SET state=? WHERE thread_id=?", (json.dumps(state), thread))
        reply = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": "我今天头痛"}).json()
        assert "继续上次" in reply["response"] and reply["intent_type"] == "consult"
        knowledge_reply = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": "什么是阴虚"}).json()
        assert knowledge_reply["trace"]["identity_mode"] == "knowledge"
        reply = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": "新的问题"}).json()
        assert "头痛" in reply["symptoms"] and "咳嗽" not in reply["symptoms"]
        context = safety_store.load_state(uid, thread)
        assert context["episodes"]["self"]["id"] != original
        assert "咳嗽" not in symptoms.describe(context["symptom_facts"], "self")


def test_memory_control_blocks_inflight_republication_and_is_account_scoped(monkeypatch):
    from app import main
    with TestClient(app) as client:
        token, uid = register(client, "control_" + uuid4().hex[:8])
        other, _ = register(client, "control_" + uuid4().hex[:8])
        h = auth(token)
        thread = client.post("/api/threads", headers=h, json={}).json()["id"]
        save = main._save_answer
        def race(run, result, trace):
            with db.connection() as conn:
                memory.bump_control(conn, uid, thread, False)
            return save(run, result, trace)
        monkeypatch.setattr(main, "_save_answer", race)
        result = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": "我咳嗽三天，没有发热"})
        assert result.status_code == 200
        assert not db.one("SELECT summary FROM profile_memories WHERE thread_id=?", (thread,))
        assert client.patch(f"/api/threads/{thread}/memory", headers=auth(other), json={"enabled": True, "revision": 1}).status_code == 404
        saved = memory.control(uid, thread)
        assert saved["enabled"] == 0 and saved["revision"] == 1
        assert client.patch(f"/api/threads/{thread}/memory", headers=h, json={"enabled": True, "revision": 0}).status_code == 409
        assert client.patch(f"/api/threads/{thread}/memory", headers=h, json={"enabled": True, "revision": 1}).status_code == 200
        assert len(client.get(f"/api/threads/{thread}/messages", headers=h).json()) == 2


def test_knowledge_followup_preserves_topics_without_patient_data(monkeypatch):
    captured = []
    async def retrieve(plan):
        captured.append(plan["semantic_query"])
        return [], {"warning": "", "selected": []}
    monkeypatch.setattr(knowledge, "retrieve_classics", retrieve)
    with TestClient(app) as client:
        token, uid = register(client, "topics_" + uuid4().hex[:8])
        h = auth(token)
        thread = client.post("/api/threads", headers=h, json={}).json()["id"]
        client.patch(f"/api/profiles/{uid}", headers=h, json={"history": "private-marker"})
        for question in ("什么是阴虚", "我咳嗽三天，没有发热", "它和阳虚有什么区别", "有哪些古籍出处"):
            response = client.post("/api/chat", headers=h, json={"thread_id": thread, "message": question})
            assert response.status_code == 200
        assert "阴虚" in captured[-1] and "阳虚" in captured[-1]
        assert all("private-marker" not in q and "咳嗽" not in q for q in captured)
        assert safety_store.load_state(uid, thread)["subject"] == "self"


def test_other_discomfort_denial_is_not_a_knowledge_pronoun():
    assert not knowledge.is_followup("三天，未有其它不适")
    assert knowledge.is_followup("它与阳虚有什么区别")


def test_cancelled_knowledge_turn_does_not_update_topics(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered = threading.Event()
    async def blocked(state):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(knowledge, "answer_knowledge", blocked)
    async def retrieve(plan):
        return [], {"warning": ""}
    monkeypatch.setattr(knowledge, "retrieve_classics", retrieve)
    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        token, _ = register(client, "cancel_topic_" + uuid4().hex[:8])
        h = auth(token)
        thread = client.post("/api/threads", headers=h, json={}).json()["id"]
        run_id = str(uuid4())
        pending = executor.submit(client.post, "/api/chat", headers=h,
                                  json={"thread_id": thread, "run_id": run_id, "message": "什么是阴虚"})
        assert entered.wait(5)
        assert client.post(f"/api/runs/{run_id}/cancel", headers=h).json()["status"] == "cancelled"
        assert pending.result(timeout=5).json()["cancelled"]
        assert not db.one("SELECT state FROM knowledge_topics WHERE thread_id=?", (thread,))


def test_checkpoint_pruning_keeps_latest_success_and_active_runs(memory_db, tmp_path, monkeypatch):
    from app import checkpoint_retention
    path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(checkpoint_retention, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("QIWEN_CHECKPOINT_SUCCESS_KEEP", "1")
    with db.connection() as conn:
        for i, status in enumerate(("completed", "completed", "running")):
            conn.execute("INSERT INTO chat_runs(id,thread_id,user_id,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                         (str(i), "current", "alice", status, str(i), str(i)))
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE checkpoints(thread_id TEXT)")
        conn.executemany("INSERT INTO checkpoints VALUES(?)", [(str(i),) for i in range(3)])
    checkpoint_retention.prune()
    with sqlite3.connect(path) as conn:
        assert {r[0] for r in conn.execute("SELECT thread_id FROM checkpoints")} == {"1", "2"}
