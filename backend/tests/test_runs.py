import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import model, workflow
from app.db import one
from test_api import register, auth
from test_model import configure, output

CASES = json.loads((Path(__file__).resolve().parents[1] / "evaluation/safety_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["category"] for case in CASES])
def test_risk_gates_and_current_allergy(case):
    with TestClient(app) as client:
        token, user_id = register(client, "risk_" + uuid4().hex[:8])
        result = client.post("/api/chat", headers=auth(token), json={"message": case["input"]})
        assert result.status_code == 200, result.text
        data = result.json()
        preflight = data["trace"]["preflight"]
        assert preflight["decision"] == case["decision"]
        assert case["category"] in preflight["risk_categories"]
        if case["decision"] != "allow":
            assert "方剂知识示例" not in data["response"]
            assert not any(event["stage"] == "formula" for event in data["trace"]["events"])
        if case.get("hide_candidate"):
            assert "药材清单已隐藏" in data["response"]
            assert "酸枣仁汤" not in data["response"]
            assert client.get(f"/api/profiles/{user_id}", headers=auth(token)).json()["allergies"] == ""


@pytest.mark.parametrize("failure", ["invalid", "timeout"])
def test_preflight_fails_closed(monkeypatch, failure):
    async def fail(*args, **kwargs):
        if failure == "timeout":
            raise TimeoutError
        raise RuntimeError("bad JSON")
    monkeypatch.setattr(workflow, "assess", fail)
    with TestClient(app) as client:
        token, _ = register(client, "fail_" + uuid4().hex[:8])
        data = client.post("/api/chat", headers=auth(token),
                           json={"message": "最近失眠三天，白天疲劳"}).json()
        assert data["intent_type"] == "safety_rejected"
        assert data["trace"]["preflight"]["risk_categories"] == ["preflight_unavailable"]
        assert not any(event["stage"] in {"bianzheng", "formula"} for event in data["trace"]["events"])


@pytest.mark.parametrize("blocked_role", ["safety_preflight", "formula"])
def test_cancel_ownership_model_cleanup_no_memory_or_success(monkeypatch, blocked_role):
    entered = threading.Event()
    closed = threading.Event()
    received = []
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.sleep(60)
            yield b"{}"
        async def aclose(self):
            closed.set()
    async def handler(request):
        role = json.loads(request.content)["model"]
        received.append(role)
        if role == blocked_role:
            return httpx.Response(200, stream=SlowBody())
        return output(json.dumps({"decision": "allow", "risk_categories": [], "reason": "stub", "required_questions": []})
                      if role == "safety_preflight" else "资料不足，不作确诊。")
    configure(monkeypatch, handler)
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
    for role in model.ROLES:
        monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), role)
    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        token, user_id = register(client, "stop_" + uuid4().hex[:8])
        other, _ = register(client, "other_" + uuid4().hex[:8])
        headers = auth(token)
        thread = client.post("/api/threads", headers=headers, json={"mode": "fixed"}).json()["id"]
        run_id = str(uuid4())
        future = executor.submit(client.post, "/api/chat", headers=headers,
                                 json={"run_id": run_id, "thread_id": thread, "message": "最近失眠三天，白天疲劳"})
        assert entered.wait(5)
        assert client.post(f"/api/runs/{run_id}/cancel", headers=auth(other)).status_code == 404
        assert client.get(f"/api/runs/{run_id}", headers=auth(other)).status_code == 404
        conflict = client.post("/api/chat", headers=headers, json={"thread_id": thread, "message": "新的问题"})
        assert conflict.status_code == 409
        result = client.post(f"/api/runs/{run_id}/cancel", headers=headers)
        assert result.json()["status"] == "cancelled", result.text
        assert closed.wait(2)
        assert future.result(timeout=5).json()["cancelled"]
        assert received.count(blocked_role) == 1
        assert "reviewer" not in received
        assert client.post(f"/api/runs/{run_id}/cancel", headers=headers).json()["status"] == "cancelled"
        messages = client.get(f"/api/threads/{thread}/messages", headers=headers).json()
        assert len(messages) == 2
        assert messages[-1]["intent_type"] == "cancelled"
        assert messages[-1]["trace"]["model_calls"][-1]["status"] == "cancelled"
        assert one("SELECT * FROM profile_memories WHERE thread_id=?", (thread,)) is None
        assert client.get(f"/api/profiles/{user_id}", headers=headers).json()["allergies"] == ""
        # A canceled partial checkpoint cannot affect the next turn.
        monkeypatch.setattr(workflow, "MODEL_MODE", "mock")
        next_turn = client.post("/api/chat", headers=headers, json={"thread_id": thread, "message": "你好"}).json()
        assert next_turn["intent_type"] == "chat"


def test_completed_result_is_authoritative_and_cancel_is_noop():
    with TestClient(app) as client:
        token, _ = register(client, "final_" + uuid4().hex[:8])
        headers = auth(token)
        with client.stream("POST", "/api/chat?stream=true", headers=headers, json={"message": "最近失眠"}) as response:
            events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        final = events[-1]
        assert final["done"] and final["message"]["content"] == final["response"]
        assert final["type"] == "answer.final"
        assert final["payload"]["response"] == final["response"]
        assert client.post(f'/api/runs/{final["run_id"]}/cancel', headers=headers).json()["status"] == "completed"
        saved = client.get(f'/api/threads/{final["thread_id"]}/messages', headers=headers).json()
        assert saved[-1]["trace"] == final["trace"]
        assert all("elapsed_ms" in event for event in final["trace"]["events"] if event["status"] == "done")


@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
def test_disconnect_cancels_model_and_leaves_stopped_history(monkeypatch, asgi_version):
    async def exercise(token, thread_id, run_id):
        started, closed = asyncio.Event(), asyncio.Event()
        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.sleep(60)
                yield b"{}"
            async def aclose(self):
                closed.set()
        configure(monkeypatch, lambda request: httpx.Response(200, stream=SlowBody()))
        monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": json.dumps({
            "run_id": run_id, "thread_id": thread_id, "message": "最近失眠三天，白天疲劳",
        }).encode(), "more_body": False})
        sent = []
        async def receive():
            return await incoming.get()
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": asgi_version},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/api/chat", "raw_path": b"/api/chat", "query_string": b"stream=true",
                 "root_path": "", "server": ("test", 80), "client": ("test", 1234),
                 "headers": [(b"authorization", ("Bearer " + token).encode()), (b"content-type", b"application/json")]}
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(started.wait(), 5)
        incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 5)
        await asyncio.wait_for(closed.wait(), 5)
        # The worker finalizer is retained independently of the disconnected response.
        for _ in range(30):
            if one("SELECT status FROM chat_runs WHERE id=?", (run_id,))["status"] == "cancelled":
                break
            await asyncio.sleep(0.05)
        assert one("SELECT status FROM chat_runs WHERE id=?", (run_id,))["status"] == "cancelled"
        assert one("SELECT summary FROM profile_memories WHERE thread_id=?", (thread_id,)) is None
        assert not any(b'"token"' in message.get("body", b"") for message in sent)
    with TestClient(app) as client:
        token, _ = register(client, "disconnect_" + uuid4().hex[:8])
        thread = client.post("/api/threads", headers=auth(token), json={"mode": "fixed"}).json()["id"]
        asyncio.run(exercise(token, thread, str(uuid4())))
        history = client.get(f"/api/threads/{thread}/messages", headers=auth(token)).json()
        assert history[-1]["intent_type"] == "cancelled"


def test_model_cannot_relax_rule_and_bad_plan_never_appears(monkeypatch):
    async def handler(request):
        role = json.loads(request.content)["model"]
        if role == "safety_preflight":
            return output('{"decision":"allow","risk_categories":[],"reason":"stub","required_questions":[]}')
        if role == "formula":
            return output('{"name":"rejected-secret-plan","herbs":"甘草"}')
        return output("资料不足，不作确诊。")
    configure(monkeypatch, handler)
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
    for role in model.ROLES:
        monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), role)
    with TestClient(app) as client:
        token, _ = register(client, "invalid_" + uuid4().hex[:8])
        pregnancy = client.post("/api/chat", headers=auth(token), json={"message": "怀孕期间失眠三天"}).json()
        assert pregnancy["trace"]["preflight"]["decision"] == "clarify"
        assert not any(event["stage"] == "formula" for event in pregnancy["trace"]["events"])
        failed = client.post("/api/chat", headers=auth(token), json={"message": "最近失眠三天，白天疲劳"})
        assert failed.status_code == 503
        thread = client.get("/api/threads", headers=auth(token)).json()[0]["id"]
        saved = client.get(f"/api/threads/{thread}/messages", headers=auth(token)).json()[-1]
        assert saved["intent_type"] == "failed"
        assert "rejected-secret-plan" not in saved["content"]
        assert saved["trace"]["model_calls"][-1]["status"] == "failed"


def test_cancel_before_summary_commit_preserves_previous_memory(monkeypatch):
    from app import runs
    entered = threading.Event()
    async def wait_before_write(state):
        entered.set()
        await asyncio.sleep(60)
        return {"context_summary": "must never be committed"}
    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        token, _ = register(client, "memory_" + uuid4().hex[:8])
        headers = auth(token)
        prior = client.post("/api/chat", headers=headers, json={"message": "最近失眠三天，白天疲劳"}).json()
        thread = prior["thread_id"]
        summary = one("SELECT summary FROM profile_memories WHERE thread_id=?", (thread,))["summary"]
        monkeypatch.setattr(workflow, "profile_write", wait_before_write)
        run_id = str(uuid4())
        future = executor.submit(client.post, "/api/chat", headers=headers,
                                 json={"run_id": run_id, "thread_id": thread, "message": "最近鼻塞三天，伴有怕冷"})
        assert entered.wait(5)
        client.post(f"/api/runs/{run_id}/cancel", headers=headers).raise_for_status()
        assert future.result(timeout=5).json()["cancelled"]
        assert one("SELECT summary FROM profile_memories WHERE thread_id=?", (thread,))["summary"] == summary
        assert run_id not in runs.active


def test_restart_recovers_interrupted_run():
    from app.db import connection, now
    with TestClient(app) as client:
        token, user_id = register(client, "restart_" + uuid4().hex[:8])
        thread = client.post("/api/threads", headers=auth(token), json={"mode": "fixed"}).json()["id"]
    run_id = str(uuid4())
    with connection() as db:
        db.execute("INSERT INTO chat_runs(id,thread_id,user_id,status,telemetry,created_at,updated_at) VALUES(?,?,?,'running',?,?,?)",
                   (run_id, thread, user_id, json.dumps({"model_calls": [{"role": "formula", "status": "running"}]}), now(), now()))
    with TestClient(app) as client:
        recovered = client.get(f"/api/runs/{run_id}", headers=auth(token)).json()
        assert recovered["status"] == "cancelled"
        assert recovered["telemetry"]["model_calls"][0]["status"] == "cancelled"
        assert client.get(f"/api/threads/{thread}/messages", headers=auth(token)).json()[-1]["intent_type"] == "cancelled"


def test_successful_provider_pipeline_persists_role_node_usage(monkeypatch):
    def handler(request):
        role = json.loads(request.content)["model"]
        replies = {
            "safety_preflight": {"decision": "allow", "risk_categories": [], "reason": "stub", "required_questions": []},
            "formula": {"name": "酸枣仁汤", "herbs": []},
            "reviewer": {"approved": True, "reason": "stub"},
        }
        from app.prescription import structured_herb
        replies["formula"]["herbs"] = [structured_herb(name) for name in ["酸枣仁", "茯苓", "知母", "川芎", "甘草"]]
        content = json.dumps(replies[role]) if role in replies else "资料不足，不作确诊。"
        return output(content, {"prompt_tokens": 12, "completion_tokens": 3})
    configure(monkeypatch, handler)
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
    for role in model.ROLES:
        monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), role)
    with TestClient(app) as client:
        token, _ = register(client, "usage_" + uuid4().hex[:8])
        headers = auth(token)
        result = client.post("/api/chat", headers=headers, json={"message": "最近失眠三天，白天疲劳", "mode": "multi"})
        assert result.status_code == 200, result.text
        data = result.json()
        calls = data["trace"]["model_calls"]
        assert {call["role"] for call in calls} == {"safety_preflight", "bianzheng", "formula", "reviewer"}
        assert sum(call["input_tokens"] for call in calls) == 48
        assert sum(call["output_tokens"] for call in calls) == 12
        assert all(call["node"] and call["status"] == "success" for call in calls)
        saved = client.get(f'/api/threads/{data["thread_id"]}/messages', headers=headers).json()[-1]
        assert saved["trace"]["model_calls"] == calls
