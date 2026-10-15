import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import model, workflow, safety_store
from app.main import app
from app.requirements import is_requirements_query
from test_api import register, auth
from test_model import configure


def test_provider_failure_keeps_http_status_and_omits_secret_body_headers_url(monkeypatch):
    secret = "synthetic-private-provider-detail"
    configure(monkeypatch, lambda request: httpx.Response(500, json={"error": secret}))
    monkeypatch.setenv("QIWEN_MODEL_SAFETY_PREFLIGHT_RETRIES", "0")

    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            with pytest.raises(model.ModelServiceError) as failure:
                await model.acomplete("system", "test", role="safety_preflight")
            assert failure.value.details == {"category": "upstream_unavailable", "http_status": 500}
            assert secret not in str(failure.value)
            call = context.calls[0]
            assert call["failure"] == call["attempts"][0]["failure"] == failure.value.details
            assert secret not in json.dumps(call)
            assert "stub.invalid" not in json.dumps(call)
        finally:
            model.invocation.reset(token)

    asyncio.run(run())


def test_error_details_and_read_only_lookup_do_not_expose_or_require_model_input(monkeypatch):
    error = model.ModelServiceError({"category": "private-value", "url": "private-value", "http_status": True})
    assert error.details == {"category": "unavailable"}
    def forbidden(state):
        raise AssertionError("资料查询不应构造模型上下文")
    monkeypatch.setattr(workflow, "_context", forbidden)
    assert workflow.context_budget({"intent_type": "requirements"}) == {"context_chars": 0}


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_requirements_query_survives_outage_without_new_model_call_and_preserves_pending(monkeypatch, mode):
    calls = []

    async def unavailable(*args, **kwargs):
        calls.append(1)
        raise model.ModelServiceError({"category": "upstream_unavailable", "http_status": 500})

    with TestClient(app) as client:
        token, uid = register(client, "outage_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{uid}", headers=h, json={"allergies": "青霉素"})
        tid = client.post("/api/threads", headers=h, json={"mode": mode}).json()["id"]
        first = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "本人腹胀三天，怕冷、大便稀"}).json()
        assert first["intent_type"] == "diagnose"
        monkeypatch.setattr(workflow, "assess", unavailable)
        failed = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "继续分析"}).json()
        assert failed["flow"]["status"] == "unavailable"
        assert "HTTP 500" in failed["response"] and "不是在要求你补充病情" in failed["response"]
        assert failed["message"]["prescription"] is None
        assert not {"bianzheng", "formula", "herb"} & {event["stage"] for event in failed["trace"]["events"]}
        before = safety_store.load_state(uid, tid)["clinical"]["self"]
        for _ in range(2):
            lookup = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "还需要补充什么内容才可以生成处方和辨证结果"}).json()
            assert lookup["intent_type"] == "requirements" and lookup["flow"]["status"] == "answered"
            assert not lookup["flow"]["steps"] and not lookup["trace"]["model_calls"]
            assert lookup["message"]["prescription"] is None
            assert "腹胀" in lookup["response"] and "青霉素" in lookup["response"]
            assert "模型服务不可用" in lookup["response"]
            assert len(calls) == 1
        after = safety_store.load_state(uid, tid)["clinical"]["self"]
        assert after == before
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert any(row.get("trace", {}).get("presentation", {}).get("status") == "unavailable" for row in history if row["role"] == "assistant")
        # A new symptom is not a read-only requirements request.
        new = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "还需要补充什么资料，我今天头痛"}).json()
        assert new["flow"]["status"] == "unavailable" and len(calls) == 2
        urgent = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "我现在胸痛，还需要补充什么资料"}).json()
        assert urgent["flow"]["status"] == "blocked" and len(calls) == 2


@pytest.mark.parametrize("text,expected", [("还需要补充什么内容才可以生成处方和辨证结果", True),
                                          ("还缺哪些资料", True), ("我今天头痛，还缺哪些资料", False),
                                          ("请直接生成处方", False)])
def test_requirements_detection_does_not_consume_new_symptoms(text, expected):
    assert is_requirements_query({"message": text, "safety_state": {"subject": "self"}}) is expected
