"""Invalid provider formatting must recover once without weakening safety."""
import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import model, preflight, workflow
from app.main import app
from app.safety_context import make_context
from test_api import register, auth
from test_model import configure, output


VALID = {"decision": "allow", "risk_categories": [], "reason": "未发现新增风险", "required_questions": [], "observations": []}


def test_single_json_fence_is_accepted_but_clinical_schema_is_unchanged(monkeypatch):
    configure(monkeypatch, lambda _: output('```json\n' + json.dumps(VALID) + '\n```'))
    result = asyncio.run(model.acomplete_json("system", "synthetic", role="safety_preflight", schema=model.SafetyPreflight))
    assert result == VALID


@pytest.mark.parametrize("invalid", [
    '说明文字\n' + json.dumps(VALID),
    json.dumps(VALID) + '\n' + json.dumps(VALID),
    json.dumps({**VALID, "decision": "unknown"}),
    json.dumps({**VALID, "extra": "private-output"}),
])
def test_json_compatibility_does_not_extract_prose_or_accept_invalid_values(monkeypatch, invalid):
    configure(monkeypatch, lambda _: output(invalid))
    with pytest.raises(model.InvalidModelOutput):
        asyncio.run(model.acomplete_json("system", "synthetic", role="safety_preflight", schema=model.SafetyPreflight))


def test_invalid_output_diagnostics_include_only_schema_locations_not_private_content(monkeypatch):
    secret = "private-synthetic-output"
    configure(monkeypatch, lambda _: output(json.dumps({**VALID, "decision": secret, secret: secret})))
    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            with pytest.raises(model.InvalidModelOutput) as failure:
                await model.acomplete_json("system", "synthetic", role="safety_preflight", schema=model.SafetyPreflight)
            diagnostic = context.calls[0]["validation"]
            assert diagnostic == failure.value.validation
            assert diagnostic["kind"] == "schema"
            assert {tuple(issue["path"]) for issue in diagnostic["issues"]} == {("decision",), ("unknown_field",)}
            assert secret not in json.dumps(context.calls) and secret not in str(failure.value)
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["fixed", "multi"])
@pytest.mark.parametrize("recovers", [True, False])
def test_both_complete_workflows_recover_once_or_stop_before_experts(monkeypatch, mode, recovers):
    calls = []
    async def complete(policy, payload, **kwargs):
        calls.append((policy, payload))
        if len(calls) == 1 or not recovers:
            raise model.InvalidModelOutput("invalid", {"kind": "schema", "issues": [{"path": ["observations", 0, "category"], "type": "literal_error"}]})
        assert calls[0][1] == payload
        assert "不要为了通过格式校验改成 allow" in policy
        return dict(VALID)
    async def live_preflight(text, profile, **kwargs):
        return await preflight.assess(text, profile, mock=False, context=kwargs.get("context"))
    monkeypatch.setattr(preflight, "acomplete_json", complete)
    monkeypatch.setattr(workflow, "assess", live_preflight)
    with TestClient(app) as client:
        token, uid = register(client, "format_recovery_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{uid}", headers=h, json={"allergies": "青霉素"})
        data = client.post("/api/chat", headers=h, json={"mode": mode, "message": "本人腹胀三天，怕冷、大便稀"}).json()
        assert len(calls) == 2
        stages = {event["stage"] for event in data["trace"]["events"]}
        if recovers:
            assert data["intent_type"] == "diagnose"
            assert {"pre_safety", "inquiry", "bianzheng", "formula", "safety"} <= stages
            if mode == "multi":
                assert {"herb", "reviewer"} <= stages
        else:
            assert data["flow"]["status"] == "unavailable"
            assert data["message"]["prescription"] is None
            assert not {"inquiry", "bianzheng", "formula", "herb"} & stages


def test_format_repair_does_not_allow_forged_sources_or_weaken_existing_urgent_rules(monkeypatch):
    calls = []
    async def complete(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise model.InvalidModelOutput("invalid")
        return {**VALID, "observations": [{"category": "allergy", "entity": "甘草", "subject": "self", "assertion": "present", "temporal_scope": "current", "quote": "对甘草过敏"}]}
    monkeypatch.setattr(preflight, "acomplete_json", complete)
    state, _ = make_context({}, "本人", "")
    with pytest.raises(RuntimeError, match="原始证据"):
        asyncio.run(preflight.assess("本人", {}, mock=False, context=state))
    assert len(calls) == 2
    async def allowed(*args, **kwargs):
        return dict(VALID)
    monkeypatch.setattr(preflight, "acomplete_json", allowed)
    state, _ = make_context({}, "本人现在胸痛", "")
    result = asyncio.run(preflight.assess("本人现在胸痛", {}, mock=False, context=state))
    assert result["decision"] == "block"


def test_provider_http_failure_is_not_retried_as_a_format_problem(monkeypatch):
    calls = []
    async def unavailable(*args, **kwargs):
        calls.append(1)
        raise model.ModelServiceError({"category": "upstream_unavailable", "http_status": 500})
    monkeypatch.setattr(preflight, "acomplete_json", unavailable)
    with pytest.raises(model.ModelServiceError):
        asyncio.run(preflight.model_preflight("{}"))
    assert len(calls) == 1
