from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents import multi
from app.agents.tools import search_formulas, SearchArgs
from app.safety import contains_dosing
from app import symptoms
from app import preflight
from app.safety_context import make_context, extract_facts
from test_api import register, auth


@pytest.mark.parametrize("formula_has_plan", [True, False])
def test_missing_tongue_pulse_and_classics_do_not_end_known_complaint_flow(monkeypatch, formula_has_plan):
    original = multi.run_agent_graph

    async def incomplete(**kwargs):
        run = await original(**kwargs)
        if kwargs["role"] == "bianzheng":
            run.output.update(status="need_user_input", conclusion="已知失眠、入睡困难、易惊醒；辨证方向尚需舌脉与伴随表现区分。",
                              uncertainties=["舌象尚未提供", "脉象尚未提供", "经典检索未命中"], question="请补充舌象、脉象。")
        if kwargs["role"] == "formula" and not formula_has_plan:
            run.output.update(status="need_user_input", plan=None, question="个人选方资料不足。")
        return run

    monkeypatch.setattr(multi, "run_agent_graph", incomplete)
    with TestClient(app) as client:
        token, uid = register(client, "preliminary_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{uid}", headers=h, json={"allergies": "青霉素"})
        tid = client.post("/api/threads", headers=h, json={"mode": "multi"}).json()["id"]
        for text in ["最近总是失眠多梦", "本人"]:
            assert client.post("/api/chat", headers=h, json={"thread_id": tid, "message": text}).status_code == 200
        for text in ["持续了有一个星期，入睡困难，易惊醒", "偶尔头晕，发病期情绪低落"]:
            result = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": text})
            assert result.status_code == 200
            data = result.json()
            assert data["intent_type"] == "diagnose"
            assert set(data["trace"]["agents"]["artifacts"]) == {"bianzheng", "formula", "herb"}
            assert "初步辨证" in data["response"] and "方剂知识对照" in data["response"] and "本草知识对照" in data["response"]
            assert "酸枣仁汤" in data["response"] and "功效：" in data["response"] and "知识来源：" in data["response"]
            assert "仍待补充" in data["response"]
            assert not contains_dosing(data["response"])
            assert data["message"]["prescription"] is None and not data["message"]["prescription_review"]["approved"]
            assert any(f.get("duration_text") == "一个星期" for f in data["trace"]["inquiry"]["facts"])
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert "本草知识对照" in history[-1]["content"]


@pytest.mark.parametrize("change", ["unconfirmed", "no_symptoms", "no_duration", "blocked", "pending_risk", "pending_symptom"])
def test_provisional_analysis_requires_resolved_scope_and_core_facts(change):
    state = {"safety_state": {"subject": "self", "subject_confirmed": True}, "symptoms": ["失眠"],
             "episode_text": "失眠一个星期", "preflight": {"decision": "allow"}}
    if change == "unconfirmed":
        state["safety_state"]["subject_confirmed"] = False
    elif change == "no_symptoms":
        state["symptoms"] = []
    elif change == "no_duration":
        state["episode_text"] = "失眠"
    elif change == "blocked":
        state["preflight"]["decision"] = "block"
    elif change == "pending_risk":
        state["preflight"].update(decision="clarify", question_keys=["medication_details"])
    else:
        state["symptom_pending"] = {"kind": "conflict"}
    assert not multi.can_analyze_provisionally(state)


def test_colloquial_week_duration_keeps_user_source():
    facts = symptoms.extract("我失眠", message_id="a")
    updates = symptoms.bind_details(facts, "持续了有一个星期", subject="self", message_id="b", thread_id="t")
    assert updates[0]["duration_text"] == "一个星期"
    assert updates[0]["source"]["quote"] == "持续了有一个星期"


def test_formula_search_handles_multiword_symptoms_and_marks_fallback():
    with TestClient(app):
        result = search_formulas({}, SearchArgs(query="失眠 易惊醒 情绪低落"))
        assert {row["name"] for row in result["formulas"]} >= {"酸枣仁汤", "归脾汤"}
        assert not result["symptom_fallback"]
        fallback = search_formulas({"symptoms": ["失眠"]}, SearchArgs(query="没有收录的方名"))
        assert fallback["symptom_fallback"] and fallback["warning"] and fallback["formulas"]
        assert not search_formulas({}, SearchArgs(query="%_"))["formulas"]


@pytest.mark.parametrize("reported_medication", [False, True])
def test_optional_profile_questions_do_not_block_knowledge_but_reported_medication_does(monkeypatch, reported_medication):
    import asyncio
    state, _ = make_context({}, "本人", "")
    state["facts"] = extract_facts("对青霉素过敏", subject="self")
    text = "每天使用甲氨蝶呤" if reported_medication else "偶尔头晕，发病期情绪低落"
    async def assess_model(*args, **kwargs):
        return {"decision": "clarify", "risk_categories": ["medication", "age"], "reason": "仍有资料未知",
                "required_questions": ["您的年龄？", "是否正在用药？"], "observations": ([{
                    "category": "medication", "entity": "甲氨蝶呤", "subject": "self", "assertion": "present",
                    "temporal_scope": "current", "quote": text}] if reported_medication else [{
                    "category": "constitution", "entity": "气虚体质", "subject": "self", "assertion": "present",
                    "temporal_scope": "current", "quote": "模型根据头晕推测气虚体质"}])}
    monkeypatch.setattr(preflight, "acomplete_json", assess_model)
    result = asyncio.run(preflight.assess(text, {}, mock=False, context=state))
    assert result["analysis_allowed"] is not reported_medication
    if reported_medication:
        assert "medication_details" in result["question_keys"]
    else:
        assert not result["observations"] and result["review_questions"]
