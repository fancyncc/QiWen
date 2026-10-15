import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import workflow
from app.main import app
from app.prescription import migrate_plan
from app.workflow_presentation import build_presentation, fixed_sections
from test_api import register, auth


def test_fixed_plan_is_rule_checked_and_structured_sections_survive_history(monkeypatch):
    original = workflow._make_plan

    async def plan(state, *, revision=False):
        result = await original(state, revision=revision)
        return {**result, "non_drug_advice": ["测试日常建议。"], "medical_guidance": ["测试面诊指引。"]}

    monkeypatch.setattr(workflow, "_make_plan", plan)
    with TestClient(app) as client:
        token, _ = register(client, "fixed_" + uuid4().hex[:8])
        h = auth(token)
        result = client.post("/api/chat", headers=h, json={"mode": "fixed", "message": "本人失眠三天，白天疲劳，不用查古籍"})
        assert result.status_code == 200
        data = result.json()
        stages = [event["stage"] for event in data["trace"]["events"] if event["status"] == "start"]
        assert "reviewer" not in stages and "formula" in stages and "safety" in stages
        assert data["flow"]["mode"] == "fixed"
        assert next(step for step in data["flow"]["steps"] if step["stage"] == "search")["status"] == "skipped"
        assert [section["id"] for section in data["flow"]["sections"]] == ["diagnosis", "medication", "non_drug", "medical_guidance", "disclaimer"]
        assert "测试日常建议" in data["response"] and "测试面诊指引" in data["response"]
        history = client.get(f'/api/threads/{data["thread_id"]}/messages', headers=h).json()
        assert history[-1]["trace"]["presentation"] == data["flow"]


def test_fixed_clarification_stops_before_plan_and_validation(monkeypatch):
    async def clarify(state):
        return {"intent_type": "consult", "clarification_stage": "bianzheng", "response": "请补充主要不适。",
                "pending_question": "请补充主要不适。", "pending_context": state.get("episode_text", "")}

    monkeypatch.setattr(workflow, "bianzheng", clarify)
    with TestClient(app) as client:
        token, _ = register(client, "fixed_pause_" + uuid4().hex[:8])
        data = client.post("/api/chat", headers=auth(token), json={"mode": "fixed", "message": "本人失眠三天，白天疲劳，不用查古籍"}).json()
        assert data["intent_type"] == "consult" and data["flow"]["status"] == "awaiting_input"
        assert "sections" not in data["flow"]
        statuses = {step["stage"]: step["status"] for step in data["flow"]["steps"]}
        assert statuses["bianzheng"] == "waiting"
        assert statuses["formula"] == statuses["safety"] == "not_run"
        assert data["message"]["prescription"] is None


@pytest.mark.parametrize("herbs,retries,expected", [
    ([], 0, "not_checked"), (["茯苓"], 0, "passed"), (["甘草", "海藻"], 0, "conflict_retrying"),
    (["甘草", "海藻"], workflow.MAX_SAFETY_REVISIONS, "conflict_unresolved"), (["没有收录的药材"], 0, "unknown"),
])
def test_validation_outcomes_distinguish_skips_conflicts_and_unknown_data(herbs, retries, expected):
    with TestClient(app):
        state = {"treatment_plan": migrate_plan({"name": "规则测试", "herbs": herbs}), "profile": {}, "safety_retry_count": retries}
        result = workflow.fixed_safety(state)
        assert result["compatibility_check"]["status"] == expected
        if expected != "passed":
            assert not result["prescription_review"]["approved"]


def test_status_comes_from_intent_and_events_not_diagnosis_words():
    state = {"workflow_mode": "fixed", "intent_type": "consult", "response": "正文含辨证完成四个字，仍是追问。"}
    flow = build_presentation(state, {"events": [{"stage": "inquiry", "status": "done"}]})
    assert flow["status"] == "awaiting_input" and "sections" not in flow
    assert flow["steps"][0]["status"] == "waiting"
    assert all(step["status"] == "not_run" for step in flow["steps"][1:])


def test_guidance_and_summary_cannot_leak_unverified_dosing():
    sections, _ = fixed_sections({"diagnosis": "初步分析。", "diagnosis_summary": "自行增加9g",
                                  "non_drug_advice": ["测试生活建议。", "每日水煎服"], "medical_guidance": ["服用3克"]})
    assert "9g" not in sections[0]["content"]
    assert "水煎服" not in sections[2]["content"] and "3克" not in sections[3]["content"]


def test_legacy_summary_keeps_initial_direction_and_details_remain_available():
    sections, details = fixed_sections({"response": "## 辨证思路\n已报告的症状。\n\n旧记录的初步辨证方向。\n\n更详细的分析。\n## 安全校验\n内部校验记录。"}, legacy=True)
    assert "初步辨证方向" in sections[0]["content"]
    assert "更详细的分析" in details and "内部校验记录" not in details


@pytest.mark.parametrize("status", ["need_user_input", "completed"])
def test_fixed_assessment_returns_explicit_clarification_without_treatment(monkeypatch, status):
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")

    async def assess(*args, **kwargs):
        assert kwargs["schema"] is workflow.FixedAssessment
        return {"status": status, "conclusion": "仅供内部候选整理", "summary": "", "questions": ["需要确认的测试问题？"]}

    monkeypatch.setattr(workflow, "acomplete_json", assess)
    result = asyncio.run(workflow.bianzheng({"episode_text": "已有症状", "symptoms": ["失眠"]}))
    assert result["intent_type"] == "consult" and result["diagnosis"] == "仅供内部候选整理"
    assert result["response"].endswith("1. 需要确认的测试问题？")
    assert "初步辨证" in result["response"]


@pytest.mark.parametrize("answer", ["没有", "不清楚"])
def test_fixed_clarification_saved_and_short_answer_resumes_analysis(monkeypatch, answer):
    from app import safety_store

    original = workflow.bianzheng
    calls = []
    question = "目前是否伴有口渴？"

    async def assess(system, context, **kwargs):
        calls.append(context)
        if len(calls) == 1:
            return {"status": "need_user_input", "conclusion": "", "summary": "", "questions": [question]}
        assert "上一轮问题与本轮回答" in context and question in context
        assert f'"answer": "{answer}"' in context and '"answer_message_id": "' in context
        assert "失眠" in context and "疲劳" in context
        return {"status": "completed", "conclusion": "基于已知资料的初步分析，未提供的资料保留未知。", "summary": "初步分析。", "questions": []}

    async def live_assessment(state):
        # Exercise the real assessment contract; all other nodes remain offline.
        with monkeypatch.context() as scoped:
            scoped.setattr(workflow, "MODEL_MODE", "openai_compatible")
            return await original(state)

    monkeypatch.setattr(workflow, "acomplete_json", assess)
    monkeypatch.setattr(workflow, "bianzheng", live_assessment)
    with TestClient(app) as client:
        token, user_id = register(client, "resume_inquiry_" + uuid4().hex[:8])
        h = auth(token)
        first = client.post("/api/chat", headers=h, json={"mode": "fixed", "message": "本人失眠三天，白天疲劳，不用查古籍"}).json()
        tid = first["thread_id"]
        assert first["flow"]["status"] == "awaiting_input"
        saved = safety_store.load_state(user_id, tid)
        assert saved["clinical"]["self"]["pending_question"] == "1. " + question
        assert "formula" not in {event["stage"] for event in first["trace"]["events"]}
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert history[-1]["trace"]["presentation"]["status"] == "awaiting_input"
        second = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": answer}).json()
        assert len(calls) == 2
        assert second["intent_type"] == "diagnose"
        assert second["flow"]["status"] != "awaiting_input"
        assert {"formula", "safety"} <= {event["stage"] for event in second["trace"]["events"]}
        assert safety_store.load_state(user_id, tid)["clinical"]["self"]["pending_question"] == ""
