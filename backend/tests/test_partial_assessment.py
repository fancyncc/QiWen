"""Incomplete clinical information must support analysis plus real follow-up."""
import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import model, workflow, safety_store
from app.agents import multi
from app.agents.runtime import run_agent_graph
from app.consultation_followup import needs_patient_answer, followup_response, validate_followup
from app.main import app
from test_api import register, auth


ANALYSIS = "腹胀、大便稀、怕冷提示需从运化与寒热方面作初步分析；无舌质淡胖或齿痕、小便正常，不支持把这些作为脾肾阳虚依据。具体证型仍待区分。"
QUESTION = "平时是否容易精神疲乏，休息后有无改善？"
STALLED = "根据腹胀、大便稀、怕冷初步考虑相关辨证方向。但由于缺乏舌象具体描述和其他信息（如是否有精神疲乏、面色苍白等），目前尚不能完全确定证候方向。"


def artifact(**changes):
    return {"status": "completed", "information_gap": "optional", "question": QUESTION,
            "conclusion": ANALYSIS, "summary": ANALYSIS, "uncertainties": ["脉象未知"], **changes}


def test_optional_explicit_question_waits_but_unknown_tongue_pulse_can_finish():
    assert needs_patient_answer(artifact())
    assert not needs_patient_answer(artifact(status="need_user_input", question=""))
    assert not needs_patient_answer(artifact(information_gap="evidence"))


def test_stalled_screenshot_answer_gets_one_contextual_repair():
    calls = []
    async def repair(feedback):
        calls.append(feedback)
        assert "不能确定的终点" in feedback or "无法确定的终点" in feedback
        assert STALLED in feedback
        return artifact()
    corrected = asyncio.run(validate_followup({}, artifact(question="", conclusion=STALLED), repair, assessment=True))
    assert len(calls) == 1 and needs_patient_answer(corrected)
    assert corrected["conclusion"] == ANALYSIS


def test_medication_evidence_gaps_do_not_trigger_clinical_repair():
    async def forbidden(_):
        raise AssertionError("Evidence limitations must not become patient questions")
    value = artifact(question="", conclusion=STALLED, information_gap="evidence")
    assert asyncio.run(validate_followup({}, value, forbidden, assessment=True)) == value


def test_food_preference_is_not_promoted_to_syndrome_counterevidence():
    calls = []
    async def repair(feedback):
        calls.append(feedback)
        assert "饮食偏好" in feedback
        return artifact()
    corrected = asyncio.run(validate_followup({}, artifact(conclusion="怕冷提示需分析寒热。但用户偏好凉性食物，这与上述辨证方向矛盾。"),
                                             repair, assessment=True))
    assert len(calls) == 1 and corrected["conclusion"] == ANALYSIS
    assert corrected["question"] == QUESTION


def test_preference_correction_is_bounded_and_rejects_persistent_false_inference():
    value = artifact(conclusion="用户偏好凉性食物，是脾胃虚寒的反证。")
    calls = []
    async def still_wrong(feedback):
        calls.append(feedback)
        return value
    with pytest.raises(model.InvalidModelOutput):
        asyncio.run(validate_followup({}, value, still_wrong, assessment=True))
    assert len(calls) == 1


def test_remaining_preference_error_is_removed_without_losing_analysis_or_questions():
    conclusion = ANALYSIS + "但用户偏好凉性食物，与上述辨证方向矛盾。"
    value = artifact(conclusion=conclusion, summary=conclusion)
    async def still_wrong(_):
        return value
    corrected = asyncio.run(validate_followup({}, value, still_wrong, assessment=True))
    assert corrected["conclusion"] == ANALYSIS and corrected["summary"] == ANALYSIS
    assert corrected["question"] == QUESTION and corrected["status"] == value["status"]
    assert corrected["uncertainties"] == value["uncertainties"]


def test_preference_limitation_is_allowed_without_regeneration():
    value = artifact(conclusion="偏好凉性食物不能作为寒热证候的反证；根据实际症状作初步分析。")
    async def forbidden(_):
        raise AssertionError("A factual limitation should remain visible")
    assert asyncio.run(validate_followup({}, value, forbidden, assessment=True)) == value


def test_known_complaints_get_analysis_before_questions_even_without_classics():
    value = artifact(conclusion="已知腹胀、怕冷、大便稀。没有直接相关的古籍。为了进一步辨证，请补充精神和食欲情况。")
    calls = []
    async def repair(feedback):
        calls.append(feedback)
        assert "古籍未命中不代表不能分析" in feedback
        return artifact()
    corrected = asyncio.run(validate_followup({"symptoms": ["腹胀", "怕冷", "大便稀"]}, value, repair, assessment=True))
    assert len(calls) == 1 and corrected["conclusion"] == ANALYSIS
    assert corrected["question"] == QUESTION


def test_dosing_is_not_shown_in_preliminary_followup():
    value = artifact(conclusion="建议甘草10g，每日服用")
    assert followup_response({}, value, assessment=True) == QUESTION


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_partial_analysis_question_is_visible_saved_and_resumes(mode, monkeypatch):
    original = multi.run_agent_graph
    calls = []
    async def ask(**kwargs):
        run = await original(**kwargs)
        if kwargs["role"] == "bianzheng":
            calls.append(kwargs["state"])
            run.output.update(artifact(question="" if kwargs["state"].get("answered_question") else QUESTION))
        return run
    async def fixed(state):
        calls.append(state)
        value = artifact(question="" if state.get("answered_question") else QUESTION)
        if needs_patient_answer(value):
            return {"intent_type": "consult", "clarification_stage": "bianzheng", "diagnosis": ANALYSIS,
                    "response": followup_response(state, value, assessment=True), "pending_question": QUESTION,
                    "pending_context": state["episode_text"]}
        return {"diagnosis": ANALYSIS}
    monkeypatch.setattr(multi, "run_agent_graph", ask)
    if mode == "fixed":
        monkeypatch.setattr(workflow, "bianzheng", fixed)
    with TestClient(app) as client:
        token, uid = register(client, "partial_" + uuid4().hex[:8])
        h = auth(token)
        data = client.post("/api/chat", headers=h, json={"mode": mode,
            "message": "本人肚子胀、怕冷、大便稀一个星期，没有舌质淡胖或齿痕，小便正常"}).json()
        tid = data["thread_id"]
        assert data["flow"]["status"] == "awaiting_input"
        assert ANALYSIS in data["response"] and QUESTION in data["response"]
        assert data["message"]["prescription"] is None
        saved = safety_store.load_state(uid, tid)["clinical"]["self"]
        assert saved["pending_question"] == QUESTION and "初步辨证" not in saved["pending_question"]
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert history[-1]["content"] == data["response"]
        resumed = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "不清楚"}).json()
        assert resumed["intent_type"] == "diagnose"
        assert QUESTION not in resumed["response"]
        assert calls[-1]["inquiry_answers"][-1]["answer"] == "不清楚"


def test_fixed_real_assessment_preserves_analysis_while_waiting(monkeypatch):
    monkeypatch.setattr(workflow, "MODEL_MODE", "live")
    async def complete(*args, **kwargs):
        return {"status": "completed", "information_gap": "optional", "conclusion": ANALYSIS,
                "summary": ANALYSIS, "questions": [QUESTION]}
    monkeypatch.setattr(workflow, "acomplete_json", complete)
    result = asyncio.run(workflow.bianzheng({"episode_text": "本人腹胀一周"}))
    assert result["diagnosis"] == ANALYSIS
    assert result["response"].endswith(QUESTION) and ANALYSIS in result["response"]
    assert result["pending_question"].endswith(QUESTION)
    assert "初步辨证" not in result["pending_question"]


def test_multi_runtime_repairs_stalled_assessment_before_release(monkeypatch):
    calls = []
    async def complete(*args, **kwargs):
        calls.append(kwargs)
        return artifact()
    monkeypatch.setattr(model, "acomplete_json", complete)
    result = asyncio.run(run_agent_graph(name="辨证 Agent", role="bianzheng", task=json.dumps({}), state={},
        tools=[], output_schema=multi.BianzhengArtifact, max_steps=1,
        mock_planner=lambda *_: {"final": artifact(question="", conclusion=STALLED)}))
    assert len(calls) == 1 and result.output["question"] == QUESTION
