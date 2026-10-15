"""Contextual answers and duplicate follow-up checks shared by both modes."""
import asyncio
import json

import pytest

from app import model, workflow
from app.agents.multi import BianzhengArtifact
from app.agents.runtime import run_agent_graph
from app.consultation_followup import answer_readings, repeated_questions, validate_followup
from app.context import clinical_context


Q1 = "请问您是否有以下情况：1. 舌质淡胖或有齿痕？2. 脉象沉迟无力？3. 饮食偏好温热食物还是凉性食物？"
Q2 = "1. 您是否有腰膝酸软？2. 您是否有小便清长？3. 您是否感到四肢不温？"


def state_with_answers(*pairs):
    state = {"thread_id": "t", "episode_id": "e", "message": pairs[-1][1] if pairs else "", "safety_state": {"subject": "self"}}
    state["inquiry_answers"] = [{"question": q, "answer": a, "thread_id": "t", "episode_id": "e",
                                 "subject": "self", "answer_message_id": f"m{i}"} for i, (q, a) in enumerate(pairs)]
    return state


def asking(*questions):
    return {"status": "need_user_input", "information_gap": "patient", "question": "\n".join(questions),
            "conclusion": "尚待分析", "summary": "", "uncertainties": [], "evidence_ids": []}


def test_screenshot_answers_are_interpreted_individually_without_changing_clinical_facts():
    state = state_with_answers((Q1, "舌质正常，但脉象存在沉迟无力，偏好凉性食物"), (Q2, "上述情况都存在"))
    readings = answer_readings(state)
    assert len(readings) == 6
    assert readings[0]["interpretation"] == "用户自述：舌质正常"
    assert readings[1]["interpretation"] == "用户自述：脉象存在沉迟无力"
    assert readings[2]["interpretation"] == "用户自述：偏好凉性食物"
    assert all(row["interpretation"] == "用户明确表示该项存在" for row in readings[3:])
    context = clinical_context(state)
    assert "answer-readings" in context and "上述情况都存在" in context
    assert "symptom_facts" not in state  # Questions never become confirmed symptom facts.
    remaining, repeats = repeated_questions(state, asking("1. 您是否有舌质淡胖或有齿痕？2. 您的饮食偏好是温热食物还是凉性食物？3. 您的小便颜色是否清长？"))
    assert not remaining and len(repeats) == 3


@pytest.mark.parametrize("answer,expected", [("上述情况都存在", "存在"), ("都没有", "不存在"), ("都不清楚", "未知")])
def test_explicit_distributive_reply_resolves_each_yes_no_item(answer, expected):
    rows = answer_readings(state_with_answers((Q2, answer)))
    assert len(rows) == 3
    assert all(expected in row["interpretation"] for row in rows)


def test_bare_no_and_choice_answers_are_not_distributed_over_a_questionnaire():
    assert answer_readings(state_with_answers((Q2, "没有"))) == []
    rows = answer_readings(state_with_answers((Q1, "上述情况都存在")))
    assert len(rows) == 2
    assert not any("food_preference" in row["topics"] for row in rows)
    assert answer_readings(state_with_answers((Q2, "第一项没有，其他不清楚"))) == []


def test_partial_reply_does_not_resolve_unanswered_items_or_new_urine_attributes():
    state = state_with_answers((Q2, "有腰膝酸软"))
    assert len(answer_readings(state)) == 1
    remaining, repeats = repeated_questions(state, asking(Q2))
    assert len(remaining) == 2 and len(repeats) == 1
    state = state_with_answers((Q2, "上述情况都存在"))
    remaining, repeats = repeated_questions(state, asking("小便次数是否频繁？", "腰膝酸软持续多久？", "现在是否还有四肢不温，与之前相比有变化吗？", "是否有腰膝酸软和口干？"))
    assert len(remaining) == 4 and not repeats


def test_answer_readings_cannot_leak_between_patients_threads_and_episodes():
    state = state_with_answers((Q2, "上述情况都存在"))
    own = state["inquiry_answers"][0]
    state["inquiry_answers"] = [{**own, key: value} for key, value in (("thread_id", "other"), ("subject", "other:父亲"), ("episode_id", "old"))]
    assert not answer_readings(state)
    assert not repeated_questions(state, asking(Q2))[1]


def test_duplicate_guard_keeps_new_questions_without_regenerating_or_releasing_plan():
    state = state_with_answers((Q2, "上述情况都存在"))
    async def forbidden(_):
        raise AssertionError("Mixed questions should only retain the unanswered item")
    artifact = asyncio.run(validate_followup(state, asking("您是否有腰膝酸软？", "目前是否有发热？"), forbidden))
    assert artifact["question"] == "目前是否有发热？"
    assert artifact["status"] == "need_user_input"


def test_repair_is_bounded_and_never_silently_promotes_an_invalid_artifact():
    state = state_with_answers((Q2, "上述情况都存在"))
    calls = []
    async def still_repeats(feedback):
        calls.append(feedback)
        return asking("您是否感到四肢不温？")
    with pytest.raises(model.InvalidModelOutput):
        asyncio.run(validate_followup(state, asking(Q2), still_repeats))
    assert len(calls) == 1


def test_multi_runtime_repairs_duplicate_using_existing_context_and_retains_completed_analysis(monkeypatch):
    state = state_with_answers((Q2, "上述情况都存在"))
    calls = []
    async def corrected(system, payload, **kwargs):
        calls.append(payload)
        assert "用户回答释义" in payload and "上述情况都存在" in payload
        return {**asking(), "status": "completed", "information_gap": "optional", "question": "", "conclusion": "仅作初步分析，保留不确定性。"}
    monkeypatch.setattr(model, "acomplete_json", corrected)
    result = asyncio.run(run_agent_graph(name="辨证 Agent", role="bianzheng", task=json.dumps({"patient_context": clinical_context(state)}, ensure_ascii=False),
                         state=state, tools=[], output_schema=BianzhengArtifact, mock_planner=lambda *_: {"final": asking(Q2)}, max_steps=1))
    assert len(calls) == 1
    assert result.output["status"] == "completed"
    assert result.output["conclusion"] == "仅作初步分析，保留不确定性。"


@pytest.mark.parametrize("role,schema", [("bianzheng", workflow.FixedAssessment), ("formula", workflow.FixedTreatment)])
def test_fixed_roles_share_followup_validation_and_preserve_other_output_fields(monkeypatch, role, schema):
    state = state_with_answers((Q2, "上述情况都存在"))
    calls = []
    async def complete(system, user, **kwargs):
        calls.append(user)
        if len(calls) == 1:
            return {"questions": ["您是否有小便清长？"], "information_gap": "patient", "name": "待分析", "herbs": []}
        assert "用户回答释义" in user
        return {"questions": [], "information_gap": "evidence", "name": "文献证据不足", "herbs": [], "non_drug_advice": ["原有建议"]}
    monkeypatch.setattr(workflow, "acomplete_json", complete)
    result = asyncio.run(workflow._complete_with_answers(state, "system", clinical_context(state), role=role, schema=schema))
    assert len(calls) == 2 and not result["questions"]
    assert result["non_drug_advice"] == ["原有建议"]
    assert result["information_gap"] == "evidence"
