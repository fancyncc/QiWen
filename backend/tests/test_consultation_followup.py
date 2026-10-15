"""Both workflows must pause on patient questions and resume without losing answers."""
import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import workflow, safety_store
from app.agents import multi
from app.agents.runtime import AgentRun
from app.context import clinical_context
from app.consultation_followup import needs_patient_answer
from app.main import app
from app.workflow_presentation import build_presentation
from test_api import register, auth


@pytest.mark.parametrize("stage", ["bianzheng", "formula", "herb"])
@pytest.mark.parametrize("answer", ["没有", "不清楚"])
def test_multi_each_specialist_pauses_saves_question_and_resumes(monkeypatch, stage, answer):
    original = multi.run_agent_graph
    question = "本次发病前是否有明显诱因？"
    paused = False
    resumed_contexts = []

    async def ask(**kwargs):
        nonlocal paused
        if kwargs["state"].get("answered_question"):
            context = json.loads(kwargs["task"])["patient_context"]
            assert question in context and "上一轮问题与本轮回答" in context
            assert f'"answer": "{answer}"' in context
            resumed_contexts.append(kwargs["role"])
        run = await original(**kwargs)
        if kwargs["role"] == stage and not paused:
            paused = True
            # Even an accidentally successful status must honor a real question.
            run.output.update(status="approved" if stage == "herb" else "completed", question=question,
                              information_gap="patient")
        return run

    monkeypatch.setattr(multi, "run_agent_graph", ask)
    with TestClient(app) as client:
        token, uid = register(client, "multi_resume_" + uuid4().hex[:8])
        h = auth(token)
        first = client.post("/api/chat", headers=h, json={"mode": "multi", "message": "本人失眠三天，白天疲劳"}).json()
        tid = first["thread_id"]
        assert first["intent_type"] == "consult" and first["flow"]["status"] == "awaiting_input"
        assert first["response"].endswith(question)
        assert "初步辨证" in first["response"]
        assert first["message"]["prescription"] is None
        statuses = {step["stage"]: step["status"] for step in first["flow"]["steps"]}
        assert statuses[stage] == "waiting"
        order = ["bianzheng", "formula", "herb", "reviewer", "safety"]
        assert all(statuses[next_stage] == "not_run" for next_stage in order[order.index(stage) + 1:])
        assert safety_store.load_state(uid, tid)["clinical"]["self"]["pending_question"] == question
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert history[-1]["trace"]["presentation"]["status"] == "awaiting_input"
        second = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": answer}).json()
        assert second["intent_type"] == "diagnose"
        assert set(resumed_contexts) == {"bianzheng", "formula", "herb"}
        assert {"reviewer", "safety"} <= {event["stage"] for event in second["trace"]["events"]}
        saved = safety_store.load_state(uid, tid)["clinical"]["self"]
        assert saved["pending_question"] == ""
        assert saved["inquiry_answers"][-1]["question"] == question
        assert saved["inquiry_answers"][-1]["answer"] == answer


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_two_consecutive_questions_keep_both_answer_associations_and_clear_on_new_episode(monkeypatch, mode):
    original = multi.run_agent_graph
    captured = []
    questions = ["本次发病前是否有明显诱因？", "目前是否伴有口渴？"]

    def assessment(state):
        answers = state.get("inquiry_answers", [])
        # An ordinary initial inquiry may already be recorded; use our questions only.
        own = [answer for answer in answers if answer["question"] in questions]
        captured.append(clinical_context(state))
        question = questions[len(own)] if len(own) < 2 else ""
        return {"status": "need_user_input" if question else "completed", "conclusion": "初步辨证。",
                "question": question, "information_gap": "patient" if question else "none", "uncertainties": []}

    async def fixed(state):
        artifact = assessment(state)
        if artifact["question"]:
            return {"intent_type": "consult", "clarification_stage": "bianzheng", "response": artifact["question"],
                    "pending_question": artifact["question"], "pending_context": state["episode_text"]}
        return {"diagnosis": artifact["conclusion"]}

    async def ask(**kwargs):
        run = await original(**kwargs)
        if kwargs["role"] == "bianzheng":
            run.output.update(assessment(kwargs["state"]))
        return run

    monkeypatch.setattr(workflow, "bianzheng", fixed)
    monkeypatch.setattr(multi, "run_agent_graph", ask)
    with TestClient(app) as client:
        token, uid = register(client, "chain_" + uuid4().hex[:8])
        h = auth(token)
        first = client.post("/api/chat", headers=h, json={"mode": mode, "message": "本人失眠三天，白天疲劳"}).json()
        tid = first["thread_id"]
        for answer in ["没有", "不清楚"]:
            result = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": answer}).json()
        assert result["intent_type"] == "diagnose"
        assert all(question in captured[-1] for question in questions)
        saved = safety_store.load_state(uid, tid)["clinical"]["self"]["inquiry_answers"]
        assert [(a["question"], a["answer"]) for a in saved] == list(zip(questions, ["没有", "不清楚"]))
        # Starting a new episode must not reuse an old question's short denial.
        prior_calls = len(captured)
        new = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "新的问题，本人咳嗽三天，白天有痰"}).json()
        assert new["intent_type"] in {"consult", "safety_advisory"}
        if len(captured) > prior_calls:
            assert "inquiry-answers" not in captured[-1]
        else:
            # New-episode safety rechecks may pause before any expert is called.
            assert "bianzheng" not in {event["stage"] for event in new["trace"]["events"]}
        assert not safety_store.load_state(uid, tid)["clinical"]["self"]["inquiry_answers"]


@pytest.mark.parametrize("gap,expected", [("patient", True), ("none", True), ("optional", True), ("evidence", False)])
def test_explicit_information_gap_distinguishes_patient_questions_from_evidence(gap, expected):
    assert needs_patient_answer({"status": "need_user_input", "information_gap": gap, "question": "问题"}) is expected


@pytest.mark.parametrize("revision", [False, True])
def test_fixed_plan_question_pauses_before_safety(monkeypatch, revision):
    async def plan(state, **kwargs):
        return {"name": "待补充", "herbs": [], "information_gap": "patient", "questions": ["需要确认的病情信息？"]}

    monkeypatch.setattr(workflow, "_make_plan", plan)
    result = asyncio.run((workflow.revise_formula if revision else workflow.formula)({"episode_text": "失眠三天"}))
    assert result["intent_type"] == "consult" and result["clarification_stage"] == "formula"
    assert result["pending_context"] == "失眠三天" and not result["treatment_plan"]


def test_answer_context_excludes_other_patient_thread_and_episode():
    base = {"message": "本轮描述", "thread_id": "t", "episode_id": "e", "safety_state": {"subject": "self"}}
    answer = {"question": "正确问题", "answer": "正确回答", "answer_message_id": "m", "thread_id": "t", "episode_id": "e", "subject": "self"}
    foreign = [{**answer, field: value, "answer": "错误回答"} for field, value in [("thread_id", "other"), ("subject", "other:父亲"), ("episode_id", "old")]]
    context = clinical_context({**base, "inquiry_answers": [answer, *foreign]})
    assert "正确回答" in context and "错误回答" not in context


def test_revision_question_uses_actual_waiting_role_and_does_not_approve_previous_candidate():
    flow = build_presentation({"workflow_mode": "multi", "intent_type": "consult", "agent_stop": "need_user_input",
                               "clarification_stage": "formula", "compatibility_check": {"status": "passed", "dosage_status": "verified"}},
                              {"events": [{"stage": stage, "status": "done"} for stage in ["bianzheng", "formula", "herb", "reviewer", "safety", "formula"]]})
    statuses = {step["stage"]: step["status"] for step in flow["steps"]}
    assert statuses["formula"] == "waiting"
    assert all(statuses[stage] == "superseded" for stage in ["herb", "reviewer", "safety"])
    assert flow["compatibility"] == {"status": "not_checked"}


@pytest.mark.parametrize("artifact_status", ["need_user_input", "completed"])
def test_legacy_multi_question_stage_comes_from_execution_not_response_words(artifact_status):
    flow = build_presentation({"intent_type": "consult", "response": "与阶段判断无关的正文"},
                              {"mode": "multi", "events": [{"stage": stage, "status": "done"} for stage in ["inquiry", "bianzheng", "formula"]],
                               "agents": {"artifacts": {"formula": {"status": artifact_status}}}}, legacy=True)
    statuses = {step["stage"]: step["status"] for step in flow["steps"]}
    assert statuses["formula"] == "waiting" and statuses["inquiry"] == statuses["bianzheng"] == "done"


def test_herb_evidence_gap_stays_restricted_instead_of_asking_or_retrying_prescription(monkeypatch):
    async def evidence(**kwargs):
        return AgentRun({"status": "approved", "information_gap": "evidence", "question": "", "problems": [],
                         "required_actions": [], "evidence_ids": []}, [], [])

    monkeypatch.setattr(multi, "run_agent_graph", evidence)
    result = asyncio.run(multi.herb_agent({"message": "已有主诉", "treatment_plan": {}, "safety_state": {"subject": "self"}}))
    assert result["agent_stop"] == "" and "pending_question" not in result
    assert result["analysis_only"] and result["herb_review"]["status"] != "approved"
    assert workflow.safety_route({**result, "safety_errors": ["依据不足"]}) == "finish"


@pytest.mark.parametrize("message", ["失眠三天，白天疲劳", "本人现在胸痛", "本人有咳嗽，没有咳嗽"])
def test_missing_scope_urgent_and_conflicting_symptoms_still_stop_before_agents(monkeypatch, message):
    async def forbidden(**kwargs):
        raise AssertionError("前置资料未确认不得进入专家分析")

    monkeypatch.setattr(multi, "run_agent_graph", forbidden)
    with TestClient(app) as client:
        token, _ = register(client, "gate_" + uuid4().hex[:8])
        response = client.post("/api/chat", headers=auth(token), json={"mode": "multi", "message": message})
        assert response.status_code == 200
        assert response.json()["intent_type"] in {"consult", "safety_advisory", "safety_rejected"}
        assert response.json()["message"]["prescription"] is None
