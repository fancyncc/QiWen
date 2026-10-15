"""Regression for repeated and endlessly expanding specialist questionnaires."""
import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import model, workflow, safety_store, symptoms
from app.agents.runtime import run_agent_graph
from app.context_budget import budget_request
from app.agents import multi
from app.consultation_followup import (
    after_inquiry_schema, answer_readings, convergence_reason, inquiry_progress,
    observation_records, observation_ledger, repeated_questions, validate_followup,
)
from app.context import clinical_context
from app.main import app
from test_api import auth, register


def ready(*pairs):
    return {"thread_id": "t", "episode_id": "e", "message": "", "symptoms": ["腹胀", "怕冷", "大便稀"],
            "safety_state": {"subject": "self", "subject_confirmed": True}, "preflight": {"decision": "allow"},
            "inquiry_answers": [{"thread_id": "t", "episode_id": "e", "subject": "self", "stage": "bianzheng",
                                 "answer_message_id": str(i), "question": q, "answer": a} for i, (q, a) in enumerate(pairs)]}


def ask(q, gap="optional", **extra):
    return {"status": "need_user_input", "information_gap": gap, "question": q,
            "conclusion": "已知症状提示需分析脾胃运化及寒热虚实，尚待区分。", **extra}


def test_screenshot_coating_colour_is_known_but_new_thickness_and_mood_are_not():
    state = ready(("舌苔、脉象及是否精神疲乏？", "舌苔泛白，脉象无力，有精神疲乏的情况"))
    remaining, repeats = repeated_questions(state, ask("1. 舌苔的颜色是什么样的？2. 您的舌苔厚还是薄？3. 心情是否低落？"))
    assert repeats == ["舌苔的颜色是什么样的？"]
    assert remaining == ["您的舌苔厚还是薄？", "心情是否低落？"]


def test_volunteered_information_is_scoped_persistable_and_used_for_deduplication():
    state = ready()
    state["inquiry_observations"] = observation_records(state, "舌苔泛白，容易疲劳，夜间尿频", "m")
    state["message"] = "不清楚"
    remaining, repeats = repeated_questions(state, ask("舌苔的颜色是什么样的？\n是否容易疲劳？"))
    assert not remaining and len(repeats) == 2
    # Different patients/episodes and historical or hypothetical statements do not resolve current slots.
    foreign = {**state, "episode_id": "other"}
    assert not observation_ledger(foreign)
    assert not observation_records(state, "父亲舌苔泛白；如果面色苍白；去年容易疲劳", "m")


def test_short_yes_to_alternative_colours_does_not_invent_both_findings():
    state = ready(("您是否有面色苍白或萎黄？", "有"))
    reading = answer_readings(state)[0]
    assert reading["status"] == "partial"
    assert repeated_questions(state, ask("您是否有面色苍白或萎黄？"))[1]
    assert repeated_questions(state, ask("请观察您的面色，是偏苍白还是萎黄？"))[0]
    assert "partial" in clinical_context(state)


def test_shared_budget_is_per_answer_message_and_excludes_initial_intake():
    state = ready(("初始主诉？", "腹胀"), ("精神状态？", "不清楚"))
    state["inquiry_answers"][0]["stage"] = "inquiry"
    state["inquiry_answers"].append(dict(state["inquiry_answers"][1]))
    assert inquiry_progress(state)["answered_rounds"] == 1
    assert convergence_reason(state, ask("面色如何？"))
    assert not convergence_reason(state, ask("腹痛是否遇冷加重？", "patient", followup_reason="用于区分寒热并改变候选选择"))
    state["inquiry_answers"].append({**state["inquiry_answers"][1], "stage": "formula", "answer_message_id": "next"})
    assert convergence_reason(state, ask("还有其他普通症状吗？", "patient", followup_reason="区分候选")) == "已完成两轮常规追问"


@pytest.mark.parametrize("change", ["scope", "urgent", "conflict", "safety_question"])
def test_budget_never_overrides_identity_safety_or_conflict_gates(change):
    state = ready(("面色？", "不清楚"), ("舌苔？", "不清楚"))
    question = "食欲如何？"
    if change == "scope":
        state["safety_state"]["subject_confirmed"] = False
    elif change == "urgent":
        state["preflight"] = {"decision": "block"}
    elif change == "conflict":
        state["symptom_pending"] = {"kind": "conflict"}
    else:
        question = "现在是否胸痛或呼吸困难？"
    assert not convergence_reason(state, ask(question))


@pytest.mark.parametrize("role,schema", [("bianzheng", workflow.FixedAssessment), ("formula", workflow.FixedTreatment),
                                        ("bianzheng", multi.BianzhengArtifact), ("formula", multi.FormulaArtifact),
                                        ("herb", multi.HerbReviewArtifact)])
def test_all_roles_have_strict_fresh_completion_schemas(role, schema):
    final = after_inquiry_schema(schema, role=role)
    assert set(final.model_fields["information_gap"].annotation.__args__) == {"optional", "evidence"}
    if role == "herb":
        assert set(final.model_fields["status"].annotation.__args__) == {"approved", "revise", "rejected"}
    for field in ("question", "questions"):
        if field in final.model_fields:
            if field == "question":
                assert final.model_fields[field].annotation.__args__ == ("",)
            else:
                assert any(getattr(meta, "max_length", None) == 0 for meta in final.model_fields[field].metadata)


def test_optional_new_question_converges_without_first_requesting_another_answer():
    state = ready(("是否精神疲乏？", "有"))
    calls = []
    async def forbidden(_):
        raise AssertionError("Do not first repair into another questionnaire")
    async def finalize(feedback):
        calls.append(feedback)
        return {**ask(""), "status": "completed", "information_gap": "optional"}
    result = asyncio.run(validate_followup(state, ask("请观察面色？"), forbidden, finalize=finalize, assessment=True))
    assert len(calls) == 1 and result["status"] == "completed" and not result["question"]


def test_multi_screenshot_facts_and_answer_memory_fit_native_and_final_requests(monkeypatch):
    text = "本人肚子胀、怕冷、大便稀一个星期，舌苔泛白，脉象无力，精神疲乏，面色苍白，夜间尿频，食欲不振"
    state = ready(("舌苔、脉象和精神状态？", "舌苔泛白，脉象无力，精神疲乏"),
                  ("是否面色苍白？", "有"), ("夜间尿频和食欲情况？", "夜间尿频，食欲不振"))
    state.update(message=text, source_message_id="current", episode_text=text,
                 symptom_facts=symptoms.extract(text, subject="self", message_id="current", thread_id="t"))
    calls = []
    async def native(messages, tools, *, role):
        budget_request(messages, tools, role)
        calls.append("native")
        return {"final": ask("舌苔颜色是什么样的？")}
    async def complete(system, payload, *, role, schema, **kwargs):
        budget_request([{"role": "system", "content": system + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                        {"role": "user", "content": payload}], [], role)
        calls.append("final")
        return {**ask(""), "status": "completed", "information_gap": "optional"}
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    monkeypatch.setattr(model, "achat_tools", native)
    monkeypatch.setattr(model, "acomplete_json", complete)
    # Exercise the same task packaging as the specialist and the same shared
    # runtime, including the second schema appended by acomplete_json.
    result = asyncio.run(multi.bianzheng_agent(state))
    assert result["agent_stop"] == "" and result["diagnosis"]
    assert calls == ["native", "final"]


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_api_repeated_optional_question_finishes_specialists_and_survives_reload(monkeypatch, mode):
    question = "是否面色苍白？"
    original = multi.run_agent_graph
    generated = []
    async def complete(system, user, **kwargs):
        generated.append(kwargs["schema"].__name__)
        converged = kwargs["schema"].__name__.endswith("Converged")
        artifact = ask("" if converged else question)
        artifact.update(status="completed" if converged else "need_user_input", summary="已有失眠疲劳支持初步分析。",
                        uncertainties=["舌脉未确认"], evidence_ids=[])
        if mode == "fixed":
            artifact["questions"] = [] if converged else [question]
            artifact.pop("question")
            artifact.pop("evidence_ids")
        return artifact
    async def graph(**kwargs):
        if kwargs["role"] == "bianzheng":
            kwargs["mock_planner"] = lambda *_: {"final": ask(question)}
        return await original(**kwargs)
    monkeypatch.setattr(model, "acomplete_json", complete)
    if mode == "multi":
        monkeypatch.setattr(multi, "run_agent_graph", graph)
    else:
        original_fixed = workflow.bianzheng
        async def fixed(state):
            with monkeypatch.context() as local:
                local.setattr(workflow, "MODEL_MODE", "live")
                return await original_fixed(state)
        monkeypatch.setattr(workflow, "bianzheng", fixed)
        monkeypatch.setattr(workflow, "acomplete_json", complete)
    with TestClient(app) as client:
        token, uid = register(client, "converge_" + uuid4().hex[:8])
        h = auth(token)
        first = client.post("/api/chat", headers=h, json={"mode": mode, "message": "本人失眠三天，白天疲劳，不用查古籍"}).json()
        assert first["flow"]["status"] == "awaiting_input"
        tid = first["thread_id"]
        saved = safety_store.load_state(uid, tid)["clinical"]["self"]
        assert saved["pending_inquiry"]["stage"] == "bianzheng"
        second = client.post("/api/chat", headers=h, json={"thread_id": tid, "message": "有，舌苔泛白"}).json()
        assert second["intent_type"] == "diagnose", second
        stages = {e["stage"] for e in second["trace"]["events"]}
        assert {"formula", "safety"} <= stages
        if mode == "multi":
            assert {"herb", "reviewer"} <= stages
        assert "处方参考" in second["response"] and "安全说明" in second["response"]
        assert second["message"]["prescription"] is None and not second["message"]["prescription_review"]["approved"]
        saved = safety_store.load_state(uid, tid)["clinical"]["self"]
        assert not saved["pending_question"]
        assert any("舌苔泛白" in row["quote"] for row in saved["inquiry_observations"])
        history = client.get(f"/api/threads/{tid}/messages", headers=h).json()
        assert history[-1]["content"] == second["response"]
        assert any(name.endswith("Converged") for name in generated)
