"""After inquiry, partial facts can produce a sourced draft with concrete risks."""
import copy
import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import workflow
from app.agents import multi
from app.db import one, connection
from app.prescription import fingerprint, migrate_plan, review_plan, knowledge_context
from app.context_budget import budget_request
from app.prescription_reference import reference_prescription, render_reference
from app.consultation_followup import validate_followup
from app.safety import contains_dosing
from app.safety_context import make_context, extract_facts
from app.workflow_presentation import fixed_sections
from test_api import register, auth


def checked_state():
    entry = one("SELECT name,structured_herbs FROM formulas WHERE name='酸枣仁汤'")
    plan = migrate_plan({"name": entry["name"], "herbs": json.loads(entry["structured_herbs"])})
    plan, review = review_plan(plan, {})
    safe, _ = make_context({}, "本人", "")
    reason = "脉象未提供，处方参考需由医师复核"
    return {"intent_type": "diagnose", "workflow_mode": "multi", "diagnosis": "已有症状支持初步方向，脉象未知。",
            "preliminary_analysis": True, "preflight": {"decision": "allow"}, "safety_state": safe,
            "treatment_plan": plan, "prescription_review": review, "reviewer_hash": fingerprint(plan),
            "reviewer_approved": True, "manual_review_reasons": [reason], "safety_errors": [reason],
            "profile": {}, "analysis_questions": ["脉象仍未知"], "consultation_notes": [{"expert": "辨证"}]}


def test_reference_contains_composition_sources_unknowns_and_specific_risks():
    with TestClient(app):
        state = checked_state()
        state["safety_state"]["facts"] = extract_facts("对青霉素过敏", subject="self")
        reference = reference_prescription(state)
        assert reference and not reference["approved"]
        text = render_reference(reference)
        for value in ("酸枣仁汤", "茯苓", "参考组成", "方剂出处", "脉象仍未知", "安全说明", "青霉素", "合并用药", "特殊人群", "配伍", "剂量"):
            assert value in text
        assert contains_dosing(text) and "文献参考范围" in text
        assert all(item["dosage"]["min"] is None for item in reference["herbs"])
        before = copy.deepcopy(state)
        assert reference_prescription(state)
        assert state == before
        notes = workflow.final_consultation_notes(state)
        assert "独立处方参考" in notes[1]["prescription_advice"]
        assert not contains_dosing(notes[1]["prescription_advice"])
        assert text in workflow.finish(state)["response"]
        assert "独立处方参考" in fixed_sections({**state, "workflow_mode": "fixed"})[0][1]["content"]


@pytest.mark.parametrize("change", ["waiting", "scope", "risk", "reviewer", "allergy", "altered", "hash", "source", "dosage"])
def test_unknowns_cannot_override_real_safety_failures_or_source_mismatches(change):
    with TestClient(app):
        state = checked_state()
        if change == "waiting":
            state["intent_type"] = "consult"
        elif change == "scope":
            state["safety_state"]["subject_confirmed"] = False
        elif change == "risk":
            state["preflight"].update(decision="clarify", question_keys=["medication_details"])
        elif change == "reviewer":
            state["reviewer_approved"] = False
        elif change == "allergy":
            state["profile"]["allergies"] = "茯苓"
        elif change == "hash":
            state["prescription_review"]["plan_hash"] = "other"
        else:
            plan = copy.deepcopy(state["treatment_plan"])
            if change == "altered":
                plan["herbs"].pop()
            elif change == "source":
                plan["name"] = "未收录的方剂"
            else:
                plan["herbs"][0]["source_ids"] = ["fake-dose-source"]
            plan, review = review_plan(plan, {})
            state.update(treatment_plan=plan, prescription_review=review, reviewer_hash=fingerprint(plan))
        assert reference_prescription(state) is None


def test_conflict_cannot_be_reclassified_as_only_a_manual_review_need():
    with TestClient(app):
        state = checked_state()
        plan = migrate_plan({"name": "规则测试", "herbs": ["甘草", "海藻"]})
        plan, review = review_plan(plan, {})
        state.update(treatment_plan=plan, prescription_review=review, reviewer_hash=fingerprint(plan),
                     manual_review_reasons=review["errors"], safety_errors=review["errors"])
        assert reference_prescription(state) is None


def test_explicit_negative_safety_answers_are_preserved_in_risk_explanation():
    with TestClient(app):
        state = checked_state()
        state["safety_state"]["facts"] = extract_facts("本人没有过敏史，目前不用药", subject="self")
        text = render_reference(reference_prescription(state))
        assert "用户自述无过敏史" in text and "用户自述目前未用药" in text
        assert "过敏史尚未明确" not in text


def test_not_pregnant_or_lactating_cannot_become_positive_risks():
    with TestClient(app):
        state = checked_state()
        facts = extract_facts("本人30岁，未怀孕也未哺乳，没有过敏史，目前不用药", subject="self")
        assert all(f["assertion"] == "absent" for f in facts if f["category"] == "pregnancy_lactation")
        state["safety_state"]["facts"] = facts
        text = render_reference(reference_prescription(state))
        assert "已记录年龄30岁、怀孕" not in text


def test_only_independently_verified_dose_can_be_shown_in_partial_reference():
    with TestClient(app):
        state = checked_state()
        herb = next(item for item in state["treatment_plan"]["herbs"] if item["dosage"]["min"] is not None)
        source = one("SELECT * FROM dosage_evidence WHERE id=?", (herb["source_ids"][0],))
        with connection() as db:
            db.execute("UPDATE dosage_evidence SET verification_status='verified',reviewed_by='test',reviewed_at='2026-10-15',population='general_adult' WHERE id=?", (source["id"],))
        try:
            state["profile"] = {"history": "30岁成人"}
            plan, review = review_plan(state["treatment_plan"], state["profile"])
            state.update(treatment_plan=plan, prescription_review=review, reviewer_hash=fingerprint(plan))
            reference = reference_prescription(state)
            shown = next(item for item in reference["herbs"] if item["name"] == herb["name"])
            assert shown["dosage"] == herb["dosage"]
            assert contains_dosing(render_reference(reference))
            assert all(item["dosage"]["min"] is None for item in reference["herbs"] if item["name"] != herb["name"])
        finally:
            with connection() as db:
                db.execute("UPDATE dosage_evidence SET verification_status=?,reviewed_by=?,reviewed_at=?,population=? WHERE id=?",
                           (source["verification_status"], source["reviewed_by"], source["reviewed_at"], source["population"], source["id"]))


def test_optional_gaps_do_not_disable_candidate_generation_or_force_another_revision():
    state = {"analysis_only": False, "manual_review_reasons": [], "analysis_questions": []}
    updates = multi.provisional_updates(state, {"information_gap": "optional", "uncertainties": ["脉象未知"]})
    assert not updates["analysis_only"] and updates["preliminary_analysis"]
    assert workflow.safety_route({**updates, "safety_errors": updates["manual_review_reasons"]}) == "finish"
    assert multi.provisional_updates({**state, "analysis_only": True}, {"information_gap": "optional"})["analysis_only"]


def test_necessary_first_inquiry_does_not_force_a_premature_assessment():
    value = {"status": "need_user_input", "information_gap": "patient", "conclusion": "", "question": "本次发病有什么诱因？"}
    async def forbidden(_):
        raise AssertionError("必要依据未确认时，应先等待回答")
    assert asyncio.run(validate_followup({"symptoms": ["失眠", "疲劳"]}, value, forbidden, assessment=True)) == value


def test_fixed_generation_gets_actual_formula_ids_and_dosage_sources(monkeypatch):
    with TestClient(app):
        entry = one("SELECT name,structured_herbs FROM formulas WHERE name='酸枣仁汤'")
        herbs = json.loads(entry["structured_herbs"])
        async def generate(system, user, **kwargs):
            assert "先核对辨证" in system and "处方参考" in system
            assert "实际检索的方剂与药材标识" in user
            assert herbs[0]["herb_id"] in user and herbs[0]["source_ids"][0] in user
            packed_system = system + "\n只输出 JSON 对象，不要 Markdown。JSON Schema：" + json.dumps(workflow.FixedTreatment.model_json_schema(), ensure_ascii=False)
            budget_request([{"role": "system", "content": packed_system}, {"role": "user", "content": user}], [], "formula")
            assert "\"verification_status\": \"unverified\"" not in user
            return {"name": entry["name"], "herbs": herbs, "information_gap": "optional", "questions": [],
                    "rationale": "已知资料支持参考方向，脉象仍未知。", "safety_notes": ["相互作用仍需核对。"]}
        monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
        monkeypatch.setattr(workflow, "acomplete_json", generate)
        result = asyncio.run(workflow.formula({"symptoms": ["失眠"], "message": "失眠三天"}))
        assert result["treatment_plan"]["name"] == entry["name"]
        assert "脉象仍未知" in result["formula_reasoning"]
        assert result["treatment_safety_notes"] == ["相互作用仍需核对。"]


def test_unverified_dose_rows_are_not_admitted_as_verified_generation_evidence():
    with TestClient(app):
        assert json.loads(knowledge_context(["酸枣仁", "茯苓"]))
        assert not json.loads(knowledge_context(["酸枣仁", "茯苓"], verified_only=True))


def test_fixed_repeated_unknown_observation_generates_fresh_restricted_reference(monkeypatch):
    with TestClient(app):
        state = checked_state()
        state.update(thread_id="t", episode_id="e", symptoms=["失眠"], analysis_only=True, reviewer_hash="",
                     inquiry_answers=[{"thread_id": "t", "episode_id": "e", "subject": "self", "answer_message_id": "m",
                                       "question": "您的脉象如何？", "answer": "不清楚"}])
        calls = []
        async def generate(system, user, **kwargs):
            calls.append(kwargs["schema"])
            if len(calls) == 1:
                return {"name": "待补充", "herbs": [], "information_gap": "patient", "questions": ["您的脉象如何？"]}
            assert issubclass(kwargs["schema"], workflow.FixedTreatment)
            assert kwargs["schema"].__name__.endswith("Converged")
            assert "追问收束原因" in user
            return {**state["treatment_plan"], "information_gap": "optional", "questions": [],
                    "rationale": "有来源的候选参考，脉象仍未知，个人适用性需医师复核。", "safety_notes": ["剂量留空。"]}
        monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
        monkeypatch.setattr(workflow, "acomplete_json", generate)
        state.update(asyncio.run(workflow.formula(state)))
        state.update(workflow.fixed_safety(state))
        assert len(calls) == 2 and state["preliminary_analysis"]
        assert not state["prescription_review"]["approved"]
        assert reference_prescription(state)


def test_repeated_unknown_medication_question_never_forces_a_reference():
    with TestClient(app):
        state = checked_state()
        state.update(thread_id="t", episode_id="e", symptoms=["失眠"],
                     inquiry_answers=[{"thread_id": "t", "episode_id": "e", "subject": "self", "answer_message_id": "m",
                                       "question": "正在使用什么药物？", "answer": "不清楚"}])
        value = {"name": "待补充", "herbs": [], "information_gap": "patient", "questions": ["正在使用什么药物？"]}
        async def repair(_):
            return value
        async def forbidden(_):
            raise AssertionError("关键用药安全信息不能被强制跳过")
        from app.model import InvalidModelOutput
        with pytest.raises(InvalidModelOutput):
            asyncio.run(validate_followup(state, value, repair, finalize=forbidden))


@pytest.mark.parametrize("mode", ["fixed", "multi"])
def test_inquiry_then_unknown_optional_information_keeps_reference_and_risk_explanation(mode, monkeypatch):
    question = "本次睡眠问题是否伴有明显口渴？"
    conclusion = "已有失眠和疲劳支持初步辨证分析，脉象仍未知，具体证候需继续复核。"
    original_multi, original_fixed = multi.run_agent_graph, workflow.bianzheng

    async def ask_multi(**kwargs):
        run = await original_multi(**kwargs)
        if kwargs["role"] == "bianzheng":
            answered = kwargs["state"].get("answered_question")
            run.output.update(status="completed" if answered else "need_user_input", conclusion=conclusion,
                              information_gap="optional" if answered else "patient", question="" if answered else question,
                              uncertainties=["脉象仍未知"])
        return run

    async def assessment(system, user, **kwargs):
        answered = "上一轮问题与本轮回答" in user
        return {"status": "completed" if answered else "need_user_input", "conclusion": conclusion, "summary": conclusion,
                "questions": [] if answered else [question], "information_gap": "optional" if answered else "patient",
                "uncertainties": ["脉象仍未知"]}

    async def ask_fixed(state):
        with monkeypatch.context() as scoped:
            scoped.setattr(workflow, "MODEL_MODE", "openai_compatible")
            return await original_fixed(state)

    monkeypatch.setattr(multi, "run_agent_graph", ask_multi)
    if mode == "fixed":
        monkeypatch.setattr(workflow, "acomplete_json", assessment)
        monkeypatch.setattr(workflow, "bianzheng", ask_fixed)
    with TestClient(app) as client:
        token, _ = register(client, "rx_ref_" + uuid4().hex[:8])
        h = auth(token)
        first = client.post("/api/chat", headers=h, json={"mode": mode, "message": "本人失眠三天，白天疲劳，不用查古籍"}).json()
        assert first["flow"]["status"] == "awaiting_input"
        assert "处方参考" not in first["response"] and first["message"]["prescription"] is None
        second = client.post("/api/chat", headers=h, json={"thread_id": first["thread_id"], "message": "不清楚"}).json()
        assert second["intent_type"] == "diagnose" and second["flow"]["status"] == "restricted"
        assert conclusion in second["response"]
        for text in ("处方参考", "酸枣仁汤", "参考组成", "茯苓", "安全说明", "合并用药", "脉象仍未知"):
            assert text in second["response"]
        reference = second["flow"]["prescription_reference"]
        assert not reference["approved"] and all(item["dosage"]["min"] is None for item in reference["herbs"])
        assert "文献参考范围" in second["response"] and "本次个人拟用量" in reference["dose_notice"]
        assert second["message"]["prescription"] is None
        assert not second["message"]["prescription_review"]["approved"]
        history = client.get(f'/api/threads/{first["thread_id"]}/messages', headers=h).json()
        assert history[-1]["content"] == second["response"]
        assert history[-1]["trace"]["presentation"]["prescription_reference"] == reference
        if mode == "multi":
            assert "独立处方参考" in history[-1]["notes"][1]["prescription_advice"]
        else:
            assert "独立处方参考" in history[-1]["trace"]["presentation"]["sections"][1]["content"]
