import copy
import asyncio
import json

from fastapi.testclient import TestClient

from app.main import app
from app.workflow import final_consultation_notes, finish
from app.workflow_presentation import build_presentation, fixed_sections
from app.dosage_references import literature_dose
from app.prescription_reference import reference_prescription
from app.prescription import review_plan
from app.clinical_output import assessment_text
from app import workflow
from app.context import clinical_context
from app.context_budget import budget_request
from test_prescription_reference import checked_state


def test_full_clinical_explanations_survive_finalization_in_both_modes():
    with TestClient(app):
        state = checked_state()
        state["diagnosis"] = "### 初步证候\n基于已确认症状提出的方向。\n\n### 辨证依据与病机\n" + "临床依据和具体鉴别。" * 45
        state["diagnosis_summary"] = "短摘要不应替换完整辨证。"
        state["formula_reasoning"] = "治法与候选方的关系及具体适用条件。"
        state["agent_artifacts"] = {"herb": {"clinical_review": "茯苓与酸枣仁配伍宁心，川芎兼顾气血运行；需核对相关禁忌。"}}
        state["retrieval_warning"] = "检索内部状态不应成为临床正文。"
        notes = final_consultation_notes(state)
        assert state["formula_reasoning"] in notes[1]["summary"]
        assert state["agent_artifacts"]["herb"]["clinical_review"] == notes[2]["summary"]
        assert "性味归经" not in notes[2]["summary"]
        for mode in ("fixed", "multi"):
            current = {**state, "workflow_mode": mode}
            flow = build_presentation(current, {"mode": mode})
            assert flow["prescription_reference"]["name"] == "酸枣仁汤"
            result = finish(current)
            assert state["diagnosis"] in result["response"]
            assert state["formula_reasoning"] in result["response"]
            assert "检索内部状态" not in result["response"]
            assert "方剂出处" in result["response"] and "文献参考范围" in result["response"]
        assert fixed_sections(state)[0][0]["content"] == state["diagnosis"]


def test_literature_ranges_do_not_promote_personal_plan_or_mutate_evidence():
    with TestClient(app):
        state = checked_state()
        before = copy.deepcopy(state)
        ref = reference_prescription(state)
        item = next(item for item in ref["herbs"] if item["name"] == "茯苓")
        assert item["reference_dose"]["min"] == 10 and item["reference_dose"]["max"] == 15
        assert item["reference_dose"]["url"].endswith("nmm-01ei")
        assert "每剂或每日" in item["reference_dose"]["basis"]
        assert item["dosage"]["min"] is None and item["verification_status"] == "unverified"
        assert state == before
        # Display references never enter the plan's source_ids or approval path.
        plan, reviewed = review_plan(state["treatment_plan"], {})
        assert reviewed["dosage_status"] == "unverified"
        assert all(item["verification_status"] != "verified" for item in plan["herbs"])


def test_preparation_identity_is_exact_and_missing_ranges_stay_missing():
    assert literature_dose("炙甘草")["url"] != literature_dose("甘草")["url"]
    assert literature_dose("人参") and literature_dose("党参") is None
    assert literature_dose("芍药") is None and literature_dose("白芍")
    assert literature_dose("炮姜") is None and literature_dose("干姜")
    assert literature_dose("模型编造的药物") is None


def test_structured_assessment_keeps_clinical_links_and_differential_conditions():
    artifact = {"conclusion": "初步方向待面诊核对。", "clinical_basis": ["已确认症状与病机的具体对应。"],
                "differentials": ["尚未询问的信息只能作为待确认的鉴别条件。"], "treatment_principle": "与辨证对应的治法。"}
    text = assessment_text(artifact)
    for value in ("辨证依据与病机", "鉴别与判断条件", "### 治法", *artifact["clinical_basis"], *artifact["differentials"]):
        assert value in text
    assert assessment_text({"conclusion": "旧记录的临床内容。"}) == "旧记录的临床内容。"


def test_independent_reference_review_receives_actual_sources_without_duplicate_plan(monkeypatch):
    with TestClient(app):
        state = checked_state()
        async def review(system, user, **kwargs):
            sources = json.loads(user.split("实际查得的方剂与药材条目（不是个人适用性批准）：", 1)[1])
            assert sources["formula"]["name"] == "酸枣仁汤" and sources["formula"]["source"]
            assert "structured_herbs" not in sources["formula"]
            assert {row["name"] for row in sources["herbs"]} == {row["name"] for row in state["treatment_plan"]["herbs"]}
            assert "不批准个人适用性" in system
            budget_request([{"role": "system", "content": system + json.dumps(kwargs["schema"].model_json_schema(), ensure_ascii=False)},
                            {"role": "user", "content": user}], [], "reviewer")
            return {"approved": True, "reason": "文献内容可供医师核对，个人适用性仍待确认。"}
        monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
        monkeypatch.setattr(workflow, "acomplete_json", review)
        assert asyncio.run(workflow._review(state["treatment_plan"], clinical_context(state)))[0]
        assert not reference_prescription(state)["approved"]
