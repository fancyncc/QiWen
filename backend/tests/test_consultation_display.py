import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.agents.multi import BianzhengArtifact
from app.safety_context import make_context, extract_facts
from app.workflow import final_consultation_notes, finish
from app.prescription import migrate_plan, render_plan


def test_display_notes_keep_real_confidence_and_concise_safety_warnings():
    with TestClient(app):
        safe, _ = make_context({}, "本人", "")
        safe["facts"] = extract_facts("对青霉素过敏", subject="self")
        state = {"consultation_notes": [{"expert": "辨证"}], "diagnosis": "待辨证候方向。", "safety_state": safe,
                 "safety_errors": ["verification_status 为 unverified，内置演示种子资料为空"], "analysis_only": True,
                 "profile": {"allergies": "青霉素"}, "agent_artifacts": {"bianzheng": {
                     "syndrome": "待辨证候方向", "confidence": 0.62, "summary": "可供核对的初步分析摘要。"}}}
        notes = final_consultation_notes(state)
        assert notes[0]["confidence"] == 0.62 and notes[0]["summary"] == "可供核对的初步分析摘要。"
        assert "不能形成可执行的个人处方" in notes[1]["prescription_advice"]
        warnings = "；".join(notes[2]["public_warnings"])
        assert "青霉素" in warnings and "复核" in warnings
        assert "verification_status" not in warnings and "unverified" not in warnings
        del state["agent_artifacts"]["bianzheng"]["confidence"]
        assert final_consultation_notes(state)[0]["confidence"] is None


def test_confidence_is_optional_and_bounded_in_model_schema():
    assert BianzhengArtifact(status="completed", conclusion="说明").confidence is None
    with pytest.raises(ValidationError):
        BianzhengArtifact(status="completed", conclusion="说明", confidence=95)


def test_structured_advice_can_show_reviewed_dose_but_model_summary_cannot():
    with TestClient(app):
        plan = migrate_plan({"name": "知识示例", "herbs": ["茯苓"]})
        plan["herbs"][0]["dosage"].update(min=3, max=6)
        state = {"intent_type": "diagnose", "consultation_notes": [{"expert": "辨证"}],
                 "diagnosis": "辨证结果待面诊确认。", "treatment_plan": plan, "safety_errors": [],
                 "agent_artifacts": {"bianzheng": {"summary": "可核对的摘要。"}}}
        result = finish(state)
        assert result["consultation_notes"] and render_plan(plan) in result["consultation_notes"][1]["prescription_advice"]
        state["agent_artifacts"]["bianzheng"]["summary"] = "模型自行添加剂量999克"
        assert not finish(state)["consultation_notes"]
