import copy
import json
from uuid import uuid4

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app import workflow
from app.db import connection, one, now
from app.prescription import CandidatePlan, Dosage, fingerprint, render_plan, review_plan, structured_herb
from test_api import auth, register


@pytest.mark.parametrize("value", [
    {"min": -1, "max": 2, "unit": "g"}, {"min": 3, "max": 2, "unit": "g"},
    {"min": 3, "max": None, "unit": "g"}, {"min": 1, "max": 2, "unit": "unknown"},
    {"min": float("inf"), "max": float("inf"), "unit": "g"},
])
def test_invalid_dosage_schema(value):
    with pytest.raises(ValidationError):
        Dosage(**value)


def sourced_plan():
    source = one("SELECT * FROM dosage_evidence WHERE herb_name='甘草' AND min_dose IS NOT NULL LIMIT 1")
    item = structured_herb("甘草", raw_dosage=source["raw_dosage"], preparation=source["preparation"], source_ids=[source["id"]])
    item["verification_status"] = "verified"
    return {"name": "结构测试", "herbs": [item]}, source


def test_seed_provenance_and_verification_cannot_be_forged():
    with TestClient(app):
        plan, source = sourced_plan()
        checked, review = review_plan(plan, {})
        assert checked["herbs"][0]["verification_status"] == "unverified"
        assert review["compatibility_status"] == "passed" and review["dosage_status"] == "unverified"
        assert source["source_locator"] and source["data_version"] and source["verification_status"] == "unverified"
        with connection() as db:
            db.execute("UPDATE dosage_evidence SET verification_status='verified',reviewed_by='test-curator',reviewed_at='2026-09-26',population='general_adult' WHERE id=?", (source["id"],))
        try:
            assert review_plan(plan, {})[1]["dosage_status"] == "unverified"
            verified, result = review_plan(plan, {"history": "30岁成人"})
            assert result["dosage_status"] == "verified"
            assert verified["herbs"][0]["verification_status"] == "verified"
            restricted, result = review_plan(plan, {"history": "肾病"})
            assert result["dosage_status"] == "unverified"
            assert restricted["herbs"][0]["verification_status"] == "unverified"
        finally:
            with connection() as db:
                db.execute("UPDATE dosage_evidence SET verification_status='unverified',reviewed_by=NULL,reviewed_at=NULL,population='unknown' WHERE id=?", (source["id"],))


def test_sources_units_identifiers_and_empty_doses():
    with TestClient(app):
        plan, source = sourced_plan()
        wrong_unit = copy.deepcopy(plan)
        wrong_unit["herbs"][0]["dosage"]["unit"] = "mg"
        checked, review = review_plan(wrong_unit, {})
        assert review["compatibility_status"] == "passed"
        assert review["dosage_status"] == "rejected" and not review["approved"]
        wrong_preparation = copy.deepcopy(plan)
        wrong_preparation["herbs"][0]["preparation"] = source["preparation"] + "无出处炮制"
        assert not review_plan(wrong_preparation, {})[1]["approved"]
        wrong_id = copy.deepcopy(plan)
        wrong_id["herbs"][0]["herb_id"] = "other"
        assert not review_plan(wrong_id, {})[1]["approved"]
        no_source = copy.deepcopy(plan)
        no_source["herbs"][0]["source_ids"] = []
        checked, review = review_plan(no_source, {})
        assert checked["herbs"][0]["dosage"]["min"] is None
        assert checked["herbs"][0]["verification_status"] == "unverified"
        assert "剂量未提供" in render_plan(checked)
        missing = copy.deepcopy(plan)
        missing["herbs"][0]["source_ids"] = ["missing-source"]
        assert not review_plan(missing, {})[1]["approved"]


def test_body_card_data_and_edit_invalidate_review(monkeypatch):
    with TestClient(app) as client:
        token, _ = register(client, "dose_" + uuid4().hex[:8])
        headers = auth(token)
        first = client.post("/api/chat", headers=headers, json={"message": "最近失眠三天，白天疲劳"}).json()
        message = first["message"]
        assert render_plan(message["prescription"]) in message["content"]
        assert message["prescription_review"]["plan_hash"] == fingerprint(message["prescription"])
        assert all(item["verification_status"] == "unverified" for item in message["prescription"]["herbs"])
        original_finish = workflow.finish
        def edited_finish(state):
            output = original_finish(state)
            plan = copy.deepcopy(state["treatment_plan"])
            herb = plan["herbs"][0]
            herb["dosage"]["min"] = herb["dosage"]["max"] = 999
            return {**output, "treatment_plan": plan}
        monkeypatch.setattr(workflow, "finish", edited_finish)
        modified = client.post("/api/chat", headers=headers,
                               json={"message": "最近失眠三天，白天疲劳", "thread_id": first["thread_id"]})
        assert modified.status_code == 503
        latest = client.get(f'/api/threads/{first["thread_id"]}/messages', headers=headers).json()[-1]
        assert latest["intent_type"] == "failed" and latest["prescription"] is None


def test_old_text_only_examples_restore_empty_unverified_cards():
    with TestClient(app) as client:
        token, _ = register(client, "legacy_card_" + uuid4().hex[:8])
        headers = auth(token)
        thread = client.post("/api/threads", headers=headers, json={}).json()["id"]
        content = "## 方剂知识示例\n**酸枣仁汤**：酸枣仁、甘草、知母、茯苓、川芎\n此为知识演示。"
        with connection() as db:
            db.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
                       (str(uuid4()), thread, "assistant", content, now()))
        message = client.get(f"/api/threads/{thread}/messages", headers=headers).json()[0]
        assert message["content"] == content
        assert message["prescription_review"] is None
        assert len(message["prescription"]["herbs"]) == 5
        assert all(item["dosage"]["min"] is None and item["verification_status"] == "unverified"
                   for item in message["prescription"]["herbs"])
