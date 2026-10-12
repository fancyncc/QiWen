from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from test_api import register, auth
from app.main import app
from app.db import one, all_rows, connection
from app.safety_context import extract_facts
from app import workflow, preflight


def chat(client, headers, message, thread=None):
    response = client.post("/api/chat", headers=headers, json={"message": message, **({"thread_id": thread} if thread else {})})
    assert response.status_code == 200, response.text
    data = response.json()
    assert "response" in data, data
    return data


def suggestions(client, headers, thread):
    response = client.get("/api/profile-suggestions", headers=headers, params={"thread_id": thread})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("text,decision", [
    ("我没有胸痛，只有头痛三天", "allow"),
    ("我没有胸痛，但现在呼吸困难", "block"),
    ("我没有胸痛和呼吸困难", "allow"),
    ("我并不是没有胸痛", "clarify"),
    ("我昨晚胸痛，现在没有了", "clarify"),
    ("如果出现胸痛会怎样", "allow"),
    ("我没有怀孕，也不在哺乳", "allow"),
    ("我正在哺乳，还没有结束", "clarify"),
    ("请科普什么是胸痛", "allow"),
])
def test_negation_scope_and_time(text, decision):
    with TestClient(app) as client:
        token, _ = register(client, "neg_" + uuid4().hex[:8])
        data = chat(client, auth(token), text)
        assert data["trace"]["preflight"]["decision"] == decision
        if decision != "allow":
            assert not {e["stage"] for e in data["trace"]["events"]} & {"formula", "bianzheng"}


@pytest.mark.parametrize("text", ["如果我对甘草过敏怎么办", "你说我是气虚体质", "我妈妈对甘草过敏", "朋友是湿热体质"])
def test_no_self_proposal_for_hypothesis_model_inference_or_others(text):
    with TestClient(app) as client:
        token, user_id = register(client, "who_" + uuid4().hex[:8])
        h = auth(token)
        data = chat(client, h, text)
        assert not suggestions(client, h, data["thread_id"])["items"]
        assert not client.get(f"/api/profiles/{user_id}", headers=h).json()["facts"]


def test_inherited_other_subject_does_not_use_account_portrait():
    with TestClient(app) as client:
        token, user_id = register(client, "other_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{user_id}", headers=h, json={"allergies": "甘草"})
        first = chat(client, h, "帮我妈妈咨询，她最近失眠")
        second = chat(client, h, "三天，白天疲劳，对茯苓过敏", first["thread_id"])
        assert second["trace"]["safety"]["subject"].startswith("other:")
        assert not suggestions(client, h, first["thread_id"])["items"]
        state = json.loads(one("SELECT state FROM thread_safety_state WHERE thread_id=?", (first["thread_id"],))["state"])
        assert any(f["entity"] == "茯苓" and f["subject"].startswith("other:") for f in state["facts"])
        assert "画像中的过敏信息：甘草" not in second["response"]


def test_pregnancy_short_answer_continuity_and_no_repeated_question():
    with TestClient(app) as client:
        token, _ = register(client, "preg_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我正在哺乳，头痛")
        tid = first["thread_id"]
        assert first["trace"]["safety"]["pending"]["key"] == "pregnancy_status"
        unrelated = chat(client, h, "头痛三天", tid)
        assert unrelated["trace"]["safety"]["pending"]["id"] == first["trace"]["safety"]["pending"]["id"]
        second = chat(client, h, "是", tid)
        assert second["trace"]["safety"]["status"] == "restricted"
        assert second["trace"]["safety"]["pending"] is None
        third = chat(client, h, "三天，未有其它不适", tid)
        assert third["trace"]["safety"]["status"] == "restricted"
        assert "目前仍然适用吗" not in third["response"]
        assert not {e["stage"] for e in third["trace"]["events"]} & {"formula", "bianzheng"}


def test_denial_after_safety_question_restores_original_complaint():
    with TestClient(app) as client:
        token, _ = register(client, "resume_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我以前在哺乳，最近头痛三天")
        second = chat(client, h, "不是", first["thread_id"])
        assert second["trace"]["preflight"]["decision"] == "allow"
        assert "头痛" in second["symptoms"]


def test_unspecified_identity_confirmation_and_long_term_read_without_retrieval(monkeypatch):
    with TestClient(app) as client:
        token, user_id = register(client, "portrait_" + uuid4().hex[:8])
        other, _ = register(client, "steal_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "对甘草过敏")
        tid = first["thread_id"]
        assert not suggestions(client, h, tid)["items"]
        chat(client, h, "是我本人", tid)
        proposal = suggestions(client, h, tid)
        assert proposal["items"][0]["fact"]["subject"] == "self"
        assert client.get("/api/profile-suggestions", headers=auth(other), params={"thread_id": tid}).status_code == 404
        item = proposal["items"][0]["id"]
        body = {"revision": proposal["revision"], "decisions": [{"id": item, "action": "accept", "confirm_self": True}]}
        assert client.post("/api/profile-suggestions/resolve", headers=auth(other), json=body).status_code == 404
        missing = {**body, "decisions": [{"id": item, "action": "accept"}]}
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=missing).status_code == 422
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=body).status_code == 200
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=body).status_code == 200
        portrait = client.get(f"/api/profiles/{user_id}", headers=h).json()
        assert "甘草" in portrait["allergies"]
        assert portrait["facts"][0]["confirmation"] == "confirmed"
        new = chat(client, h, "我最近失眠三天，白天疲劳")
        assert "药材清单已隐藏" in new["response"]
        assert not suggestions(client, h, new["thread_id"])["items"]
        revision = portrait["safety_revision"]
        fact_id = portrait["facts"][0]["id"]
        assert client.delete(f"/api/profile-facts/{fact_id}", headers=auth(other), params={"revision": revision}).status_code == 404
        assert client.delete(f"/api/profile-facts/{fact_id}", headers=h, params={"revision": revision}).status_code == 200
        assert not client.get(f"/api/profiles/{user_id}", headers=h).json()["facts"]


def test_decline_deduplicates_but_current_allergy_still_applies():
    with TestClient(app) as client:
        token, user_id = register(client, "decline_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我对甘草过敏，最近失眠三天，白天疲劳")
        assert "药材清单已隐藏" in first["response"]
        p = suggestions(client, h, first["thread_id"])
        assert client.post("/api/profile-suggestions/resolve", headers=h, json={
            "revision": p["revision"], "decisions": [{"id": p["items"][0]["id"], "action": "dismiss"}]}).status_code == 200
        second = chat(client, h, "本人对甘草过敏，最近失眠三天，白天疲劳", first["thread_id"])
        assert not suggestions(client, h, first["thread_id"])["items"]
        assert "药材清单已隐藏" in second["response"]
        assert client.get(f"/api/profiles/{user_id}", headers=h).json()["allergies"] == ""


def test_conflict_requires_explicit_replace_and_stale_revision_is_rejected():
    with TestClient(app) as client:
        token, user_id = register(client, "conflict_" + uuid4().hex[:8])
        h = auth(token)
        client.patch(f"/api/profiles/{user_id}", headers=h, json={"allergies": "甘草"})
        data = chat(client, h, "我没有过敏史")
        p = suggestions(client, h, data["thread_id"])
        assert p["items"][0]["conflict"]
        body = {"revision": p["revision"], "decisions": [{"id": p["items"][0]["id"], "action": "accept", "confirm_self": True}]}
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=body).status_code == 409
        body["decisions"][0]["replace"] = True
        client.patch(f"/api/profiles/{user_id}", headers=h, json={"history": "用户新补充"})
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=body).status_code == 409
        body["revision"] = suggestions(client, h, data["thread_id"])["revision"]
        assert client.post("/api/profile-suggestions/resolve", headers=h, json=body).status_code == 200
        assert client.get(f"/api/profiles/{user_id}", headers=h).json()["allergies"] == ""


def test_mixed_people_and_unknown_allergen_are_not_silently_ignored():
    with TestClient(app) as client:
        token, _ = register(client, "mixed_" + uuid4().hex[:8])
        h = auth(token)
        data = chat(client, h, "我没有过敏史，但我妈妈对花生过敏")
        assert data["trace"]["safety"]["pending"]["key"] == "subject"
        p = suggestions(client, h, data["thread_id"])
        assert not p["items"]  # Patient identity must be resolved before proposing a self save.
        peanut = chat(client, h, "我对花生过敏")
        assert peanut["trace"]["preflight"]["decision"] == "clarify"
        assert suggestions(client, h, peanut["thread_id"])["items"][0]["fact"]["entity"] == "花生"


def test_new_constitution_and_history_are_batched_and_never_automatically_saved():
    with TestClient(app) as client:
        token, user_id = register(client, "batch_" + uuid4().hex[:8])
        h = auth(token)
        data = chat(client, h, "我是湿热体质，有高血压，对青霉素过敏")
        p = suggestions(client, h, data["thread_id"])
        assert {item["fact"]["category"] for item in p["items"]} == {"constitution", "history", "allergy"}
        assert not client.get(f"/api/profiles/{user_id}", headers=h).json()["facts"]
        decisions = [{"id": item["id"], "action": "accept", "confirm_self": True} for item in p["items"]]
        response = client.post("/api/profile-suggestions/resolve", headers=h, json={"revision": p["revision"], "decisions": decisions})
        assert response.status_code == 200, response.text
        portrait = client.get(f"/api/profiles/{user_id}", headers=h).json()
        assert "用户自述" in portrait["constitution"] and "青霉素" in portrait["allergies"]
        assert len(portrait["facts"]) == 3


def test_model_evidence_validation_and_cannot_weaken_rules(monkeypatch):
    async def forged(*a, **kw):
        return {"decision": "allow", "risk_categories": [], "reason": "stub", "required_questions": [],
                "observations": [{"category": "allergy", "entity": "甘草", "subject": "self", "assertion": "present",
                                  "temporal_scope": "current", "quote": "我对甘草过敏"}]}
    monkeypatch.setattr(preflight, "acomplete_json", forged)
    monkeypatch.setattr(workflow, "MODEL_MODE", "openai_compatible")
    with TestClient(app) as client:
        token, _ = register(client, "evidence_" + uuid4().hex[:8])
        h = auth(token)
        data = chat(client, h, "我最近头痛三天")
        assert data["trace"]["preflight"]["risk_categories"] == ["preflight_unavailable"]
        assert not suggestions(client, h, data["thread_id"])["items"]


def test_migration_is_repeatable():
    from app.db import init_db
    with TestClient(app):
        init_db()
        init_db()
        assert one("SELECT 1 FROM sqlite_master WHERE name='thread_safety_state'")


def test_allergy_detail_answer_without_repeating_allergy_keyword():
    with TestClient(app) as client:
        token, _ = register(client, "details_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我有药物过敏，最近失眠三天")
        assert first["trace"]["safety"]["pending"]["key"] == "allergy_details"
        second = chat(client, h, "甘草，服用后起疹子", first["thread_id"])
        assert second["trace"]["preflight"]["decision"] == "allow"
        assert "药材清单已隐藏" in second["response"]
        p = suggestions(client, h, first["thread_id"])
        assert any(item["fact"]["entity"] == "甘草" and "疹子" in item["fact"]["details"] for item in p["items"])


def test_food_is_not_medication_and_explicit_allergy_denial_is_not_positive():
    with TestClient(app) as client:
        token, _ = register(client, "food_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我正在吃饭，没有其他不适")
        assert first["trace"]["preflight"]["decision"] == "allow"
        assert not suggestions(client, h, first["thread_id"])["items"]
        second = chat(client, h, "我对甘草不过敏，失眠三天，白天疲劳")
        assert second["trace"]["preflight"]["decision"] == "allow"
        assert "药材清单已隐藏" not in second["response"]


def test_explicit_consultation_switch_preserves_separate_safety_questions():
    with TestClient(app) as client:
        token, _ = register(client, "switch_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我正在哺乳，头痛")
        tid = first["thread_id"]
        second = chat(client, h, "现在替我母亲咨询，她正在吃降压药", tid)
        assert second["trace"]["safety"]["subject"] == "other:母亲"
        assert second["trace"]["safety"]["pending"]["key"] == "medication_details"
        third = chat(client, h, "现在帮我自己咨询", tid)
        assert third["trace"]["safety"]["subject"] == "self"
        assert third["trace"]["safety"]["pending"]["id"] == first["trace"]["safety"]["pending"]["id"]


def test_portrait_change_during_generation_prevents_commit(monkeypatch):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    original = workflow.formula
    async def paused(state):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.02)
        return await original(state)
    monkeypatch.setattr(workflow, "formula", paused)
    with TestClient(app) as client, ThreadPoolExecutor() as pool:
        token, user_id = register(client, "race_" + uuid4().hex[:8])
        h = auth(token)
        tid = client.post("/api/threads", headers=h, json={}).json()["id"]
        future = pool.submit(client.post, "/api/chat", headers=h,
                             json={"message": "我最近失眠三天，白天疲劳，不用查古籍", "thread_id": tid})
        try:
            assert entered.wait(8)
            assert client.patch(f"/api/profiles/{user_id}", headers=h, json={"allergies": "甘草"}).status_code == 200
        finally:
            release.set()
        result = future.result(timeout=12)
        assert result.status_code == 503, result.text
        assert one("SELECT status FROM chat_runs WHERE thread_id=?", (tid,))["status"] == "failed"
        assert one("SELECT * FROM thread_safety_state WHERE thread_id=?", (tid,)) is None
        assert one("SELECT * FROM profile_suggestions WHERE thread_id=?", (tid,)) is None
        assert not any(row["intent_type"] == "diagnose" for row in all_rows("SELECT intent_type FROM messages WHERE thread_id=?", (tid,)))


def test_negating_child_age_does_not_imply_adulthood():
    with TestClient(app) as client:
        token, _ = register(client, "age_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我的孩子8岁，咳嗽")
        second = chat(client, h, "不是8岁", first["thread_id"])
        assert second["trace"]["preflight"]["decision"] == "clarify"
        assert second["trace"]["safety"]["pending"]["key"] == "conflict"
        third = chat(client, h, "更正，不是8岁，而是18岁", first["thread_id"])
        assert third["trace"]["preflight"]["decision"] == "allow"


def test_unresolved_proposal_follows_repeated_fact_to_new_thread():
    with TestClient(app) as client:
        token, _ = register(client, "pending_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我是湿热体质")
        old = suggestions(client, h, first["thread_id"])["items"][0]
        second = chat(client, h, "本人是湿热体质")
        current = suggestions(client, h, second["thread_id"])["items"]
        assert len(current) == 1 and current[0]["id"] == old["id"]
        assert not suggestions(client, h, first["thread_id"])["items"]
