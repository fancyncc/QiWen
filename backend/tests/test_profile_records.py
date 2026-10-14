from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.db import connection, now
from app.safety_context import extract_facts, profile_candidates, make_context
from app import safety_store
from test_api import register, auth


def test_temporary_symptoms_and_states_are_excluded_even_when_model_labels_history():
    facts = extract_facts("我是阳虚体质，对青霉素过敏，有高血压，正在哺乳，正在服用华法林")
    for symptom, quote in [("头晕", "偶尔头晕"), ("情绪低落", "发病时情绪低落")]:
        facts.append({**facts[0], "category": "history", "entity": symptom,
                      "source": {**facts[0]["source"], "quote": quote}})
    proposed = profile_candidates(facts)
    assert {f["category"] for f in proposed} == {"allergy", "constitution", "history"}
    assert [f["entity"] for f in proposed if f["category"] == "history"] == ["高血压"]
    assert any(f["category"] == "medication" for f in facts)  # Session risks remain available.
    assert profile_candidates(extract_facts("我曾确诊为胃炎"))[0]["entity"] == "胃炎"
    assert not profile_candidates(extract_facts("我妈妈曾确诊为胃炎"))


def test_legacy_pending_symptom_suggestions_are_hidden_and_cannot_be_accepted():
    with TestClient(app) as client:
        token, uid = register(client, "legacy_" + uuid4().hex[:8])
        h = auth(token)
        tid = client.post("/api/threads", headers=h, json={}).json()["id"]
        fact = extract_facts("我有高血压")[0]
        fact.update(category="history", entity="头晕")
        fact["source"]["quote"] = "偶尔头晕"
        suggestion_id = str(uuid4())
        run_id = str(uuid4())
        with connection() as db:
            db.execute("INSERT INTO chat_runs(id,thread_id,user_id,status,created_at,updated_at) VALUES(?,?,?,'completed',?,?)",
                       (run_id, tid, uid, now(), now()))
            db.execute("INSERT INTO profile_suggestions(id,user_id,thread_id,run_id,dedup_key,fact,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',?,?)",
                       (suggestion_id, uid, tid, run_id, str(uuid4()), safety_store.dumps(fact), now(), now()))
        result = client.get("/api/profile-suggestions", headers=h, params={"thread_id": tid}).json()
        assert result["items"] == []
        response = client.post("/api/profile-suggestions/resolve", headers=h, json={"revision": result["revision"],
            "decisions": [{"id": suggestion_id, "action": "accept", "confirm_self": True}]})
        assert response.status_code == 422


def test_medical_records_persist_edit_and_delete_with_ownership_and_revision_checks():
    with TestClient(app) as client:
        token, uid = register(client, "records_" + uuid4().hex[:8])
        other, other_uid = register(client, "records_other_" + uuid4().hex[:8])
        h = auth(token)
        path = f"/api/profiles/{uid}"
        profile = client.get(path, headers=h).json()
        assert profile["medical_records"] == []
        body = {"revision": profile["safety_revision"], "confirm_self": True, "record_date": "2025-09-10",
                "institution": "测试医院", "diagnosis": "既往胃炎", "treatment": "按当时病历记录治疗", "outcome": "复诊后好转"}
        assert client.post(path + "/medical-records", headers=auth(other), json=body).status_code == 403
        assert client.post(path + "/medical-records", headers=h, json={**body, "confirm_self": False}).status_code == 422
        assert client.post(path + "/medical-records", headers=h, json={**body, "record_date": "2025-02-30"}).status_code == 422
        created = client.post(path + "/medical-records", headers=h, json=body)
        assert created.status_code == 201
        profile = created.json()
        record = profile["medical_records"][0]
        assert record["diagnosis"] == body["diagnosis"] and record["outcome"] == body["outcome"]
        record_path = path + "/medical-records/" + record["id"]
        assert client.put(record_path, headers=h, json=body).status_code == 409
        body.update(revision=profile["safety_revision"], outcome="后续复诊稳定")
        assert client.put(record_path, headers=h, json=body).status_code == 200
        persisted = client.get(path, headers=h).json()
        assert persisted["medical_records"][0]["outcome"] == "后续复诊稳定"
        state, _ = make_context({}, "本人", "")
        _, clinical_profile = safety_store.load(uid, "test-thread", state)
        assert clinical_profile["medical_records"][0]["diagnosis"] == body["diagnosis"]
        other_path = f"/api/profiles/{other_uid}/medical-records/{record['id']}"
        assert client.put(other_path, headers=auth(other), json={**body, "revision": 0}).status_code == 404
        assert client.delete(record_path, headers=h, params={"revision": 0}).status_code == 409
        assert client.delete(record_path, headers=h, params={"revision": persisted["safety_revision"]}).status_code == 200
        assert client.get(path, headers=h).json()["medical_records"] == []
