from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from test_api import register, auth
from app.main import app
from app.db import one
from app import symptoms, workflow


@pytest.mark.parametrize("text,positive,denied", [
    ("头痛，没有咳嗽", ["头痛"], ["咳嗽"]),
    ("不咳嗽，但头痛", ["头痛"], ["咳嗽"]),
    ("头痛没有咳嗽", ["头痛"], ["咳嗽"]),
    ("不是没有咳嗽", ["咳嗽"], []),
    ("咳嗽不明显", ["咳嗽"], []),
    ("以前咳嗽，现在没有咳嗽", [], ["咳嗽"]),
    ("没有咳嗽、发热和鼻塞", [], ["咳嗽", "发热", "鼻塞"]),
    ("没有咳嗽发热有头痛", ["头痛"], ["咳嗽", "发热"]),
    ("没有口干口苦、腹痛，舌苔发白", ["舌苔白"], ["腹痛", "口干", "口苦"]),
    ("我没有发热而且头痛", ["头痛"], ["发热"]),
    ("以前咳嗽，现在没有了", [], ["咳嗽"]),
    ("头疼三天", ["头痛"], []),
    ("没有明显咳嗽", [], ["咳嗽"]),
    ("不怎么咳嗽", ["咳嗽"], []),
    ("没有过敏史头痛", ["头痛"], []),
])
def test_scope_examples(text, positive, denied):
    assert workflow.symptom_observations(text) == (positive, denied)


def test_degree_time_and_offsets_remain_source_backed():
    text = "以前咳嗽，现在咳嗽不明显"
    facts = symptoms.extract(text, message_id="message-one", thread_id="thread-one")
    assert [(f["assertion"], f["time_scope"]) for f in facts] == [("present", "past"), ("present", "current")]
    assert facts[-1]["degree_text"] == "不明显"
    assert "轻度" not in json.dumps(facts, ensure_ascii=False)
    for fact in facts:
        assert text[fact["source"]["start"]:fact["source"]["end"]] == "咳嗽"
        assert fact["source"]["quote"] in text
        assert fact["source"]["message_id"] == "message-one"


def test_postfix_and_next_uncertainty_do_not_leak_between_mentions():
    facts = symptoms.extract("咳嗽不明显头痛")
    assert facts[0]["degree_text"] == "不明显"
    assert facts[1]["degree_text"] == ""
    positive, denied, pending = symptoms.view(symptoms.extract("头痛不能排除咳嗽"), "self")
    assert positive == ["头痛"] and denied == []
    assert pending["symptom"] == "咳嗽"


@pytest.mark.parametrize("text", ["不能排除咳嗽", "不知道有没有咳嗽", "咳嗽没有加重", "咳嗽没好转"])
def test_uncertain_presence_never_enters_positive_or_denied(text):
    positive, denied, pending = symptoms.view(symptoms.extract(text), "self")
    assert positive == denied == []
    assert pending["kind"] == "assertion"


def test_trend_does_not_erase_existing_presence():
    facts = symptoms.merge(symptoms.extract("我有咳嗽", message_id="a"), symptoms.extract("咳嗽没有加重", message_id="b"))
    assert symptoms.view(facts, "self") == (["咳嗽"], [], None)
    assert facts[-1]["trend_text"] == "没有加重"
    assert facts[-1]["assertion"] == "unspecified"


def test_individual_durations_and_patient_scope():
    facts = symptoms.extract("母亲咳嗽一周我头痛三天")
    assert [(f["subject"], f["duration_text"]) for f in facts] == [("other:母亲", "一周"), ("self", "三天")]
    assert symptoms.view(facts, "self") == (["头痛"], [], None)
    assert symptoms.view(facts, "other:母亲") == (["咳嗽"], [], None)


@pytest.mark.parametrize("text", ["如果我咳嗽怎么办", "请科普什么是咳嗽", "模型判断我头痛"])
def test_hypotheses_and_model_claims_not_patient_facts(text):
    assert not symptoms.extract(text)


def chat(client, h, text, thread=None):
    result = client.post("/api/chat", headers=h, json={"message": text, "thread_id": thread})
    assert result.status_code == 200, result.text
    return result.json()


def test_conflict_requires_confirmation_before_any_analysis_and_survives_reload():
    with TestClient(app) as client:
        token, _ = register(client, "symptom_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我咳嗽，没有咳嗽")
        tid = first["thread_id"]
        assert first["intent_type"] == "consult"
        assert first["trace"]["inquiry"]["pending"]["kind"] == "conflict"
        assert not {e["stage"] for e in first["trace"]["events"]} & {"search", "formula", "bianzheng"}
        saved = json.loads(one("SELECT state FROM thread_safety_state WHERE thread_id=?", (tid,))["state"])
        assert len(saved["symptom_facts"]) == 2
        # Authoritative business state remains usable without any graph checkpoint.
        unknown = chat(client, h, "不知道", tid)
        assert unknown["intent_type"] == "consult"
        assert unknown["trace"]["inquiry"]["pending"]["kind"] == "conflict"
        assert unknown["trace"]["inquiry"]["pending"]["id"] == first["trace"]["inquiry"]["pending"]["id"]
        resolved = chat(client, h, "现在没有咳嗽", tid)
        assert resolved["trace"]["inquiry"]["pending"] is None
        assert "咳嗽" not in resolved["symptoms"]
        assert len(resolved["trace"]["inquiry"]["facts"]) == 3
        assert any(f.get("superseded_by") for f in resolved["trace"]["inquiry"]["facts"])


def test_targeted_short_answer_and_latest_correction():
    with TestClient(app) as client:
        token, _ = register(client, "answer_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "不知道有没有咳嗽")
        assert first["trace"]["inquiry"]["pending"]["symptom"] == "咳嗽"
        second = chat(client, h, "有", first["thread_id"])
        assert second["trace"]["inquiry"]["pending"] is None
        assert second["symptoms"] == ["咳嗽"]
        assert "question" in second["trace"]["inquiry"]["facts"][-1]["source"]
        third = chat(client, h, "现在没有咳嗽了", first["thread_id"])
        assert "咳嗽" not in third["symptoms"]
        assert not {e["stage"] for e in third["trace"]["events"]} & {"search", "formula"}


def test_prior_time_not_searched_as_current_and_degree_reaches_analysis(monkeypatch):
    captured = []
    original = workflow._context
    def context(state):
        value = original(state)
        captured.append(value)
        return value
    monkeypatch.setattr(workflow, "_context", context)
    with TestClient(app) as client:
        token, _ = register(client, "temporal_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "以前咳嗽，现在没有咳嗽，头痛三天且不明显")
        assert first["symptoms"] == ["头痛"]
        assert "不明显" in first["trace"]["inquiry"]["facts"][-1]["degree_text"]
        assert any("不明显" in item and "past" in item for item in captured)


def test_subject_separation_and_ownership_question():
    with TestClient(app) as client:
        token, _ = register(client, "patient_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "母亲咳嗽，我没有咳嗽")
        assert first["intent_type"] == "consult"
        assert first["trace"]["inquiry"]["pending"]["kind"] == "subject"
        second = chat(client, h, "本次替母亲咨询", first["thread_id"])
        assert second["symptoms"] == ["咳嗽"]
        assert second["trace"]["inquiry"]["pending"] is None
        third = chat(client, h, "改为替我咨询，我头痛三天", first["thread_id"])
        assert third["symptoms"] == ["头痛"]
        assert {f["subject"] for f in third["trace"]["inquiry"]["facts"]} == {"self", "other:母亲"}


def test_prior_self_identity_does_not_answer_new_attachment_ownership():
    with TestClient(app) as client:
        token, _ = register(client, "attachment_" + uuid4().hex[:8])
        h = auth(token)
        first = chat(client, h, "我头痛三天")
        upload = client.post("/api/upload", headers=h,
                             files={"file": ("symptom.txt", "咳嗽不明显".encode("utf-8"))}).json()["id"]
        result = client.post("/api/chat", headers=h,
                             json={"message": "请看附件", "files": [upload], "thread_id": first["thread_id"]}).json()
        assert result["trace"]["inquiry"]["pending"]["kind"] == "subject"
        not_answered = chat(client, h, "三天", first["thread_id"])
        assert not_answered["trace"]["inquiry"]["pending"]["kind"] == "subject"
        confirmed = chat(client, h, "这份附件是我本人的", first["thread_id"])
        assert confirmed["trace"]["inquiry"]["pending"] is None
        fact = next(f for f in confirmed["trace"]["inquiry"]["facts"] if f["source"]["kind"] == "attachment")
        assert fact["source"]["upload_ids"] == [upload]
        assert fact["subject"] == "self" and fact.get("ownership_confirmed_by")


@pytest.mark.parametrize("text,positive", [("我耳朵嗡嗡响三天", ["耳鸣"]), ("我没有胃口", ["食欲不振"]),
                                          ("腰疼，不咳嗽", ["腰痛"]), ("最近胃不舒服", ["胃部不适"]),
                                          ("腰有点酸", ["腰酸"]), ("晚上容易出汗", ["出汗"])])
def test_expanded_catalog_aliases_keep_original_evidence(text, positive):
    facts = symptoms.extract(text)
    assert symptoms.view(facts, "self")[0] == positive
    assert all(f["source"]["quote"] in text for f in facts)


def test_unmapped_symptom_is_confirmed_and_searched_without_inventing_translation(monkeypatch):
    calls = []
    async def capture(plan):
        calls.append(plan)
        return [], {"warning": ""}
    monkeypatch.setattr(workflow, "retrieve_classics", capture)
    with TestClient(app) as client:
        token, _ = register(client, "unmapped_" + uuid4().hex[:8])
        first = chat(client, auth(token), "我有头皮发紧三天")
        assert first["intent_type"] == "consult" and calls == []
        assert first["trace"]["inquiry"]["pending"]["kind"] == "mapping"
        second = chat(client, auth(token), "是的", first["thread_id"])
        assert second["symptoms"] == ["头皮发紧"]
        assert second["trace"]["inquiry"]["pending"] is None
        assert len(calls) == 3
        for plan in calls:
            assert plan["keyword_query"] == "头皮发紧" and "三天" in plan["semantic_query"]
            assert plan["unmapped"] == ["头皮发紧"] and not plan["expansions"]
        assert any(f.get("mapping_confirmed") for f in second["trace"]["inquiry"]["facts"])


def test_unmapped_mixed_with_known_and_denied_remains_separate():
    facts = symptoms.extract("头痛但没有头皮发紧，牙龈灼热")
    positive, denied, pending = symptoms.view(facts, "self")
    assert positive == ["头痛"] and denied == ["头皮发紧"]
    assert pending["symptom"] == "牙龈灼热"
    assert not symptoms.extract("我有一个代码问题")
    assert not symptoms.extract("我有药物过敏史")


def test_ambiguous_colloquial_term_is_not_forced_into_a_clinical_equivalent():
    facts = symptoms.extract("我有打嗝三天")
    assert len(facts) == 1 and facts[0]["symptom"] == "打嗝" and facts[0]["unmapped"]
    assert symptoms.view(facts, "self")[2]["kind"] == "mapping"
