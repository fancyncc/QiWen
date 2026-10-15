from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import threading

import pytest

from app import classics, db, evidence_sources, local_ranking, retrieval, terms
from app.query_plan import knowledge_plan
from evaluation.calibration import calibrate
from evaluation.evidence_audit import audit_report, fingerprint, locate_gold


def test_related_terms_are_frontloaded_without_becoming_patient_facts():
    state = {"symptoms": ["失眠", "疲劳"], "symptom_facts": [
        {"symptom": "失眠", "subject": "self", "time_scope": "current", "assertion": "present", "duration_text": "三周"},
        {"symptom": "口苦", "subject": "self", "time_scope": "current", "assertion": "absent"},
        {"symptom": "咳嗽", "subject": "other:母亲", "time_scope": "current", "assertion": "present"}]}
    original = copy.deepcopy(state)
    plan = terms.search_plan(state)
    assert state == original and plan["primary"] == ["失眠", "疲劳"]
    assert {"不寐", "不得眠", "倦怠"} <= set(plan["lexical_terms"])
    assert "三周" in plan["semantic_query"]
    assert "口苦" not in plan["rerank_query"] and "咳嗽" not in plan["keyword_query"]
    assert len(plan["recall_queries"]) == 2 and plan["expansions"]
    # A narrower retry has a different query and must not reuse the first
    # no-match result. Merely changing its attempt counter still shares cache.
    assert classics.retrieval_cache_key(plan) != classics.retrieval_cache_key(terms.search_plan(state, 2))
    assert classics.retrieval_cache_key(plan) == classics.retrieval_cache_key({**plan, "attempt": 99})


def test_expansion_catalog_rejects_invented_source_quote(tmp_path):
    data = json.loads(terms.CATALOG_PATH.read_text(encoding="utf-8"))
    expansion = next(e for entry in data["entries"] for e in entry["expansions"])
    expansion["evidence"]["quote"] = expansion["term"] + "伪造内容"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        terms.load_catalog(path)


def test_late_relevant_sentence_keeps_exact_offsets_and_heading():
    body = "无关的背景文字。" * 90 + "不得眠，心中烦。此段用于说明睡眠相关原文。" + "附录。" * 120
    row = {"id": "test", "title": "测试古籍", "chapter": "睡眠章", "body": body}
    window = classics.scoring_window(row, terms.search_plan({"symptoms": ["失眠"]}))
    assert window["start"] > 500 and "不得眠" in window["quote"]
    assert window["complete_sentences"] and len(window["quote"]) <= 380
    assert body[window["start"]:window["end"]] == window["quote"]
    assert window["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert "测试古籍 · 睡眠章\n" in window["text"]


def test_relevant_clause_does_not_fill_window_with_unrelated_prescriptions():
    body = "虚劳虚烦不得眠，酸枣汤主之。\n\n酸枣汤方：\n用药原文。\n\n另一段无关的原文。"
    row = {"title": "金匮要略", "chapter": "血痹虚劳病脉证并治", "body": body}
    window = classics.scoring_window(row, terms.search_plan({"symptoms": ["失眠", "疲劳"]}))
    assert "不得眠" in window["quote"] and "酸枣汤方" not in window["quote"]
    assert window["complete_sentences"] and window["quote"] in body


def test_source_registry_separates_definition_and_named_classics():
    concepts = evidence_sources.matching_concepts(knowledge_plan("什么是阴虚？"))
    assert len(concepts) == 1
    assert not evidence_sources.matching_concepts(knowledge_plan("《伤寒论》中阴虚是什么？"))
    assert not evidence_sources.matching_concepts(knowledge_plan("阴虚和阳虚的区别是什么？"))
    assert not evidence_sources.matching_concepts(knowledge_plan("什么是阴虚和阳虚？"))
    assert not evidence_sources.matching_concepts(knowledge_plan("ZXQWUNLISTED2026 是什么？"))
    assert not evidence_sources.matching_concepts(terms.search_plan({"symptoms": ["失眠"]}))


def test_concept_still_needs_gate_and_revalidates_source(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sources.sqlite3")
    db.init_db()
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    def unexpected(*args):
        raise AssertionError("Definition source must not query the classical vector index")
    monkeypatch.setattr(retrieval, "_embed", unexpected)
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *_: [-3.] * len(texts))
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("什么是阴虚？")))
    assert not docs and trace["rerank"]["ranked"][0]["rejection_reason"] == "below_relevance_threshold"
    assert trace["vector_status"] == "not_applicable_concept_source"
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *_: [math.log(9)] * len(texts))
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("什么是阴虚？")))
    assert len(docs) == 1 and trace["selected"][0]["evidence"] in docs[0]["body"]
    monkeypatch.setattr(evidence_sources, "concept_rows", lambda *args: [])
    assert not classics.current_sources(docs)


def test_token_diagnostics_measure_actual_encoder_pair(monkeypatch):
    class Values:
        def tolist(self):
            return [1.]
    class Encoder:
        def tokenizer(self, query, passage, **kwargs):
            assert kwargs == {"truncation": False}
            return {"input_ids": list(range(600))}
        def predict(self, pairs, **kwargs):
            return Values()
    monkeypatch.setattr(local_ranking, "_load", lambda *args: Encoder())
    audit = []
    token = local_ranking.score_audit.set(audit)
    try:
        assert local_ranking.score_logits("问题", ["片段"], local_ranking.settings(), threading.Event()) == [1.]
    finally:
        local_ranking.score_audit.reset(token)
    assert audit == [{"status": "measured", "input_token_count": 600, "max_length": 512, "truncated": True}]


@pytest.mark.parametrize("stage", ["source_absent", "recall_miss", "pool_exclusion", "rerank_not_completed",
                                   "relevance_gate_rejection", "post_gate_exclusion", "selected"])
def test_gold_source_loss_is_located_at_exact_stage(stage):
    trace = {"recall_audit": [{"id": "gold", "in_rerank_pool": True}], "rerank": {"ranked": [
        {"id": "gold", "accepted": True, "selected": True}]}}
    source_ids = {"gold"}
    if stage == "source_absent":
        source_ids.clear()
    elif stage == "recall_miss":
        trace["recall_audit"] = []
    elif stage == "pool_exclusion":
        trace["recall_audit"][0]["in_rerank_pool"] = False
    elif stage == "rerank_not_completed":
        trace["rerank"]["ranked"] = []
    elif stage == "relevance_gate_rejection":
        trace["rerank"]["ranked"][0]["accepted"] = False
    elif stage == "post_gate_exclusion":
        trace["rerank"]["ranked"][0]["selected"] = False
    assert locate_gold("gold", trace, source_ids) == stage


def calibration_samples():
    tasks, labels = [], []
    for split in ("development", "validation"):
        for kind in ("concept_definition", "clinical_classics"):
            for positive, score in ((True, .85), (False, .2)):
                key = f"{split}:{kind}:{positive}"
                task = {"task_id": key, "family": key, "query_kind": kind, "doc_id": key,
                        "window_sha256": key, "score": score}
                task["binding_sha256"] = fingerprint(task)
                tasks.append(task)
                labels.append({"task_id": key, "binding_sha256": task["binding_sha256"], "split": split,
                    "label_status": "expert_reviewed", "reviewer": "synthetic_test_reviewer", "reviewed_at": "2026-10-12",
                    "rationale": "synthetic fixture only", "supported_scope": "fixture question" if positive else "none",
                    "passage_relevant": positive, "supports_question": positive, "hard_negative": not positive})
    return {"tasks": tasks}, labels


def test_calibration_needs_independent_labels_and_enough_samples():
    audit, labels = calibration_samples()
    assert calibrate(audit, labels)["status"] == "blocked"  # Eight fixtures cannot calibrate real medicine.
    result = calibrate(audit, labels, min_per_class=1)
    assert result["status"] == "proposal_ready_for_review" and not result["runtime_threshold_changed"]
    labels[0]["label_status"] = "engineering_contract"
    assert calibrate(audit, labels, min_per_class=1)["status"] == "blocked"


def test_calibration_rejects_score_edits_and_cross_split_duplicates():
    audit, labels = calibration_samples()
    changed = copy.deepcopy(audit)
    changed["tasks"][0]["score"] = .99
    with pytest.raises(ValueError, match="binding changed"):
        calibrate(changed, labels)
    audit["tasks"][4]["family"] = audit["tasks"][0]["family"]
    audit["tasks"][4]["binding_sha256"] = fingerprint({k: v for k, v in audit["tasks"][4].items() if k != "binding_sha256"})
    labels[4]["binding_sha256"] = audit["tasks"][4]["binding_sha256"]
    assert "query_family_split_leakage" in calibrate(audit, labels, min_per_class=1)["reasons"]


def test_high_scoring_hard_negative_cannot_make_calibration_green():
    audit, labels = calibration_samples()
    for task, label in zip(audit["tasks"], labels):
        if label["hard_negative"]:
            task["score"] = .95
            task["binding_sha256"] = fingerprint({k: v for k, v in task.items() if k != "binding_sha256"})
            label["binding_sha256"] = task["binding_sha256"]
    assert calibrate(audit, labels, min_per_class=1)["status"] == "blocked"


def test_audit_keeps_each_followup_turn_and_unscored_recall():
    attempt = {"outcome": "no_match", "evidence_kind": "clinical_classics", "recall_audit": [
        {"id": "outside", "in_rerank_pool": False}], "rerank": {"ranked": []}}
    report = {"workloads": [{"details": [{"id": "followup", "mode": "fixed", "turn_results": [
        {"retrieval": {"attempts": [attempt]}}, {"retrieval": {"attempts": [attempt]}}]}]}]}
    audit = audit_report(report)
    assert len(audit["cases"]) == len(audit["tasks"]) == 2
    assert audit["tasks"][0]["score"] is None
    assert audit["tasks"][0]["rejection_reason"] == "excluded_from_rerank_pool"


def test_stage_diagnostics_do_not_consume_model_tool_evidence_budget():
    from app.context import project_observation
    original = {"tool": "search_classics", "result": {"status": "success", "documents": [
        {"id": "source", "title": "古籍", "chapter": "章节", "excerpt": "可引用原文。",
         "retrieval": {"channel_hits": ["large diagnostic" * 1000]}}],
        "retrieval_trace": {"input_windows": "large diagnostic" * 1000}, "allergy_risks": ["甘草"]}}
    packed = project_observation(original, 500)
    assert packed["result"]["documents"][0]["id"] == "source"
    assert packed["result"]["allergy_risks"] == ["甘草"]
    assert "retrieval_trace" not in packed["result"] and "retrieval" not in packed["result"]["documents"][0]
    assert "retrieval_trace" in original["result"]  # Full trace remains reviewable.


def test_detailed_query_export_requires_synthetic_driver_opt_in():
    from evaluation.deployment import retrieval_summary
    trace = {"outcome": "no_match", "query_plan": {"semantic_query": "private query"},
             "rerank": {"ranked": [{"id": "source", "score": .6, "evidence": "private window"}],
                        "input_windows": [{"quote": "private window"}]}}
    data = {"trace": {"classics_retrieval": {"final": trace}}}
    compact = str(retrieval_summary(data))
    assert "private query" not in compact and "private window" not in compact
    detailed = retrieval_summary(data, include_synthetic_diagnostics=True)
    assert detailed["attempts"][0]["query_plan"] == trace["query_plan"]
