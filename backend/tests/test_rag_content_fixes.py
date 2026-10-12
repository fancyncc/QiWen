from __future__ import annotations

import asyncio
import copy
import hashlib

import pytest

from app import classics, db, local_ranking, retrieval
from app.query_plan import knowledge_plan
from data_pipeline.cleaning import clean_source_text
from evaluation.benchmark import relevant
from evaluation.benchmark import ClaimFaithfulnessVerdict, validate_verdict, validate_claims
from evaluation.evidence_audit import audit_report, deduplicate_tasks, fingerprint


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "corpus.sqlite3")
    db.init_db()
    monkeypatch.setattr(retrieval, "RAG_MODE", "bm25")
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "none")


def test_domain_single_terms_and_book_scope_are_separate():
    plan = knowledge_plan("《素问》如何解释怒、喜、悲、恐等气机变化？")
    assert {"怒", "喜", "悲", "恐"} <= set(plan["lexical_terms"])
    assert "恐等" not in plan["lexical_terms"] and "素问" not in plan["lexical_terms"]
    assert plan["source_hints"] == ["素问"]
    assert not knowledge_plan("《素问》请介绍一下资料？")["lexical_terms"]
    assert not knowledge_plan("这句话说了几个字？").get("required_term_groups")


def test_aliases_have_identical_bm25_scores_without_changing_quotes():
    rows = [{"id": "target", "body": "黄耆味甘，主补虚。", "chapter": "黄耆"},
            {"id": "other", "body": "甘草味甘。", "chapter": "甘草"}]
    assert retrieval.bm25_scores("黄芪", rows) == retrieval.bm25_scores("黄耆", rows)
    assert knowledge_plan("黄耆在《神农本草经》中主治什么？")["required_entities"] == ["黄芪"]
    assert rows[0]["body"].startswith("黄耆")


def test_book_mentions_cannot_admit_unrelated_drug_or_preamble():
    plan = knowledge_plan("黄耆在《神农本草经》中主治什么？")
    rows = [{"id": "preamble", "body": "《神农本草经》上药养命，中药养性。", "title": "神农本草经", "chapter": "序言", "retrieval": {}},
            {"id": "other", "body": "甘草味甘平，主治百病。", "title": "神农本草经", "chapter": "草部", "retrieval": {}},
            {"id": "target", "body": "黄芪味甘微温，主治痈疽，补虚。", "title": "神农本草经", "chapter": "草部", "retrieval": {}}]
    admitted = classics.fallback_admission(rows, plan)
    assert [row["id"] for row in admitted] == ["target"]
    assert admitted[0]["retrieval"]["evidence"] in rows[-1]["body"]


@pytest.mark.parametrize("phrase", ["不可发汗", "不可发其汗", "不可汗", "勿汗"])
def test_negative_action_survives_and_affirmative_passage_is_rejected(phrase):
    plan = knowledge_plan("哪些脉证属于病不可发汗？")
    assert "不可发汗" in plan["lexical_terms"]
    rows = [{"id": "positive", "body": "脉浮，病可发汗。", "title": "脉经", "chapter": "病可发汗", "retrieval": {}},
            {"id": "negative", "body": f"尺中脉微者，{phrase}。", "title": "脉经", "chapter": "禁忌", "retrieval": {}}]
    assert [r["id"] for r in classics.fallback_admission(rows, plan)] == ["negative"]


def test_relation_coverage_precedes_pool_truncation(corpus, monkeypatch):
    rows = [{"id": f"incidental{i}", "title": "黄帝内经-素问", "chapter": "其他", "body": "阳气阳气阳气，生长收藏。", "metadata": "{}"} for i in range(15)]
    rows.append({"id": "gold", "title": "黄帝内经-素问", "chapter": "生气通天论", "body": "阳气者，若天与日，失其所则折寿而不彰。", "metadata": "{}"})
    monkeypatch.setattr(retrieval, "all_rows", lambda *_: rows)
    candidates, trace = retrieval.retrieve_candidates(knowledge_plan("《素问》阳气与天日的关系是什么？"), candidate_limit=4)
    assert candidates[0]["id"] == "gold"
    assert next(r for r in trace["recall_audit"] if r["id"] == "gold")["in_rerank_pool"]
    assert classics.scoring_window(candidates[0], knowledge_plan("阳气与天日的关系是什么？"))["quote"] == rows[-1]["body"]


def test_cross_encoder_high_score_cannot_override_explicit_negation(corpus, monkeypatch):
    with db.connection() as conn:
        conn.execute("DELETE FROM corpus")
        conn.executemany("INSERT INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)", [
            ("positive", "脉经", "病可发汗", "脉浮，病可发汗。", "now"),
            ("negative", "脉经", "病不可发汗", "尺中脉微者，不可发其汗。", "now")])
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *_: [3.] * len(texts))
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("哪些脉证属于病不可发汗？")))
    assert [r["id"] for r in docs] == ["negative"]
    positive = next(r for r in trace["rerank"]["ranked"] if r["id"] == "positive")
    assert positive["score"] > .7 and positive["rejection_reason"] == "negative_action_missing"


def test_footer_cleaning_preserves_medical_negation_and_source_ranges():
    medical = "尺中脉微者，不可发汗。\n"
    text = medical + "URN\n喜欢\n我们的网站\n网站的设计与内容\n广告推广\n### 下卷\n黄芪味甘，主补虚。"
    cleaned, audit = clean_source_text(text, "神农本草经")
    assert cleaned == medical + "黄芪味甘，主补虚。"
    assert audit["raw_body_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert audit["removed_source_ranges"] and "不可发汗" in cleaned


def test_proxy_scores_actual_quote_and_open_source_scope():
    case = {"expected_books": [], "keywords": ["不可发汗"]}
    row = {"title": "伤寒论", "body": "脉微，不可发汗。脉浮，可发汗。",
           "retrieval": {"evidence": "脉浮，可发汗。"}}
    assert not relevant(row, case)
    row["retrieval"]["evidence"] = "脉微，不可发汗。"
    assert relevant(row, case)
    assert not relevant(row, {**case, "expected_books": ["脉经"]})


def test_formula_window_retains_prescription_instead_of_only_symptoms():
    plan = knowledge_plan("妇人宿有癥病、妊娠漏下时记载何方？")
    body = "妇人宿有症病，经断未及三月，而得漏下不止。胎动在脐上者，为症痼害。" + "此段为条件说明。" * 5 + "桂枝茯苓丸主之。"
    row = {"title": "金匮要略", "chapter": "妇人妊娠", "body": body}
    window = classics.scoring_window(row, plan)
    assert "桂枝茯苓丸主之" in window["quote"] and "漏下" in window["quote"]
    assert "喘" in knowledge_plan("太阳伤寒无汗而喘时用什么方？")["lexical_terms"]
    assert ["胃家实"] in knowledge_plan("阳明病提纲为何称胃家实？")["required_term_groups"]
    definition = classics.scoring_window({"title": "伤寒论", "chapter": "太阳病",
        "body": "太阳之为病，脉浮，头项强痛而恶寒。太阳病，发热汗出，恶风。"}, knowledge_plan("太阳病的提纲证是什么？"))
    assert "太阳之为病" in definition["quote"]


def test_comparison_does_not_drop_one_action_polarity():
    plan = knowledge_plan("病可发汗与不可发汗的区别是什么？")
    assert plan["action_polarity"] == "contrast_sweat"
    rows = [{"id": key, "title": "脉经", "chapter": "脉证", "body": body, "retrieval": {}}
            for key, body in [("positive", "脉浮，病可发汗。"), ("negative", "尺中脉微，不可发汗。")]]
    assert {row["id"] for row in classics.fallback_admission(rows, plan)} == {"positive", "negative"}


def test_expert_label_is_bound_to_actual_quote_and_source_body():
    quote = "尺中脉微，不可发汗。"
    row = {"id": "source", "body": quote + "另一条原文。", "retrieval": {"evidence": quote}}
    label = {"quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
             "body_sha256": hashlib.sha256(row["body"].encode()).hexdigest(),
             "label_status": "expert_reviewed", "relevance": 3, "supports_question": True}
    case = {"qrels": {"source": 3}, "passage_labels": {"source": [label]}}
    assert relevant(row, case)
    row["retrieval"]["evidence"] = "另一条原文。"
    assert not relevant(row, case)
    row["retrieval"]["evidence"] = quote
    row["body"] = "被修改的来源。" + quote
    assert not relevant(row, case)


def test_live_component_missing_cached_models_is_blocked_before_retrieval(monkeypatch):
    from evaluation.retrieval_regression import run
    async def unavailable():
        return {"status": "unavailable", "error_type": "ModelUnavailable"}
    async def unexpected(*args):
        raise AssertionError("Missing local models must stop the live component run")
    monkeypatch.setattr(local_ranking, "warm_models", unavailable)
    monkeypatch.setattr(classics, "retrieve_classics", unexpected)
    result = asyncio.run(run([{"id": "fixture", "query": "合成问题"}], live=True))
    assert result["status"] == "blocked" and result["passed"] == 0 and result["details"] == []


def test_review_dedup_keeps_distinct_questions_and_scores():
    attempt = {"query_plan": {"semantic_query": "阳气与天日的关系", "primary": ["阳气", "天日"]},
        "evidence_kind": "named_classics", "recall_audit": [{"id": "source", "in_rerank_pool": True}],
        "rerank": {"ranked": [{"id": "source", "score": .8, "evidence": "原文", "selected": True}]}}
    report = {"workloads": [{"details": [{"id": "test", "mode": mode,
              "retrieval": {"attempts": [copy.deepcopy(attempt)]}} for mode in ("fixed", "multi")]}]}
    raw = audit_report(report)
    grouped = deduplicate_tasks(raw)
    assert len(grouped["tasks"]) == 1 and grouped["raw_task_count"] == 2
    task = grouped["tasks"][0]
    assert len(task["occurrences"]) == 2
    assert task["binding_sha256"] == fingerprint({k: v for k, v in task.items() if k != "binding_sha256"})
    report["workloads"][0]["details"][1]["retrieval"]["attempts"][0]["query_plan"]["semantic_query"] = "阳气与天日有什么区别"
    assert len(deduplicate_tasks(audit_report(report))["tasks"]) == 2


def test_bundled_cleaning_migrates_only_original_source_and_is_idempotent(corpus):
    import json
    from app.corpus_seed import RAW_CHUNK_DIR
    chunk = next(row for row in json.loads((RAW_CHUNK_DIR / "神农本草经.json").read_text(encoding="utf-8"))
                 if row["id"] == "SNBCJ_1469")
    raw = chunk["content"].strip()
    raw_id = hashlib.sha256(f"神农本草经\n{raw}".encode()).hexdigest()
    with db.connection() as conn:
        conn.execute("INSERT OR IGNORE INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)",
                     (raw_id, "神农本草经", chunk["metadata"]["category"], raw, "now"))
        conn.execute("INSERT INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)",
                     ("user_import", "用户资料", "用户章节", "用户自行导入的正文资料。", "now"))
    db.init_db()
    assert not db.one("SELECT id FROM corpus WHERE id=?", (raw_id,))
    assert db.one("SELECT body FROM corpus WHERE id='user_import'")["body"] == "用户自行导入的正文资料。"
    before = db.all_rows("SELECT id,body FROM corpus ORDER BY id")
    db.init_db()
    assert db.all_rows("SELECT id,body FROM corpus ORDER BY id") == before


@pytest.mark.parametrize("support,ids,conditions,contradiction", [
    ("supported", ["invented"], [], False), ("supported", [], [], False),
    ("supported", ["actual"], ["遗漏脉证条件"], False), ("contradicted", ["actual"], [], False)])
def test_claim_judge_cannot_invent_sources_or_hide_missing_conditions(support, ids, conditions, contradiction):
    verdict = validate_verdict({"case_id": "case", "score": 4, "contradiction": contradiction, "reason": "fixture",
        "claims": [{"statement": "合成结论", "evidence_ids": ids, "support": support,
                    "missing_conditions": conditions, "reason": "fixture"}]}, ClaimFaithfulnessVerdict, "case")
    with pytest.raises(ValueError):
        validate_claims(verdict, [{"id": "actual"}])
