from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app import classics, db, local_ranking, model, retrieval, workflow
from app.query_plan import knowledge_plan
from app.terms import search_plan
from evaluation import runtime


@pytest.fixture
def passages(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "corpus.sqlite3")
    db.init_db()
    with db.connection() as connection:
        connection.execute("DELETE FROM corpus")
        connection.executemany("INSERT INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)", [
            ("one", "合成知识甲", "阴虚", "阴虚者常见津液不足的合成说明。", "now"),
            ("two", "合成知识乙", "阴虚", "阴虚与津液的另一段合成说明。", "now"),
        ])
    monkeypatch.setattr(retrieval, "RAG_MODE", "bm25")
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "none")
    return knowledge_plan("请解释阴虚的含义和津液的关系？")


def test_natural_question_and_unknown_query_have_distinct_evidence_results(passages):
    assert passages == runtime.query_plan("请解释阴虚的含义和津液的关系？")
    assert {"阴虚", "津液"} <= set(passages["lexical_terms"])
    assert passages["semantic_query"] not in passages["lexical_terms"]
    docs, trace = asyncio.run(classics.retrieve_classics(passages))
    assert docs and trace["outcome"] == "success"
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("ZXQWUNLISTED2026")))
    assert not docs and trace["outcome"] == "no_match"
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("ZXQWUNLISTED2026 是什么？")))
    assert not docs and trace["outcome"] == "no_match" and trace["rerank"]["input_count"] == 0


def test_named_source_constrains_candidates_without_using_gold_labels(passages):
    plan = knowledge_plan("《合成知识乙》如何解释阴虚？")
    docs, _ = asyncio.run(classics.retrieve_classics(plan))
    assert [doc["id"] for doc in docs] == ["two"]
    docs, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("《不存在的合成书》如何解释阴虚？")))
    assert not docs and trace["outcome"] == "no_match" and trace["corpus_count"] == 2


def test_generic_request_words_cannot_admit_passage(passages):
    assert not knowledge_plan("请介绍一下资料的含义？")["lexical_terms"]
    docs, _ = asyncio.run(classics.retrieve_classics(knowledge_plan("请介绍一下资料的含义？")))
    assert not docs


def test_candidate_timeout_stops_retries_and_retains_timing(passages, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_TOTAL_TIMEOUT", ".04")
    stopped = threading.Event()
    def slow(*_):
        from app.blocking import check_cancelled
        try:
            while True:
                time.sleep(.005)
                check_cancelled()
        finally:
            stopped.set()
    monkeypatch.setattr(retrieval, "retrieve_candidates", slow)
    async def run():
        token = model.invocation.set(model.InvocationContext())
        try:
            start = time.monotonic()
            docs, trace = await classics.retrieve_classics(passages)
            assert not docs and trace["outcome"] == "timeout" and not trace["retryable"]
            assert trace["timings"]["candidate_retrieval"]["elapsed_ms"] >= 20
            assert workflow.retrieval_route({"retrieval_trace": trace, "search_attempts": 1}) == "ready"
            await classics.retrieve_classics(knowledge_plan("不同的问题"))
            assert time.monotonic() - start < .2
            await asyncio.to_thread(stopped.wait, .2)
            assert stopped.is_set()
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


def test_rerank_timeout_preserves_candidate_dependency_diagnostics(passages, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_TOTAL_TIMEOUT", ".04")
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    def slow(query, texts, config, stop):
        stop.wait(.2)
        local_ranking.check_stop(stop)
    monkeypatch.setattr(local_ranking, "score_logits", slow)
    _, trace = asyncio.run(classics.retrieve_classics(passages))
    assert trace["outcome"] == "timeout"
    assert trace["candidate_count"] == 2 and trace["candidate_pool_limit"] == 12
    assert trace["vector_status"] == "disabled"


def test_readiness_detects_stale_vector_ids_even_when_counts_match(passages, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "RAG_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "API_BASE", "https://synthetic.invalid")
    monkeypatch.setattr(retrieval, "API_KEY", "synthetic")
    monkeypatch.setenv("QIWEN_EMBEDDING_MODEL", "synthetic")
    class Collection:
        def count(self):
            return 2
        def get(self, **kwargs):
            return {"ids": ["one", "obsolete"]}
    monkeypatch.setattr(retrieval, "_collection", Collection)
    result = asyncio.run(main.readiness())
    assert result["status"] == "degraded" and result["checks"]["vector_index"]["covered_count"] == 1
    rendered = str(result)
    assert "https://synthetic.invalid" not in rendered and "obsolete" not in rendered


def test_no_match_does_not_retry_with_insufficient_remaining_budget(passages, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_TOTAL_TIMEOUT", "1")
    original = retrieval.retrieve_candidates
    def slow(*args):
        time.sleep(.65)
        return original(*args)
    monkeypatch.setattr(retrieval, "retrieve_candidates", slow)
    _, trace = asyncio.run(classics.retrieve_classics(knowledge_plan("ZXQWUNLISTED2026")))
    assert trace["outcome"] == "no_match" and not trace["retryable"]
    assert trace["remaining_ms"] < trace["retry_minimum_ms"]
    state = {"symptoms": ["咳嗽"], "search_attempts": 1, "search_plan": search_plan({"symptoms": ["咳嗽"]}),
             "retrieval_trace": trace, "retrieval_score": 0}
    assert workflow.retrieval_route(state) == "ready"


def test_deployment_summary_keeps_all_fixed_attempts_and_rerank_diagnostics():
    from evaluation.deployment import retrieval_summary
    first = {"outcome": "no_match", "admitted_count": 0, "query_plan": {"semantic_query": "private synthetic context"},
             "rerank": {"status": "success", "ranked": [{"score": .6}], "threshold": .7}}
    last = {"outcome": "timeout"}
    result = retrieval_summary({"trace": {"classics_retrieval": {"attempts": [first, last], "final": last}}})
    assert len(result["attempts"]) == 2 and result["retrieval_failures"] == 1
    assert result["attempts"][0]["rerank"]["highest_score"] == .6
    assert "private synthetic context" not in str(result)


def test_rewrites_share_one_deadline(passages, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_TOTAL_TIMEOUT", ".08")
    original = retrieval.retrieve_candidates
    def slow(*args):
        time.sleep(.045)
        return original(*args)
    monkeypatch.setattr(retrieval, "retrieve_candidates", slow)
    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            _, first = await classics.retrieve_classics(knowledge_plan("ZXQWUNLISTED2026"))
            end = context.retrieval_deadline
            _, second = await classics.retrieve_classics(knowledge_plan("ANOTHERUNLISTED2026"))
            assert first["outcome"] == "no_match" and second["outcome"] == "timeout"
            assert end == context.retrieval_deadline
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


def test_mmr_timeout_preserves_only_admitted_evidence(passages, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_TOTAL_TIMEOUT", ".15")
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *_: [2.0] * len(texts))
    def slow(rows, config, stop):
        stop.wait(1)
        local_ranking.check_stop(stop)
    monkeypatch.setattr(local_ranking, "candidate_embeddings", slow)
    docs, trace = asyncio.run(classics.retrieve_classics(passages))
    assert len(docs) == 2 and trace["mmr"]["status"] == "failed" and trace["outcome"] == "success"
    assert all(row["retrieval"]["rerank_score"] >= .7 for row in docs)


def test_expired_inference_queue_never_loads_model(monkeypatch):
    from app.blocking import deadline
    monkeypatch.setattr(local_ranking, "_load", lambda *_: pytest.fail("Expired work must not load a model"))
    local_ranking._inference_lock.acquire()
    async def run():
        token = deadline.set(time.monotonic() + .03)
        try:
            with pytest.raises(TimeoutError):
                await local_ranking.bounded_worker(local_ranking.score_logits, "synthetic", ["text"], local_ranking.settings(), timeout=.1)
        finally:
            deadline.reset(token)
    try:
        asyncio.run(run())
    finally:
        local_ranking._inference_lock.release()


def test_cancelled_invocation_cannot_reuse_cached_evidence(passages):
    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            await classics.retrieve_classics(passages)
            context.cancelled = True
            with pytest.raises(asyncio.CancelledError):
                await classics.retrieve_classics(passages)
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


def test_no_rewrite_for_identical_or_failed_query():
    state = {"symptoms": ["自定义不适"], "search_attempts": 1, "retrieval_score": 0,
             "search_plan": search_plan({"symptoms": ["自定义不适"]})}
    assert workflow.retrieval_route(state) == "ready"
    state["retrieval_trace"] = {"outcome": "model_unavailable", "retryable": False}
    state["symptoms"] = ["头痛"]
    assert workflow.retrieval_route(state) == "ready"


def test_rag_gate_rejects_vacuous_valid_sources():
    result = {"status": "completed", "count": 2, "invalid_sources": 0,
              "evidence_contract": {"positive_count": 1, "negative_count": 1, "failed": 1, "empty_positive_count": 1}}
    assert runtime.gate({"rag": result}, ["rag"])["status"] == "failed"
    result["evidence_contract"].update(positive_count=2, empty_positive_count=1)
    gate = runtime.gate({"rag": result}, ["rag"])
    assert gate["status"] == "passed" and gate["retrieval_effectiveness"]["status"] == "failed"


def test_warmup_uses_only_local_cached_weights(monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_MMR_EMBEDDING_BACKEND", "local")
    seen = []
    monkeypatch.setattr(local_ranking, "_load", lambda kind, config, local_only: seen.append((kind, local_only)))
    result = asyncio.run(local_ranking.warm_models())
    assert result["status"] == "available" and seen == [("cross_encoder", True), ("embedding", True)]


def test_missing_warmup_weights_do_not_claim_readiness(monkeypatch):
    def missing(*_):
        raise local_ranking.ModelUnavailable("synthetic missing weights")
    monkeypatch.setattr(local_ranking, "_load", missing)
    result = asyncio.run(local_ranking.warm_models())
    assert result["status"] == "unavailable" and result["error_type"] == "ModelUnavailable"


def test_readiness_reports_failed_startup_models(passages, monkeypatch):
    from app import main
    monkeypatch.setattr(main.app.state, "retrieval_warmup", {"status": "unavailable"}, raising=False)
    result = asyncio.run(main.readiness())
    assert result["status"] == "degraded"


def test_agent_query_cannot_inject_denied_or_other_patient_symptoms(passages, monkeypatch):
    from app.agents import tools
    seen = []
    async def capture(plan):
        seen.append(plan)
        return [], {"outcome": "no_match", "warning": "", "selected": []}
    monkeypatch.setattr(tools, "retrieve_classics", capture)
    asyncio.run(tools.search_classics({"identity_mode": "clinical", "symptoms": ["失眠"],
                                      "safety_state": {"subject": "self"}}, tools.SearchArgs(query="没有口苦，母亲头痛")))
    assert all(word not in seen[0]["semantic_query"] for word in ("口苦", "头痛", "母亲"))


def test_cancelled_worker_keeps_capacity_until_actual_work_stops(monkeypatch):
    monkeypatch.setattr(local_ranking, "_worker_slots", threading.BoundedSemaphore(1))
    entered, release = threading.Event(), threading.Event()
    def work(stop):
        entered.set()
        release.wait(1)
    async def run():
        first = asyncio.create_task(local_ranking.bounded_worker(work, timeout=1))
        await asyncio.to_thread(entered.wait, .5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(local_ranking.InferenceBusy):
            await local_ranking.bounded_worker(work, timeout=.1)
        release.set()
    try:
        asyncio.run(run())
    finally:
        release.set()
