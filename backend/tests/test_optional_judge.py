import asyncio

from app import config
from evaluation import benchmark, runtime
from evaluation.review import review_queue


def test_live_dependencies_without_judge_do_not_require_or_probe_judge(monkeypatch):
    for name in ("QIWEN_MODEL_BASE_URL", "QIWEN_MODEL_API_KEY", "QIWEN_MODEL_NAME"):
        monkeypatch.setenv(name, "synthetic")
    for name in ("QIWEN_JUDGE_BASE_URL", "QIWEN_JUDGE_API_KEY", "QIWEN_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(benchmark, "probe_model", lambda base, key, model: calls.append(model) or {"available": True})
    assert runtime.dependencies("live_core")["status"] == "ready"
    assert calls == ["synthetic"]
    assert runtime.dependencies("live_core", judge_required=True)["status"] == "blocked"


def test_rule_quality_collects_actual_answer_without_fabricating_scores(monkeypatch):
    monkeypatch.setattr(config, "MODEL_MODE", "openai_compatible")
    case = {"id": "synthetic", "input": "test", "expected": "test"}
    monkeypatch.setattr(benchmark, "load_jsonl", lambda *a: [case])
    monkeypatch.setattr(benchmark, "register", lambda *a: {})
    monkeypatch.setattr(benchmark, "actual_references", lambda *a: ([{"id": "s", "excerpt": "quote"}], True))
    monkeypatch.setattr(benchmark, "chat", lambda *a, **kw: {"status_code": 200, "data": {
        "response": "answer", "trace": {"model_calls": [{"model": "qwen-plus", "status": "success"}]}}})
    result = benchmark.run_quality(None, None, object(), ("fixed", "multi"))
    assert result["count"] == result["automatic_passed"] == 2
    assert result["scored"] == 0 and result["metrics"] == {}
    assert result["model_judge"]["status"] == "not_enabled"
    queue = review_queue({"results": {"quality": result}})
    assert len(queue) == 2 and all(r["status"] == "pending" for r in queue)
    assert queue[0]["answer"] == "answer" and queue[0]["references"][0]["id"] == "s"
    assert all(r["reason"] == "semantic_review_required" for r in queue)


def test_rule_checks_reject_empty_invalid_or_failed_generation(monkeypatch):
    monkeypatch.setattr(config, "MODEL_MODE", "openai_compatible")
    for answer, valid, calls in (("", True, [{"model": "qwen-plus", "status": "success"}]),
                                 ("answer", False, [{"model": "qwen-plus", "status": "success"}]),
                                 ("answer", True, [{"model": "qwen-plus", "status": "failed"}]),
                                 ("answer", True, [{"role": "cross_encoder", "model": "reranker", "status": "success"}])):
        assert not benchmark.answer_contract({"response": answer, "trace": {"model_calls": calls}}, [], valid)["passed"]


def test_pending_semantics_does_not_block_execution_gate():
    result = {"status": "completed", "count": 2, "failed": 0, "scored": 0,
              "semantic_review": {"status": "pending", "count": 2}}
    assert runtime.gate({"quality": result}, ["quality"])["status"] == "passed"


def test_mock_collection_never_claims_real_generation(monkeypatch):
    monkeypatch.setattr(config, "MODEL_MODE", "mock")
    result = benchmark.answer_contract({"response": "mock"}, [], True)
    assert result["checks"]["real_model_executed"] is None


def test_deterministic_safety_clarification_is_not_failed_model_generation(monkeypatch):
    monkeypatch.setattr(config, "MODEL_MODE", "openai_compatible")
    result = benchmark.answer_contract({"response": "clarify subject", "trace": {
        "preflight": {"decision": "clarify"}, "model_calls": []}}, [], True)
    assert result["passed"] and result["deterministic_boundary_answer"]
    assert result["checks"]["real_model_executed"] is False


def test_full_retrieval_regression_accepts_actual_no_match_negative(monkeypatch):
    from app import classics
    from evaluation.retrieval_regression import run
    async def retrieve(*a):
        return [], {"outcome": "no_match"}
    monkeypatch.setattr(classics, "retrieve_classics", retrieve)
    result = asyncio.run(run([{"id": "negative", "query": "synthetic", "expected_evidence": False}], live=False))
    assert result["passed"] == 1


def test_short_citation_ids_resolve_only_to_current_admitted_sources(monkeypatch):
    import pytest
    from app import knowledge, model
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    state = {"message": "question", "docs": [
        {"id": "hash-a", "title": "book-a", "chapter": "chapter-a", "body": "quote-a"},
        {"id": "hash-b", "title": "book-b", "chapter": "chapter-b", "body": "quote-b"}]}
    async def answer(system, user, **kwargs):
        import json
        assert [d["id"] for d in json.loads(user)["evidence"]] == ["S1", "S2"]
        return {"answer": "answer", "source_ids": ["S2"]}
    monkeypatch.setattr(model, "acomplete_json", answer)
    response = asyncio.run(knowledge.answer_knowledge(state))["response"]
    assert "quote-b" in response and "quote-a" not in response
    async def invented(*a, **kw):
        return {"answer": "answer", "source_ids": ["S3"]}
    monkeypatch.setattr(model, "acomplete_json", invented)
    with pytest.raises(RuntimeError):
        asyncio.run(knowledge.answer_knowledge(state))


def test_review_uses_rendered_citations_not_all_retrieved_documents(monkeypatch):
    from app import db
    rows = [{"id": "a", "title": "book-a", "chapter": "chapter-a", "body": "quote-a"},
            {"id": "b", "title": "book-b", "chapter": "chapter-b", "body": "quote-b"}]
    monkeypatch.setattr(db, "all_rows", lambda *a: rows)
    citation = {"id": "b", "title": "book-b", "chapter": "chapter-b", "excerpt": "quote-b", "quote_displayed": True}
    data = {"response": "answer quote-b", "trace": {"answer_citations": [citation],
            "classics_retrieval": {"final": {"selected": rows}}}}
    refs, valid = benchmark.actual_references(data)
    assert valid and [r["id"] for r in refs] == ["b"]
    data["response"] = "answer without quote"
    assert not benchmark.actual_references(data)[1]
    data["trace"]["answer_citations"] = []
    assert benchmark.actual_references(data) == ([], True)
