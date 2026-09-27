from __future__ import annotations

import asyncio
import hashlib
import math
import threading

import pytest

from app import classics, db, local_ranking, retrieval, symptoms, terms, workflow


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.sqlite3")
    db.init_db()
    with db.connection() as connection:
        connection.execute("DELETE FROM corpus")
        connection.executemany("INSERT INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)", [
            ("a", "甲书", "第一章", "头痛头痛的资料", "now"),
            ("b", "乙书", "第二章", "头痛的另一段资料", "now"),
            ("c", "丙书", "第三章", "有关巅顶不适的原文", "now"),
            ("d", "丁书", "第四章", "完全无关的内容", "now"),
        ])
    monkeypatch.setattr(retrieval, "RAG_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "_embed", lambda texts: [[1.0, 0.0] for _ in texts])
    class Collection:
        def count(self):
            return 4
        def query(self, **kwargs):
            return {"ids": [["c", "b", "d", "a"]], "distances": [[0.01, 0.1, 0.4, 0.5]]}
    monkeypatch.setattr(retrieval, "_collection", Collection)
    monkeypatch.setenv("QIWEN_RAG_MIN_SIMILARITY", "")
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    def logits(query, texts, config, stop):
        return [math.log(99) if "巅顶" in text else math.log(4) if "另一段" in text else -3.0 for text in texts]
    monkeypatch.setattr(local_ranking, "score_logits", logits)
    monkeypatch.setattr(local_ranking, "candidate_embeddings", lambda rows, config, stop:
                        [[1.0, 0.0] if row["id"] == "c" else [0.0, 1.0] for row in rows])
    return terms.search_plan({"symptoms": ["头痛"]})


def test_rrf_preserves_independent_ranks_and_deduplicates_ids(corpus):
    rows, info = retrieval.retrieve_candidates(corpus)
    assert info["active_mode"] == "hybrid" and info["candidate_count"] == 4
    by_id = {row["id"]: row["retrieval"] for row in rows}
    assert by_id["c"]["bm25_rank"] is None and by_id["c"]["vector_rank"] == 1
    assert by_id["b"]["bm25_rank"] == 2 and by_id["b"]["vector_rank"] == 2
    assert by_id["b"]["rrf_score"] == pytest.approx(2 / 62)
    assert by_id["b"]["channels"] == ["bm25", "vector"]
    assert len(by_id) == len(rows)


def test_cross_encoder_after_rrf_keeps_semantic_only_hit(corpus):
    docs, trace = asyncio.run(classics.retrieve_classics(corpus, limit=1))
    assert [row["id"] for row in docs] == ["c"]
    assert trace["rerank"]["input_count"] == 4
    assert trace["rerank"]["status"] == "success"
    assert trace["rerank"]["activation"] == "sigmoid" and trace["rerank"]["threshold"] == 0.7
    assert trace["filtered_count"] == 2 and trace["truncated_count"] == 1
    assert workflow.evaluate_retrieval({"docs": docs})["retrieval_score"] == 1
    assert "巅顶不适" in workflow._context({"docs": docs})
    assert docs[0]["retrieval"]["evidence"] in docs[0]["body"]
    assert docs[0]["retrieval"]["rerank_score"] == pytest.approx(0.99)


@pytest.mark.parametrize("bad", [[], [3.0], [float("nan")] * 4, [[1.0]] * 4, [True] * 4, [float("inf")] * 4])
def test_invalid_cross_encoder_output_is_discarded_without_bypassing_cutoff(corpus, monkeypatch, bad):
    monkeypatch.setattr(local_ranking, "score_logits", lambda *args: bad)
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert docs == [] and trace["rerank"]["status"] == "failed"
    assert trace["filtered_count"] == 4 and trace["admitted_count"] == 0


def test_exact_threshold_includes_07_and_rejects_lower(corpus, monkeypatch):
    value = math.log(0.7 / 0.3)
    monkeypatch.setattr(local_ranking, "score_logits",
                        lambda query, texts, config, stop: [value if "巅顶" in text else value - 0.000001 for text in texts])
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert [row["id"] for row in docs] == ["c"]
    assert docs[0]["retrieval"]["rerank_score"] == 0.7
    assert trace["mmr"]["status"] == "not_needed"


def test_cross_encoder_can_reject_all_without_padding(corpus, monkeypatch):
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *args: [-4.0] * len(texts))
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert docs == [] and trace["filtered_count"] == 4 and trace["rerank"]["status"] == "success"


def test_cancelled_rerank_propagates(corpus, monkeypatch):
    def cancelled(*args):
        raise asyncio.CancelledError()
    monkeypatch.setattr(local_ranking, "score_logits", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(classics.retrieve_classics(corpus))


def test_changed_source_is_dropped_after_rerank(corpus, monkeypatch):
    original = local_ranking.score_logits
    def rank(*args):
        with db.connection() as connection:
            connection.execute("DELETE FROM corpus WHERE id='c'")
            connection.execute("UPDATE corpus SET body='替换正文' WHERE id='b'")
        return original(*args)
    monkeypatch.setattr(local_ranking, "score_logits", rank)
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert docs == [] and trace["stale_count"] == 2


def test_missing_model_and_empty_index_are_explicit(corpus, monkeypatch):
    def missing(*args):
        raise local_ranking.ModelUnavailable("weights missing")
    monkeypatch.setattr(local_ranking, "score_logits", missing)
    class Empty:
        def count(self):
            return 0
    monkeypatch.setattr(retrieval, "_collection", Empty)
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert trace["active_mode"] == "bm25" and trace["vector_status"] == "empty_index"
    assert trace["rerank"]["status"] == "model_unavailable" and docs == []
    assert "向量索引为空" in trace["warning"] and "0.7" in trace["warning"]


@pytest.mark.parametrize("vectors", [[], [[float("nan"), 0]], [[0, 0]], [[True, 1]], [[1, 2], [1]]])
def test_invalid_embedding_degrades_without_losing_keywords(corpus, monkeypatch, vectors):
    monkeypatch.setattr(retrieval, "_embed", lambda texts: vectors)
    docs, trace = retrieval.retrieve_candidates(corpus)
    assert trace["vector_status"] == "fallback" and trace["active_mode"] == "bm25"
    assert {d["id"] for d in docs} == {"a", "b"}


def test_explicitly_disabled_mode_is_labeled_as_not_thresholded(corpus, monkeypatch):
    monkeypatch.setenv("QIWEN_RAG_RERANK_MODE", "none")
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert {d["id"] for d in docs} == {"a", "b"} and trace["rerank"]["status"] == "disabled"
    assert "未经过 0.7" in trace["warning"]
    monkeypatch.setenv("QIWEN_RAG_MIN_SIMILARITY", "0.95")
    docs, _ = asyncio.run(classics.retrieve_classics(corpus))
    assert "c" in {d["id"] for d in docs}


def test_mmr_prefers_distinct_passage_with_candidate_pairwise_cosine():
    rows = [{"id": item, "retrieval": {"rerank_score": score, "rrf_rank": rank}}
            for rank, (item, score) in enumerate([("a", .95), ("b", .9), ("c", .8)], 1)]
    docs = local_ranking.mmr_select(rows, [[1, 0], [.99, .01], [0, 1]], 2, .7)
    assert [row["id"] for row in docs] == ["a", "c"]
    assert docs[1]["retrieval"]["max_selected_similarity"] == 0
    assert docs[1]["retrieval"]["mmr_score"] == pytest.approx(.56)
    assert [row["id"] for row in local_ranking.mmr_select(rows, [[1, 0], [.99, .01], [0, 1]], 2, 1)] == ["a", "b"]


def test_pipeline_mmr_uses_only_thresholded_candidates(corpus, monkeypatch):
    seen = []
    def embeddings(rows, config, stop):
        seen.extend(rows)
        return [[1, 0], [0, 1]]
    monkeypatch.setattr(local_ranking, "candidate_embeddings", embeddings)
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert {row["id"] for row in seen} == {"b", "c"}
    assert all(row["retrieval"]["rerank_score"] >= .7 for row in seen)
    assert trace["mmr"]["status"] == "success"
    assert all(row["retrieval"]["mmr_rank"] for row in docs)


def test_mmr_embedding_failure_preserves_only_already_thresholded_results(corpus, monkeypatch):
    monkeypatch.setattr(local_ranking, "candidate_embeddings", lambda *args: [[float("nan"), 1], [1, 0]])
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert trace["mmr"]["status"] == "failed"
    assert {row["id"] for row in docs} == {"b", "c"}
    assert all(row["retrieval"]["rerank_score"] >= .7 for row in docs)


def test_stale_source_is_checked_again_after_mmr(corpus, monkeypatch):
    def embeddings(rows, *args):
        with db.connection() as connection:
            connection.execute("DELETE FROM corpus WHERE id='c'")
        return [[1, 0], [0, 1]]
    monkeypatch.setattr(local_ranking, "candidate_embeddings", embeddings)
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert [row["id"] for row in docs] == ["b"] and trace["stale_count"] == 1


def test_exact_duplicate_sources_do_not_consume_final_slots(corpus, monkeypatch):
    with db.connection() as connection:
        connection.execute("UPDATE corpus SET body='头痛的另一段资料' WHERE id='a'")
    monkeypatch.setattr(local_ranking, "score_logits", lambda query, texts, *args: [2.0] * len(texts))
    monkeypatch.setattr(local_ranking, "candidate_embeddings", lambda rows, *args: [[1, 0] for _ in rows])
    docs, trace = asyncio.run(classics.retrieve_classics(corpus))
    assert trace["mmr"]["duplicate_count"] == 1 and len(docs) == 3


def test_sigmoid_is_stable_and_applied_once():
    assert local_ranking.sigmoid(0) == .5
    assert local_ranking.sigmoid(1000) == 1 and local_ranking.sigmoid(-1000) == 0
    assert local_ranking.sigmoid(math.log(4)) == .8


def test_model_load_uses_identity_and_offline_cache(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    config = local_ranking.settings()
    class Identity:
        pass
    class Array:
        def tolist(self):
            return [math.log(4)]
    class Encoder:
        num_labels = 1
        def __init__(self, name, **kwargs):
            assert isinstance(kwargs["activation_fn"], Identity)
            assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 1 and not kwargs["apply_softmax"]
            return Array()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None, nn=SimpleNamespace(Identity=Identity)))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Encoder, SentenceTransformer=Encoder))
    from dataclasses import replace
    config = replace(config, cache=str(tmp_path))
    local_ranking._load.cache_clear()
    try:
        values = local_ranking.score_logits("头痛", ["头痛原文"], config, threading.Event())
        assert values == [math.log(4)] and local_ranking.sigmoid(values[0]) == .8
    finally:
        local_ranking._load.cache_clear()


def test_mmr_large_finite_vectors_do_not_collapse_cosine():
    rows = [{"id": item, "retrieval": {"rerank_score": .9, "rrf_rank": rank}}
            for rank, item in enumerate(["a", "b"], 1)]
    selected = local_ranking.mmr_select(rows, [[1e308, 1e308], [1e308, 1e308]], 2, .7)
    assert selected[1]["retrieval"]["max_selected_similarity"] == pytest.approx(1)


def test_worker_timeout_signals_remaining_batches_to_stop():
    stopped = threading.Event()
    def work(stop):
        stop.wait(1)
        if stop.is_set():
            stopped.set()
        local_ranking.check_stop(stop)
    async def run():
        with pytest.raises(TimeoutError):
            await local_ranking.bounded_worker(work, timeout=.01)
        await asyncio.to_thread(stopped.wait, 1)
        assert stopped.is_set()
    asyncio.run(run())


def test_namespace_changes_with_embedding_model(monkeypatch, tmp_path):
    import chromadb
    names = []
    class Client:
        def get_or_create_collection(self, name, metadata):
            names.append(name)
            assert metadata == {"hnsw:space": "cosine"}
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: Client())
    monkeypatch.setattr(retrieval, "DATA_DIR", tmp_path)
    monkeypatch.setenv("QIWEN_EMBEDDING_MODEL", "first")
    retrieval._collection()
    monkeypatch.setenv("QIWEN_EMBEDDING_MODEL", "second")
    retrieval._collection()
    assert names[0] != names[1] and all(name.startswith("classics_") for name in names)


def test_catalog_evidence_matches_bundled_source():
    from app.config import ROOT
    import json
    sources = {}
    for path in (ROOT / "backend/resources/classics_segments").glob("*.json"):
        for row in json.loads(path.read_text(encoding="utf-8")):
            title = row["metadata"]["book"]
            body = row["content"].strip()
            sources[hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()] = (title, body)
    assert len(terms.CATALOG) >= 50
    expansions = [e for entry in terms.CATALOG.values() for e in entry["expansions"]]
    assert expansions
    for e in expansions:
        title, body = sources[e["evidence"]["id"]]
        assert e["evidence"]["book"] == title
        assert e["term"] in e["evidence"]["quote"] and e["evidence"]["quote"] in body


def test_rewrites_keep_original_and_exclude_denied_past_other():
    facts = symptoms.extract("我头痛三天不明显，没有咳嗽，母亲耳鸣")
    state = {"symptoms": ["头痛"], "symptom_facts": facts, "safety_state": {"subject": "self"}}
    for attempt in range(3):
        plan = terms.search_plan(state, attempt)
        assert "头痛" in plan["keyword_query"] and "头痛" in plan["semantic_query"]
        assert "咳嗽" not in plan["semantic_query"] and "耳鸣" not in plan["keyword_query"]
        assert "三天" in plan["semantic_query"] and "不明显" in plan["semantic_query"]
    assert terms.search_plan(state, 2)["expansions"]
    assert [f["symptom"] for f in facts] == ["头痛", "咳嗽", "耳鸣"]
