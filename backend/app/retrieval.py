from __future__ import annotations

import hashlib
import math
import re
import warnings
import os
import json
from collections import Counter
from pathlib import Path

import httpx

from .config import API_BASE, API_KEY, DATA_DIR, RAG_MODE
from .db import all_rows, connection
from data_pipeline.ingest import chunk_text
from .corpus_seed import store_chunks
from .retrieval_budget import timed_stage
from .retrieval_terms import normalize_search, content_priority


def chunks(text: str, title: str = "导入文本"):
    """Compatibility body iterator over the shared structural pipeline."""
    yield from (chunk.content for chunk in chunk_text(text, title))


def ingest_file(path: Path, title: str | None = None, *, book: str | None = None) -> int:
    raw = path.read_text(encoding="utf-8-sig")
    title = title or path.stem
    prepared = chunk_text(raw, title, book=book)
    with connection() as db:
        count = store_chunks(db, title, prepared)
    if RAG_MODE == "hybrid":
        try:
            index_vectors(all_rows("SELECT id,title,chapter,body,metadata FROM corpus WHERE title=?", (title,)))
        except Exception as exc:
            warnings.warn(f"向量索引未完成，关键词检索仍可用：{type(exc).__name__}")
    return count


def _tokens(text: str) -> list[str]:
    text = normalize_search(text)
    chinese = [char for char in text if "\u3400" <= char <= "\u9fff"]
    bigrams = [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    latin = re.findall(r"[a-z0-9]+", text)
    return chinese + bigrams + latin


def bm25_scores(query: str, rows: list[dict], tokenizer=_tokens) -> list[tuple[float, str]]:
    query_terms = set(tokenizer(query))
    if not query_terms or not rows:
        return []
    counts = [Counter(tokenizer(row["body"] + " " + row.get("chapter", ""))) for row in rows]
    lengths = [sum(item.values()) for item in counts]
    average = max(sum(lengths) / len(lengths), 1)
    document_frequency = Counter(term for counter in counts for term in counter)
    ranked = []
    for row, counter, length in zip(rows, counts, lengths):
        score = 0.0
        for term in query_terms:
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(rows) - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        if score:
            ranked.append((score, row["id"]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _bm25(query: str, rows: list[dict]) -> list[str]:
    return [item_id for _, item_id in bm25_scores(query, rows)]


def _embed(texts: list[str]) -> list[list[float]]:
    from .blocking import check_cancelled, remaining_timeout
    check_cancelled()
    model = os.getenv("QIWEN_EMBEDDING_MODEL", "")
    if not API_BASE or not API_KEY or not model:
        raise RuntimeError("向量模型配置不完整")
    base = API_BASE.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if not base.endswith("/embeddings"):
        base += "/embeddings"
    response = httpx.post(
        base, headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "input": texts}, timeout=remaining_timeout(30),
    )
    check_cancelled()
    response.raise_for_status()
    data = response.json()["data"]
    if sorted(item["index"] for item in data) != list(range(len(texts))):
        raise ValueError("Embedding response indices do not match inputs")
    return validate_vectors([item["embedding"] for item in sorted(data, key=lambda item: item["index"])], len(texts))


def validate_vectors(vectors, count):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding count mismatch")
    dimensions = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Empty embedding")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError("Invalid embedding shape or values")
        if sum(v * v for v in vector) <= 0:
            raise ValueError("Zero embedding")
    return vectors


def _collection():
    import chromadb
    # Separate embedding models/providers and distance definitions. Old indexes
    # remain recoverable but cannot be silently queried with another model.
    key = hashlib.sha256((API_BASE.rstrip("/").removesuffix("/chat/completions") + "|" + os.getenv("QIWEN_EMBEDDING_MODEL", "") + "|cosine-v1").encode()).hexdigest()[:16]
    return chromadb.PersistentClient(path=str(DATA_DIR / "chroma")).get_or_create_collection(
        "classics_" + key, metadata={"hnsw:space": "cosine"})


def index_vectors(rows: list[dict], *, batch_size: int = 32, progress=None) -> int:
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("Index batch size must be between 1 and 128")
    if not rows:
        return 0
    collection = _collection()
    indexed = 0
    existing = set()
    for start in range(0, len(rows), 128):
        existing.update(collection.get(ids=[row["id"] for row in rows[start:start + 128]])["ids"])
    retained = [row for row in rows if row["id"] in existing]
    for start in range(0, len(retained), 128):
        batch = retained[start:start + 128]
        collection.update(ids=[row["id"] for row in batch], metadatas=[
            {"title": row["title"], "chapter": row["chapter"], "chunk_metadata": row.get("metadata", "{}")} for row in batch])
    # Batch only missing rows. Sparse gaps must not cause one remote call per gap.
    missing_rows = [row for row in rows if row["id"] not in existing]
    if progress:
        progress(0, len(missing_rows))
    for start in range(0, len(missing_rows), batch_size):
        missing = missing_rows[start:start + batch_size]
        embeddings = validate_vectors(_embed([row["body"] for row in missing]), len(missing))
        collection.upsert(
            ids=[row["id"] for row in missing],
            embeddings=embeddings,
            documents=[row["body"] for row in missing],
            metadatas=[{"title": row["title"], "chapter": row["chapter"], "chunk_metadata": row.get("metadata", "{}")}
                       for row in missing],
        )
        indexed += len(missing)
        if progress:
            progress(indexed, len(missing_rows))
    return indexed


def retrieve_candidates(plan: dict, recall_k=30, candidate_limit=30) -> tuple[list[dict], dict]:
    """Retrieve independently, retain channel ranks, fuse IDs before reranking."""
    from .blocking import check_cancelled
    check_cancelled()
    with timed_stage("corpus_read"):
        rows = all_rows("SELECT id,title,chapter,body,metadata FROM corpus")
    corpus_count = len(rows)
    corpus_rows = rows
    from .evidence_sources import matching_concepts, source_lookup
    source_manifest = source_fingerprint(list(source_lookup(rows).values()))
    concepts = matching_concepts(plan)
    if concepts:
        rows = corpus_rows = concepts
    hints = plan.get("source_hints", [])
    matching_sources = [row for row in rows if any(hint in row["title"] or hint in row["chapter"] for hint in hints)]
    if hints:
        rows = matching_sources
    lookup = {row["id"]: row for row in rows}
    queries = plan.get("recall_queries") or [{"kind": "original", "keyword_query": plan["keyword_query"],
                                            "semantic_query": plan["semantic_query"]}]
    # Bound fan-out and deduplicate identical variants; one batched embedding call.
    queries = list({(q["keyword_query"], q["semantic_query"]): q for q in queries[:4]}.values())
    keyword_orders, vector_orders, channel_details = [], [], {}
    raw_scores = {}
    with timed_stage("bm25"):
        # Keep corpus-wide IDF stable when a named book/chapter narrows sources.
        for query_index, query in enumerate(queries):
            keyword_scores = [(score, item_id) for score, item_id in bm25_scores(query["keyword_query"], corpus_rows)
                              if item_id in lookup][:recall_k]
            order = {item_id: rank for rank, (_, item_id) in enumerate(keyword_scores, 1)}
            keyword_orders.append(order)
            for score, item_id in keyword_scores:
                raw_scores[item_id] = max(raw_scores.get(item_id, 0), score)
                channel_details.setdefault(item_id, []).append({"channel": "bm25", "query_index": query_index,
                                                                "kind": query["kind"], "rank": order[item_id]})
    check_cancelled()
    keyword_rank = {}
    for order in keyword_orders:
        for item_id, rank in order.items():
            keyword_rank[item_id] = min(keyword_rank.get(item_id, rank), rank)
    keyword_ids = list(keyword_rank)
    vector_ids: list[str] = []
    vector_rank = {}
    similarities = {}
    indexed_count = 0
    vector_status, warning = "disabled", ""
    if concepts:
        vector_status = "not_applicable_concept_source"
    if RAG_MODE == "hybrid" and rows and not concepts:
        try:
            with timed_stage("index_open"):
                collection = _collection()
                indexed_count = collection.count()
            if indexed_count == 0:
                vector_status = "empty_index"
                warning = "向量索引为空，已退化为关键词检索；请先运行古籍向量建索引。"
            else:
                with timed_stage("query_embedding"):
                    vectors = validate_vectors(_embed([q["semantic_query"] for q in queries]), len(queries))
                check_cancelled()
                with timed_stage("vector_query"):
                    result = collection.query(query_embeddings=vectors, n_results=min(recall_k, indexed_count), include=["distances"])
                if len(result["ids"]) != len(queries) or len(result["distances"]) != len(queries):
                    raise ValueError("Invalid vector ranking")
                for query_index, (ids, distances) in enumerate(zip(result["ids"], result["distances"])):
                    if len(ids) != len(distances) or len(ids) != len(set(ids)):
                        raise ValueError("Invalid vector ranking")
                    order = {}
                    for rank, (item_id, distance) in enumerate(zip(ids, distances), 1):
                        if (isinstance(distance, bool) or not isinstance(distance, (int, float)) or
                                not math.isfinite(distance) or not -1e-5 <= distance <= 2.00001):
                            raise ValueError("Invalid vector distance")
                        if item_id in lookup:
                            similarity = max(-1.0, min(1.0, 1.0 - distance))
                            similarities[item_id] = max(similarities.get(item_id, -1), similarity)
                            vector_rank[item_id] = min(vector_rank.get(item_id, rank), rank)
                            order[item_id] = rank
                            channel_details.setdefault(item_id, []).append({"channel": "vector", "query_index": query_index,
                                                                            "kind": queries[query_index]["kind"], "rank": rank})
                    vector_orders.append(order)
                vector_ids = list(vector_rank)
                vector_status = "success" if vector_ids else "no_current_matches"
                if not vector_ids:
                    warning = "向量结果没有有效的当前古籍片段，已退化为关键词检索。"
                elif indexed_count < corpus_count:
                    warning = "古籍向量索引尚未覆盖全部片段；请继续建索引，未覆盖部分仍参加 BM25。"
        except Exception as exc:
            vector_ids, vector_rank, similarities, vector_orders, vector_status = [], {}, {}, [], "fallback"
            channel_details = {key: [hit for hit in hits if hit["channel"] != "vector"] for key, hits in channel_details.items()}
            warning = f"向量检索暂不可用，已退化为关键词检索：{type(exc).__name__}"
            vector_error = "timeout" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else "dependency_error"
    check_cancelled()
    scores: dict[str, float] = {}
    for order in [*keyword_orders, *vector_orders]:
        for item_id, rank in order.items():
            scores[item_id] = scores.get(item_id, 0) + 1 / (60 + rank) / len(queries)
    fusion_order = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    fusion_ranks = {item_id: rank for rank, item_id in enumerate(fusion_order, 1)}
    best = sorted(fusion_order, key=lambda item_id: (*(-v for v in content_priority(lookup[item_id], plan)),
                                                  fusion_ranks[item_id]))
    candidates = [{**lookup[item_id], "retrieval": {
        "bm25_rank": keyword_rank.get(item_id), "bm25_score": raw_scores.get(item_id),
        "vector_rank": vector_rank.get(item_id), "vector_similarity": similarities.get(item_id),
        "rrf_rank": fusion_ranks[item_id], "rrf_score": scores[item_id], "pool_rank": rank,
        "content_priority": content_priority(lookup[item_id], plan),
        "channels": [channel for channel, order in (("bm25", keyword_rank), ("vector", vector_rank)) if item_id in order],
        "source_metadata": json.loads(lookup[item_id].get("metadata", "{}")),
        "channel_hits": channel_details.get(item_id, []),
    }} for rank, item_id in enumerate(best[:candidate_limit], 1) if item_id in lookup]
    return candidates, {"requested_mode": RAG_MODE, "active_mode": "hybrid" if vector_status == "success" else "bm25",
                        "vector_status": vector_status, "warning": warning,
                        "dependency_status": locals().get("vector_error", "index_empty" if vector_status == "empty_index" else
                                                                 "index_incomplete" if 0 < indexed_count < corpus_count else "ok"),
                        "keyword_count": len(keyword_ids), "vector_count": len(vector_ids),
                        "candidate_count": len(candidates), "rrf_k": 60,
                        "corpus_count": corpus_count + len(concepts), "source_scope_count": len(rows), "indexed_count": indexed_count,
                        "evidence_kind": plan.get("evidence_kind", "general_classics"),
                        "recall_query_count": len(queries),
                        "source_manifest_sha256": source_manifest,
                        "source_scope_sha256": source_fingerprint(rows),
                        "recall_audit": [{"id": item_id, "rrf_rank": fusion_ranks[item_id], "pool_rank": rank,
                                          "content_priority": content_priority(lookup[item_id], plan), "in_rerank_pool": rank <= candidate_limit,
                                          "channel_hits": channel_details.get(item_id, [])}
                                         for rank, item_id in enumerate(best, 1)]}


def source_fingerprint(rows):
    values = [{key: row[key] for key in ("id", "title", "chapter", "body")} for row in sorted(rows, key=lambda row: row["id"])]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def retrieve_with_status(query: str, limit: int = 4) -> tuple[list[dict], str]:
    """Synchronous candidate interface for old CLI clients; graph uses classics."""
    docs, info = retrieve_candidates({"keyword_query": query, "semantic_query": query}, candidate_limit=limit)
    return docs, info["warning"]


def retrieve(query: str, limit: int = 4) -> list[dict]:
    return retrieve_with_status(query, limit)[0]
