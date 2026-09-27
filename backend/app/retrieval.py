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
    text = text.lower()
    chinese = [char for char in text if "\u3400" <= char <= "\u9fff"]
    bigrams = [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    latin = re.findall(r"[a-z0-9]+", text)
    return chinese + bigrams + latin


def bm25_scores(query: str, rows: list[dict], tokenizer=_tokens) -> list[tuple[float, str]]:
    query_terms = set(tokenizer(query))
    if not query_terms or not rows:
        return []
    counts = [Counter(tokenizer(row["body"] + " " + row["title"])) for row in rows]
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
        json={"model": model, "input": texts}, timeout=30,
    )
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


def index_vectors(rows: list[dict]) -> int:
    if not rows:
        return 0
    collection = _collection()
    indexed = 0
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        existing = set(collection.get(ids=[row["id"] for row in batch])["ids"])
        missing = [row for row in batch if row["id"] not in existing]
        embeddings = validate_vectors(_embed([row["body"] for row in missing]), len(missing)) if missing else []
        # Existing vectors retain their IDs; refresh metadata without re-embedding.
        metadata_rows = [{"title": row["title"], "chapter": row["chapter"],
                          "chunk_metadata": row.get("metadata", "{}")} for row in batch]
        if existing:
            retained = [(row, meta) for row, meta in zip(batch, metadata_rows) if row["id"] in existing]
            collection.update(ids=[row["id"] for row, _ in retained], metadatas=[meta for _, meta in retained])
        if not missing:
            continue
        collection.upsert(
            ids=[row["id"] for row in missing],
            embeddings=embeddings,
            documents=[row["body"] for row in missing],
            metadatas=[{"title": row["title"], "chapter": row["chapter"], "chunk_metadata": row.get("metadata", "{}")}
                       for row in missing],
        )
        indexed += len(missing)
    return indexed


def retrieve_candidates(plan: dict, recall_k=30, candidate_limit=30) -> tuple[list[dict], dict]:
    """Retrieve independently, retain channel ranks, fuse IDs before reranking."""
    rows = all_rows("SELECT id,title,chapter,body,metadata FROM corpus")
    lookup = {row["id"]: row for row in rows}
    keyword_scores = bm25_scores(plan["keyword_query"], rows)[:recall_k]
    keyword_ids = [item_id for _, item_id in keyword_scores]
    vector_ids: list[str] = []
    vector_rank = {}
    similarities = {}
    indexed_count = 0
    vector_status, warning = "disabled", ""
    if RAG_MODE == "hybrid" and rows:
        try:
            vector = validate_vectors(_embed([plan["semantic_query"]]), 1)[0]
            collection = _collection()
            indexed_count = collection.count()
            if indexed_count == 0:
                vector_status = "empty_index"
                warning = "向量索引为空，已退化为关键词检索；请先运行古籍向量建索引。"
            else:
                result = collection.query(query_embeddings=[vector], n_results=min(recall_k, indexed_count), include=["distances"])
                ids, distances = result["ids"][0], result["distances"][0]
                if len(ids) != len(distances) or len(ids) != len(set(ids)):
                    raise ValueError("Invalid vector ranking")
                for rank, (item_id, distance) in enumerate(zip(ids, distances), 1):
                    if (isinstance(distance, bool) or not isinstance(distance, (int, float)) or
                            not math.isfinite(distance) or not -1e-5 <= distance <= 2.00001):
                        raise ValueError("Invalid vector distance")
                    if item_id in lookup:
                        similarities[item_id] = max(-1.0, min(1.0, 1.0 - distance))
                        vector_ids.append(item_id)
                        vector_rank[item_id] = rank
                vector_status = "success" if vector_ids else "no_current_matches"
                if not vector_ids:
                    warning = "向量结果没有有效的当前古籍片段，已退化为关键词检索。"
                elif indexed_count < len(rows):
                    warning = "古籍向量索引尚未覆盖全部片段；请继续建索引，未覆盖部分仍参加 BM25。"
        except Exception as exc:
            vector_ids, vector_rank, similarities, vector_status = [], {}, {}, "fallback"
            warning = f"向量检索暂不可用，已退化为关键词检索：{type(exc).__name__}"
    scores: dict[str, float] = {}
    keyword_rank = {item_id: rank for rank, item_id in enumerate(keyword_ids, 1)}
    for order in (keyword_rank, vector_rank):
        for item_id, rank in order.items():
            scores[item_id] = scores.get(item_id, 0) + 1 / (60 + rank)
    best = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    raw_scores = {item_id: score for score, item_id in keyword_scores}
    candidates = [{**lookup[item_id], "retrieval": {
        "bm25_rank": keyword_rank.get(item_id), "bm25_score": raw_scores.get(item_id),
        "vector_rank": vector_rank.get(item_id), "vector_similarity": similarities.get(item_id),
        "rrf_rank": rank, "rrf_score": scores[item_id],
        "channels": [channel for channel, order in (("bm25", keyword_rank), ("vector", vector_rank)) if item_id in order],
        "source_metadata": json.loads(lookup[item_id].get("metadata", "{}")),
    }} for rank, item_id in enumerate(best[:candidate_limit], 1) if item_id in lookup]
    return candidates, {"requested_mode": RAG_MODE, "active_mode": "hybrid" if vector_status == "success" else "bm25",
                        "vector_status": vector_status, "warning": warning,
                        "keyword_count": len(keyword_ids), "vector_count": len(vector_ids),
                        "candidate_count": len(candidates), "rrf_k": 60,
                        "corpus_count": len(rows), "indexed_count": indexed_count}


def retrieve_with_status(query: str, limit: int = 4) -> tuple[list[dict], str]:
    """Synchronous candidate interface for old CLI clients; graph uses classics."""
    docs, info = retrieve_candidates({"keyword_query": query, "semantic_query": query}, candidate_limit=limit)
    return docs, info["warning"]


def retrieve(query: str, limit: int = 4) -> list[dict]:
    return retrieve_with_status(query, limit)[0]
