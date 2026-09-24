from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import httpx

from .config import API_BASE, API_KEY, DATA_DIR, RAG_MODE
from .db import all_rows, connection, now


def chunks(text: str, size: int = 360, overlap: int = 40):
    text = re.sub(r"\s+", " ", text).strip()
    for start in range(0, len(text), size - overlap):
        part = text[start:start + size].strip()
        if len(part) >= 24:
            yield part


def ingest_file(path: Path, title: str | None = None) -> int:
    raw = path.read_text(encoding="utf-8-sig")
    title = title or path.stem
    count = 0
    with connection() as db:
        for index, body in enumerate(chunks(raw)):
            fingerprint = hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()
            cursor = db.execute(
                "INSERT OR IGNORE INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,?)",
                (fingerprint, title, f"片段 {index + 1}", body, now()),
            )
            count += cursor.rowcount
    if RAG_MODE == "hybrid":
        try:
            index_vectors(all_rows("SELECT id,title,chapter,body FROM corpus WHERE title=?", (title,)))
        except Exception as exc:
            warnings.warn(f"向量索引未完成，关键词检索仍可用：{type(exc).__name__}")
    return count


def _tokens(text: str) -> list[str]:
    text = text.lower()
    chinese = [char for char in text if "\u3400" <= char <= "\u9fff"]
    bigrams = [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    latin = re.findall(r"[a-z0-9]+", text)
    return chinese + bigrams + latin


def _bm25(query: str, rows: list[dict]) -> list[str]:
    query_terms = set(_tokens(query))
    if not query_terms or not rows:
        return []
    counts = [Counter(_tokens(row["body"] + " " + row["title"])) for row in rows]
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
    return [item_id for _, item_id in ranked]


def _embed(texts: list[str]) -> list[list[float]]:
    model = __import__("os").getenv("QIWEN_EMBEDDING_MODEL", "")
    if not API_BASE or not API_KEY or not model:
        raise RuntimeError("向量模型配置不完整")
    base = API_BASE.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if not base.endswith("/embeddings"):
        base += "/embeddings"
    response = httpx.post(
        base, headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "input": texts}, timeout=60,
    )
    response.raise_for_status()
    return [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]


def _collection():
    import chromadb
    return chromadb.PersistentClient(path=str(DATA_DIR / "chroma")).get_or_create_collection("classics")


def index_vectors(rows: list[dict]) -> None:
    if not rows:
        return
    collection = _collection()
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        embeddings = _embed([row["body"] for row in batch])
        collection.upsert(
            ids=[row["id"] for row in batch],
            embeddings=embeddings,
            documents=[row["body"] for row in batch],
            metadatas=[{"title": row["title"], "chapter": row["chapter"]} for row in batch],
        )


def retrieve_with_status(query: str, limit: int = 4) -> tuple[list[dict], str]:
    rows = all_rows("SELECT id,title,chapter,body FROM corpus")
    lookup = {row["id"]: row for row in rows}
    keyword_ids = _bm25(query, rows)
    vector_ids: list[str] = []
    warning = ""
    if RAG_MODE == "hybrid" and rows:
        try:
            vector = _embed([query])[0]
            vector_ids = _collection().query(query_embeddings=[vector], n_results=min(15, len(rows)))["ids"][0]
        except Exception as exc:
            warning = f"向量检索暂不可用，已退化为关键词检索：{type(exc).__name__}"
    scores: dict[str, float] = {}
    for order in (keyword_ids, vector_ids):
        for rank, item_id in enumerate(order, start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (60 + rank)
    best = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    return [lookup[item_id] for item_id in best[:limit] if item_id in lookup], warning


def retrieve(query: str, limit: int = 4) -> list[dict]:
    return retrieve_with_status(query, limit)[0]
