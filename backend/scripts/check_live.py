"""Smoke test real chat, embedding, vector search, and answer generation."""

from __future__ import annotations

from app.config import API_BASE, API_KEY, MODEL_MODE, MODEL_NAME, RAG_MODE
from app.db import init_db
from app.model import complete
from app.retrieval import _collection, _embed, retrieve_with_status


def main() -> int:
    if MODEL_MODE != "openai_compatible" or RAG_MODE != "hybrid":
        raise SystemExit("Set QIWEN_MODEL_MODE=openai_compatible and QIWEN_RAG_MODE=hybrid.")
    if not API_BASE or not API_KEY or not MODEL_NAME:
        raise SystemExit("Configure QIWEN_MODEL_BASE_URL, QIWEN_MODEL_API_KEY and QIWEN_MODEL_NAME.")
    init_db()
    reply = complete("只回答一个简短的中文确认词，不提供医疗建议。", "请确认模型接口可用。")
    if not reply.strip():
        raise RuntimeError("The chat model returned an empty response.")
    vector = _embed(["桂枝汤"])
    if not vector or not vector[0]:
        raise RuntimeError("The embedding model returned no vector.")
    collection = _collection()
    indexed = collection.count()
    if indexed == 0:
        raise RuntimeError("Chroma is empty. Run python -m scripts.index_vectors first.")
    vector_hits = collection.query(query_embeddings=[vector[0]], n_results=min(4, indexed))["ids"][0]
    if not vector_hits:
        raise RuntimeError("Chroma returned no vector matches.")
    docs, warning = retrieve_with_status("桂枝汤", limit=4)
    if warning or not docs:
        raise RuntimeError(warning or "Retrieval returned no classical source.")
    print(f"Chat OK; embedding dimensions: {len(vector[0])}; indexed segments: {indexed}; vector hits: {len(vector_hits)}; fused results: {len(docs)}")
    print(f"Top source: {docs[0]['title']} / {docs[0]['chapter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
