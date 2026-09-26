from __future__ import annotations

import pytest

from app import retrieval


def test_vector_index_resume_only_embeds_missing_segments(monkeypatch):
    stored: dict[str, list[float]] = {}
    embedded: list[list[str]] = []

    class Collection:
        def get(self, ids):
            return {"ids": [item for item in ids if item in stored]}

        def upsert(self, ids, embeddings, documents, metadatas):
            assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
            stored.update(zip(ids, embeddings))

    def embed(texts):
        embedded.append(texts)
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(retrieval, "_collection", Collection)
    monkeypatch.setattr(retrieval, "_embed", embed)
    rows = [
        {"id": "a", "title": "伤寒论", "chapter": "1", "body": "桂枝汤"},
        {"id": "b", "title": "金匮要略", "chapter": "2", "body": "酸枣仁汤"},
    ]
    assert retrieval.index_vectors(rows) == 2
    assert retrieval.index_vectors(rows) == 0
    assert embedded == [["桂枝汤", "酸枣仁汤"]]
    assert set(stored) == {"a", "b"}


def test_chroma_index_and_hybrid_retrieval(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setattr(retrieval, "DATA_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "RAG_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "_embed", lambda texts: [
        [1.0, 0.0] if "桂枝" in text else [0.0, 1.0] for text in texts
    ])
    rows = [
        {"id": "a", "title": "伤寒论", "chapter": "条一", "body": "桂枝汤用于经典条文示例"},
        {"id": "b", "title": "金匮要略", "chapter": "条二", "body": "酸枣仁汤用于经典条文示例"},
    ]
    monkeypatch.setattr(retrieval, "all_rows", lambda *args: rows)
    assert retrieval.index_vectors(rows) == 2
    assert retrieval.index_vectors(rows) == 0
    assert retrieval._collection().count() == 2
    found, warning = retrieval.retrieve_with_status("桂枝汤")
    assert not warning
    assert found[0]["id"] == "a"


def test_hybrid_mode_reports_empty_vector_index(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setattr(retrieval, "DATA_DIR", tmp_path)
    monkeypatch.setattr(retrieval, "RAG_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "_embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(retrieval, "all_rows", lambda *args: [
        {"id": "a", "title": "伤寒论", "chapter": "条一", "body": "桂枝汤用于经典条文示例"},
    ])
    found, warning = retrieval.retrieve_with_status("桂枝汤")
    assert found and found[0]["id"] == "a"
    assert "向量索引为空" in warning
