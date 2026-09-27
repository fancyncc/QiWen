from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

import pytest

from data_pipeline.base import Chunk
from data_pipeline.ingest import (TCM_FILES, _normalize_chunks, _split_long_chunk,
                                  chunk_all_books, chunk_text, ingest_to_chroma, resolve_book)
from app import db, retrieval


@pytest.mark.parametrize("name,expected", [("《伤寒论》", "伤寒论"),
    ("黄帝内经·素问", "黄帝内经-素问"), ("金匮要略_完整清洗版", "金匮要略"), ("其他文本", None)])
def test_book_resolution(name, expected):
    assert resolve_book(name) == expected


def test_generic_headings_and_paragraphs_have_no_window_overlap():
    a, b = "甲" * 180 + "。", "乙" * 180 + "。"
    chunks = chunk_text(f"### 第一章\n{a}\n\n{b}", "测试文本")
    assert [c.content for c in chunks] == [a, b]
    assert all(c.metadata["chapter"] == "第一章" for c in chunks)


def test_shared_filtering_and_exact_content_dedup():
    good = "这是保留的中文正文，用于核对资料清理规则。"
    text = f"### 第一章\n{good}\nChinese Text Project\nCloudflare challenge\n\n{good}\n\n短文\n\nThis is an English navigation page."
    chunks = chunk_text(text, "测试资料")
    assert len(chunks) == 1 and chunks[0].content == good
    assert chunks[0].metadata["chapter"] == "第一章"
    toc = "\n".join(f"{i}. 目录条目" for i in range(1, 8))
    assert not chunk_text(toc, "测试目录")


def test_long_chunks_inherit_metadata_and_split_from():
    original = Chunk("TEST_1", "。".join("甲" * 100 for _ in range(10)) + "。",
                     {"book": "测试", "chapter": "第一章", "clause_number": "一"})
    chunks = _normalize_chunks([original], "测试")
    assert len(chunks) > 1 and max(len(c.content) for c in chunks) <= 500
    assert sum(c.content.count("甲") for c in chunks) == 1000
    assert all(c.metadata == {**original.metadata, "split_from": "TEST_1"} for c in chunks)
    assert all(c.id.startswith("TEST_1_") for c in chunks)


def test_unpunctuated_extreme_sentence_is_not_hard_cut_or_orphan_punctuation():
    original = Chunk("LONG", "甲" * 800, {"book": "测试"})
    chunks = _split_long_chunk(original, 500)
    assert len(chunks) == 1 and chunks[0].content == original.content + "。"


def test_all_books_dedup_identical_content_across_books(tmp_path):
    body = "这是两本资料中完全相同的正文，需要按内容去重。"
    for name in ["伤寒论", "金匮要略"]:
        (tmp_path / TCM_FILES[name]).write_text("### 测试章节\n" + body, encoding="utf-8")
    chunks = chunk_all_books(str(tmp_path))
    assert len(chunks["伤寒论"]) == 1 and chunks["金匮要略"] == []


def test_import_uses_book_structure_and_preserves_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "corpus.sqlite3")
    monkeypatch.setattr(retrieval, "RAG_MODE", "bm25")
    db.init_db()
    path = tmp_path / "input.txt"
    body = "这是专门用于核对结构切片与来源字段的新段落。"
    path.write_text("### 测试章节\n" + body, encoding="utf-8")
    assert retrieval.ingest_file(path, "测试版伤寒论", book="伤寒论") == 1
    assert retrieval.ingest_file(path, "测试版伤寒论", book="伤寒论") == 0
    # A second source with exactly the same content does not add a duplicate.
    assert retrieval.ingest_file(path, "另一本测试资料") == 0
    row = db.one("SELECT * FROM corpus WHERE title=?", ("测试版伤寒论",))
    metadata = json.loads(row["metadata"])
    assert row["body"] == body and row["chapter"] == "测试章节"
    assert metadata == {"book": "伤寒论", "chapter": "测试章节", "source_chunk_id": "SHL_0001"}
    found, trace = retrieval.retrieve_candidates({"keyword_query": body, "semantic_query": body}, candidate_limit=100)
    assert next(r for r in found if r["id"] == row["id"])["retrieval"]["source_metadata"] == metadata


def test_old_database_metadata_migration_preserves_existing_rows(tmp_path, monkeypatch):
    from app.corpus_seed import CHUNK_DIR
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "old.sqlite3")
    item = json.loads((CHUNK_DIR / "伤寒论.json").read_text(encoding="utf-8"))[0]
    body = item["content"].strip()
    identifier = hashlib.sha256(("伤寒论\n" + body).encode()).hexdigest()
    with sqlite3.connect(db.DB_PATH) as connection:
        connection.execute("CREATE TABLE corpus(id TEXT PRIMARY KEY,title TEXT,chapter TEXT,body TEXT,created_at TEXT)")
        connection.executemany("INSERT INTO corpus VALUES(?,?,?,?,?)", [
            (identifier, "伤寒论", "原有章节", body, "original"),
            ("user-owned", "用户资料", "用户章节", "用户自行导入的独立正文。", "original")])
    db.init_db()
    row = db.one("SELECT * FROM corpus WHERE id=?", (identifier,))
    assert row["chapter"] == "原有章节" and row["created_at"] == "original"
    assert json.loads(row["metadata"])["source_chunk_id"] == item["id"]
    user = db.one("SELECT * FROM corpus WHERE id='user-owned'")
    assert user["body"] == "用户自行导入的独立正文。" and user["metadata"] == "{}"
    before = db.all_rows("SELECT * FROM corpus ORDER BY id")
    db.init_db()
    assert db.all_rows("SELECT * FROM corpus ORDER BY id") == before


def test_chroma_entry_uses_qiwen_index_and_shared_chunks(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "chroma.sqlite3")
    monkeypatch.setattr(config, "API_BASE", "https://embedding.invalid/v1")
    monkeypatch.setattr(config, "API_KEY", "test-only")
    monkeypatch.setenv("QIWEN_EMBEDDING_MODEL", "test-embedding")
    captured = []
    monkeypatch.setattr(retrieval, "index_vectors", lambda rows: captured.extend(rows) or len(rows))
    chunk = Chunk("NEW_1", "这是一条专用于验证当前项目入库入口的完整正文。", {"book": "测试书", "chapter": "章甲"})
    count = asyncio.run(ingest_to_chroma({"测试书": [chunk]}))
    row = next(row for row in captured if row["title"] == "测试书")
    assert count == len(captured) and row["body"] == chunk.content
    assert json.loads(row["metadata"])["source_chunk_id"] == "NEW_1"


def test_chroma_entry_requires_embedding_configuration(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "API_BASE", "")
    with pytest.raises(RuntimeError, match="QIWEN_MODEL_BASE_URL"):
        asyncio.run(ingest_to_chroma({}))
