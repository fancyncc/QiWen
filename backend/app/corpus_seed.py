"""Idempotently load the six bundled classic-text segment sets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from data_pipeline.base import Chunk
from .db import now

CHUNK_DIR = Path(__file__).resolve().parents[1] / "resources" / "classics_segments"
BOOKS = (
    "伤寒论", "金匮要略", "温病条辨", "神农本草经", "脉经", "黄帝内经-素问",
)


def store_chunks(db: sqlite3.Connection, title: str, chunks: list[Chunk]) -> int:
    """Persist source metadata; dedup new content across the current corpus."""
    existing = {row[0]: (row[1], row[2]) for row in db.execute("SELECT id,body,metadata FROM corpus")}
    seen = {hashlib.md5(body.encode()).hexdigest() for body, _ in existing.values()}
    added = 0
    for chunk in chunks:
        body = chunk.content.strip()
        if not body:
            continue
        metadata = {**chunk.metadata, "source_chunk_id": chunk.id}
        chapter = (metadata.get("chapter") or metadata.get("category") or
                   (f'条文{metadata["clause_number"]}' if metadata.get("clause_number") else "未标章节"))
        identifier = hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()
        encoded = json.dumps(metadata, ensure_ascii=False)
        if identifier in existing:
            db.execute("UPDATE corpus SET metadata=? WHERE id=? AND metadata='{}'", (encoded, identifier))
            continue
        content_id = hashlib.md5(body.encode()).hexdigest()
        if content_id in seen:
            continue
        cursor = db.execute("INSERT OR IGNORE INTO corpus(id,title,chapter,body,created_at,metadata) VALUES(?,?,?,?,?,?)",
                            (identifier, title, str(chapter), body, now(), encoded))
        added += cursor.rowcount
        seen.add(content_id)
    return added


def seed_bundled_corpus(db: sqlite3.Connection, *, chunk_dir: Path = CHUNK_DIR) -> int:
    """Preserve existing user imports and skip previously loaded passages."""
    added = 0
    for book in BOOKS:
        path = chunk_dir / f"{book}.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        prepared = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            title = metadata.get("book") or book
            if title != book:
                raise ValueError(f"古籍片段书名不匹配：{path.name}")
            prepared.append(Chunk(chunk["id"], chunk.get("content", ""), metadata))
        added += store_chunks(db, book, prepared)
    return added
