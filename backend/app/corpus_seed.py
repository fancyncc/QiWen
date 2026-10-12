"""Idempotently load the six bundled classic-text segment sets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from data_pipeline.base import Chunk
from data_pipeline.cleaning import clean_source_text
from .db import now

CHUNK_DIR = Path(__file__).resolve().parents[1] / "resources" / "classics_segments"
RAW_CHUNK_DIR = CHUNK_DIR.with_name("classics_segments_raw")
LEGACY_MANIFEST = CHUNK_DIR.with_name("classics_legacy_manifest.json")
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
    if chunk_dir == CHUNK_DIR and LEGACY_MANIFEST.exists():
        # Retire only known original/intermediate bundles whose source identity,
        # chapter and body still match. User imports and edits remain recoverable.
        current_ids = set()
        for book in BOOKS:
            for row in json.loads((chunk_dir / f"{book}.json").read_text(encoding="utf-8")):
                current_ids.add(hashlib.sha256(f'{book}\n{row["content"].strip()}'.encode()).hexdigest())
        existing = {row[0]: row for row in db.execute("SELECT id,title,chapter,body FROM corpus")}
        for source in json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))["records"]:
            row = existing.get(source["id"])
            if (row and row[0] not in current_ids and row[1:3] == (source["title"], source["chapter"]) and
                    hashlib.sha256(row[3].encode()).hexdigest() == source["body_sha256"]):
                db.execute("DELETE FROM corpus WHERE id=? AND body=?", (row[0], row[3]))
    for book in BOOKS:
        path = chunk_dir / f"{book}.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        prepared = []
        retired = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            title = metadata.get("book") or book
            if title != book:
                raise ValueError(f"古籍片段书名不匹配：{path.name}")
            raw = chunk.get("content", "").strip()
            content, provenance = clean_source_text(raw, book)
            if content != raw:
                raw_id = hashlib.sha256(f"{book}\n{raw}".encode()).hexdigest()
                # Backfill provenance on legacy rows that a user re-chaptered.
                db.execute("UPDATE corpus SET metadata=? WHERE id=? AND metadata='{}'",
                    (json.dumps({**metadata, "source_chunk_id": chunk["id"]}, ensure_ascii=False), raw_id))
                source_chapter = (metadata.get("chapter") or metadata.get("category") or
                    (f'条文{metadata["clause_number"]}' if metadata.get("clause_number") else "未标章节"))
                retired.append((raw_id, book, raw, str(source_chapter)))
                metadata = {**metadata, **provenance, "raw_source_chunk_id": chunk["id"]}
            if content:
                prepared.append(Chunk(chunk["id"], content, metadata))
        # Only exact bundled source IDs/bodies are replaced; unrelated imports stay.
        # The raw JSON remains in resources for recovery and citation auditing.
        db.executemany("DELETE FROM corpus WHERE id=? AND title=? AND body=? AND chapter=?", retired)
        added += store_chunks(db, book, prepared)
    return added
