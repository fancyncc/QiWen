"""Idempotently load the six bundled classic-text segment sets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

CHUNK_DIR = Path(__file__).resolve().parents[1] / "resources" / "classics_segments"
BOOKS = (
    "伤寒论", "金匮要略", "温病条辨", "神农本草经", "脉经", "黄帝内经-素问",
)


def seed_bundled_corpus(db: sqlite3.Connection, *, chunk_dir: Path = CHUNK_DIR) -> int:
    """Preserve existing user imports and skip previously loaded passages."""
    rows: list[tuple[str, str, str, str]] = []
    for book in BOOKS:
        path = chunk_dir / f"{book}.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            title = metadata.get("book") or book
            if title != book:
                raise ValueError(f"古籍片段书名不匹配：{path.name}")
            body = chunk.get("content", "").strip()
            if not body:
                continue
            chapter = (metadata.get("chapter") or metadata.get("category")
                       or (f'条文{metadata["clause_number"]}' if metadata.get("clause_number") else "未标章节"))
            fingerprint = hashlib.sha256(f"{title}\n{body}".encode("utf-8")).hexdigest()
            rows.append((fingerprint, title, str(chapter), body))
    before = db.total_changes
    db.executemany(
        "INSERT OR IGNORE INTO corpus(id,title,chapter,body,created_at) VALUES(?,?,?,?,datetime('now'))",
        rows,
    )
    return db.total_changes - before
