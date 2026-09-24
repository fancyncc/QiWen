from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profiles (
          user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
          constitution TEXT NOT NULL DEFAULT '',
          allergies TEXT NOT NULL DEFAULT '',
          history TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS threads (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          title TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'fixed',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS threads_user_idx ON threads(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
          role TEXT NOT NULL, content TEXT NOT NULL,
          intent_type TEXT, symptoms TEXT, notes TEXT,
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_thread_idx ON messages(thread_id, created_at);
        CREATE TABLE IF NOT EXISTS uploads (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          original_name TEXT NOT NULL, storage_name TEXT NOT NULL,
          media_type TEXT NOT NULL, extracted_text TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS corpus (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, chapter TEXT NOT NULL,
          body TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS herbs (
          name TEXT PRIMARY KEY, properties TEXT NOT NULL DEFAULT '',
          effects TEXT NOT NULL DEFAULT '', source TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS formulas (
          name TEXT PRIMARY KEY, herbs TEXT NOT NULL, indication TEXT NOT NULL,
          source TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS contraindications (
          kind TEXT NOT NULL, left_name TEXT NOT NULL, right_name TEXT NOT NULL,
          source TEXT NOT NULL, PRIMARY KEY(kind,left_name,right_name)
        );
        """)
        from .safety import GROUPS, KNOWN, PAIRS
        source = "DB4403/T 142-2021 附录 C"
        db.executemany(
            "INSERT OR IGNORE INTO herbs(name,source) VALUES(?,?)",
            [(name, "内置演示种子") for name in KNOWN],
        )
        db.executemany(
            "INSERT OR IGNORE INTO formulas(name,herbs,indication,source) VALUES(?,?,?,?)",
            [
                ("酸枣仁汤", '["酸枣仁","茯苓","知母","川芎","甘草"]', "睡眠相关知识示例", "内置演示种子"),
                ("桂枝汤", '["桂枝","芍药","甘草","生姜","大枣"]', "外感相关知识示例", "内置演示种子"),
            ],
        )
        contraindications = []
        for label, (left, right) in GROUPS.items():
            contraindications.extend((label, a, b, source) for a in left for b in right)
        contraindications.extend(("十九畏", a, b, source) for a, b in PAIRS)
        db.executemany(
            "INSERT OR IGNORE INTO contraindications(kind,left_name,right_name,source) VALUES(?,?,?,?)",
            contraindications,
        )
        from .knowledge_seed import seed_bundled_knowledge
        seed_bundled_knowledge(db)
        from .corpus_seed import seed_bundled_corpus
        seed_bundled_corpus(db)


def one(sql: str, params: tuple = ()):
    with connection() as db:
        row = db.execute(sql, params).fetchone()
        return dict(row) if row else None


def all_rows(sql: str, params: tuple = ()):
    with connection() as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()]
