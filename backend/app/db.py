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
        permissions_existed = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_permissions'").fetchone()
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
          intent_type TEXT, symptoms TEXT, notes TEXT, trace TEXT, plan TEXT,
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_thread_idx ON messages(thread_id, created_at);
        CREATE TABLE IF NOT EXISTS chat_runs (
          id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          status TEXT NOT NULL, telemetry TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_running_per_thread ON chat_runs(thread_id) WHERE status IN ('running','cancelling');
        CREATE TABLE IF NOT EXISTS candidates (
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES chat_runs(id) ON DELETE CASCADE,
          status TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', plan TEXT, review TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_events (
          run_id TEXT NOT NULL REFERENCES chat_runs(id) ON DELETE CASCADE,
          sequence INTEGER NOT NULL, candidate_id TEXT, type TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(run_id,sequence)
        );
        CREATE TABLE IF NOT EXISTS dosage_evidence (
          id TEXT PRIMARY KEY, herb_id TEXT NOT NULL, herb_name TEXT NOT NULL,
          min_dose REAL, max_dose REAL, unit TEXT NOT NULL, raw_dosage TEXT NOT NULL,
          preparation TEXT NOT NULL, source TEXT NOT NULL, source_locator TEXT NOT NULL,
          data_version TEXT NOT NULL, verification_status TEXT NOT NULL DEFAULT 'unverified',
          population TEXT NOT NULL DEFAULT 'unknown', reviewed_by TEXT, reviewed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS user_permissions (
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, scope TEXT NOT NULL,
          PRIMARY KEY(user_id,scope)
        );
        CREATE TABLE IF NOT EXISTS profile_access_audit (
          id TEXT PRIMARY KEY, caller_id TEXT, patient_id TEXT, transport TEXT NOT NULL,
          result TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          original_name TEXT NOT NULL, storage_name TEXT NOT NULL,
          media_type TEXT NOT NULL, extracted_text TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS corpus (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, chapter TEXT NOT NULL,
          body TEXT NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
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
        CREATE TABLE IF NOT EXISTS profile_memories (
          thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          summary TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS profile_memories_user_idx
          ON profile_memories(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS memory_subject_scope (
          thread_id TEXT PRIMARY KEY REFERENCES profile_memories(thread_id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          subject TEXT NOT NULL, summary_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_embeddings (
          thread_id TEXT NOT NULL REFERENCES profile_memories(thread_id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          model_key TEXT NOT NULL, content_hash TEXT NOT NULL, embedding TEXT NOT NULL,
          PRIMARY KEY(thread_id,model_key)
        );
        CREATE TABLE IF NOT EXISTS memory_controls (
          thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          enabled INTEGER NOT NULL DEFAULT 1, revision INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_terms (
          thread_id TEXT NOT NULL REFERENCES profile_memories(thread_id) ON DELETE CASCADE,
          term TEXT NOT NULL, PRIMARY KEY(thread_id,term)
        );
        CREATE INDEX IF NOT EXISTS memory_terms_lookup ON memory_terms(term,thread_id);
        CREATE TABLE IF NOT EXISTS consultation_episodes (
          id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          subject TEXT NOT NULL, version INTEGER NOT NULL, summary TEXT NOT NULL,
          state TEXT NOT NULL, last_run_id TEXT NOT NULL, started_at TEXT, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS episodes_thread ON consultation_episodes(thread_id,updated_at DESC);
        CREATE TABLE IF NOT EXISTS knowledge_topics (
          thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          state TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS thread_safety_state (
          thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          version INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS patient_safety_facts (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          category TEXT NOT NULL, entity TEXT NOT NULL, fact TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS safety_facts_user_idx ON patient_safety_facts(user_id,status);
        CREATE TABLE IF NOT EXISTS profile_suggestions (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
          run_id TEXT NOT NULL REFERENCES chat_runs(id) ON DELETE CASCADE,
          dedup_key TEXT NOT NULL, fact TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id,dedup_key)
        );
        CREATE TABLE IF NOT EXISTS safety_fact_events (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          fact_id TEXT, operation TEXT NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        """)
        if "metadata" not in {row[1] for row in db.execute("PRAGMA table_info(corpus)")}:
            db.execute("ALTER TABLE corpus ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
        if "safety_revision" not in {row[1] for row in db.execute("PRAGMA table_info(profiles)")}:
            db.execute("ALTER TABLE profiles ADD COLUMN safety_revision INTEGER NOT NULL DEFAULT 0")
        if "source_message_id" not in {row[1] for row in db.execute("PRAGMA table_info(chat_runs)")}:
            db.execute("ALTER TABLE chat_runs ADD COLUMN source_message_id TEXT")
        if "source_files" not in {row[1] for row in db.execute("PRAGMA table_info(chat_runs)")}:
            db.execute("ALTER TABLE chat_runs ADD COLUMN source_files TEXT NOT NULL DEFAULT '[]'")
        if "memory_revision" not in {row[1] for row in db.execute("PRAGMA table_info(chat_runs)")}:
            db.execute("ALTER TABLE chat_runs ADD COLUMN memory_revision INTEGER NOT NULL DEFAULT 0")
        if "trace" not in {row[1] for row in db.execute("PRAGMA table_info(messages)")}:
            db.execute("ALTER TABLE messages ADD COLUMN trace TEXT")
        if "plan" not in {row[1] for row in db.execute("PRAGMA table_info(messages)")}:
            db.execute("ALTER TABLE messages ADD COLUMN plan TEXT")
        if "prescription_review" not in {row[1] for row in db.execute("PRAGMA table_info(messages)")}:
            db.execute("ALTER TABLE messages ADD COLUMN prescription_review TEXT")
        if "candidate_id" not in {row[1] for row in db.execute("PRAGMA table_info(messages)")}:
            db.execute("ALTER TABLE messages ADD COLUMN candidate_id TEXT")
        if not permissions_existed:
            db.execute("INSERT OR IGNORE INTO user_permissions SELECT id,'profile:read' FROM users")
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
        from .knowledge_seed import seed_structured_dosage
        seed_structured_dosage(db)
        from .corpus_seed import seed_bundled_corpus
        seed_bundled_corpus(db)
        # Legacy memory tables are preserved for explicit deletion only; no index backfill.



def one(sql: str, params: tuple = ()):
    with connection() as db:
        row = db.execute(sql, params).fetchone()
        return dict(row) if row else None


def all_rows(sql: str, params: tuple = ()):
    with connection() as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()]
