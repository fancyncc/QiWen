"""Bound checkpoint storage while keeping successful recovery and active runs."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from .config import CHECKPOINT_PATH
from .db import all_rows


def delete_thread_checkpoints(thread_id: str, run_ids: list[str]):
    if not CHECKPOINT_PATH.exists():
        return
    with closing(sqlite3.connect(CHECKPOINT_PATH, timeout=30)) as db, db:
        for table in ("checkpoints", "writes"):
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                db.executemany(f"DELETE FROM {table} WHERE thread_id=?", [(item,) for item in [thread_id, *run_ids]])


def prune():
    if not CHECKPOINT_PATH.exists():
        return
    keep_success = max(1, int(os.getenv("QIWEN_CHECKPOINT_SUCCESS_KEEP", "3")))
    keep_stopped = max(0, int(os.getenv("QIWEN_CHECKPOINT_STOPPED_KEEP", "2")))
    counts, removable = {}, []
    for row in all_rows("SELECT id,thread_id,status FROM chat_runs ORDER BY created_at DESC,id DESC"):
        if row["status"] in {"running", "cancelling"}:
            continue
        kind = "completed" if row["status"] == "completed" else "stopped"
        key = (row["thread_id"], kind)
        counts[key] = counts.get(key, 0) + 1
        limit = keep_success if kind == "completed" else keep_stopped
        if counts[key] > limit:
            removable.append((row["id"],))
    if removable:
        with closing(sqlite3.connect(CHECKPOINT_PATH, timeout=1)) as db, db:
            for table in ("checkpoints", "writes"):
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    db.executemany(f"DELETE FROM {table} WHERE thread_id=?", removable)
