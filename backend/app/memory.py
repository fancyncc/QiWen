"""Legacy summary deletion/control compatibility; no clinical memory retrieval.

Old tables are retained without publishing or recalling their content.
"""
from .db import one


def control(user_id: str, thread_id: str) -> dict:
    return one("SELECT enabled,revision FROM memory_controls WHERE user_id=? AND thread_id=?", (user_id, thread_id)) or {"enabled": 1, "revision": 0}


def bump_control(db, user_id: str, thread_id: str, enabled: bool | None = None):
    saved = db.execute("SELECT enabled,revision FROM memory_controls WHERE user_id=? AND thread_id=?", (user_id, thread_id)).fetchone()
    value = int(enabled) if enabled is not None else saved["enabled"] if saved else 1
    revision = saved["revision"] + 1 if saved else 1
    from .db import now
    db.execute("INSERT INTO memory_controls VALUES(?,?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET enabled=excluded.enabled,revision=excluded.revision,updated_at=excluded.updated_at",
               (thread_id, user_id, value, revision, now()))
    db.execute("DELETE FROM profile_memories WHERE thread_id=? AND user_id=?", (thread_id, user_id))
    return {"enabled": bool(value), "revision": revision}


def invalidate_user(db, user_id: str):
    # Explicit profile correction/revocation invalidates derived summaries, not chat history.
    rows = db.execute("SELECT thread_id FROM profile_memories WHERE user_id=?", (user_id,)).fetchall()
    for row in rows:
        bump_control(db, user_id, row["thread_id"])
