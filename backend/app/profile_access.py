from __future__ import annotations

from uuid import uuid4

from .db import connection, one, now


def audit(caller_id: str | None, patient_id: str | None, transport: str, result: str):
    with connection() as db:
        db.execute("INSERT INTO profile_access_audit(id,caller_id,patient_id,transport,result,created_at) VALUES(?,?,?,?,?,?)",
                   (str(uuid4()), caller_id, patient_id, transport, result, now()))


def read_profile(caller_id: str | None, patient_id: str | None, *, transport: str) -> dict:
    result = "unauthenticated"
    try:
        if not caller_id or not one("SELECT id FROM users WHERE id=?", (caller_id,)):
            raise PermissionError("调用者身份未绑定或无效")
        result = "forbidden"
        if caller_id != patient_id or not one("SELECT scope FROM user_permissions WHERE user_id=? AND scope='profile:read'", (caller_id,)):
            raise PermissionError("缺少本人画像的 profile:read 权限")
        row = one("SELECT constitution,allergies,history FROM profiles WHERE user_id=?", (caller_id,))
        result = "not_found"
        if row is None:
            raise PermissionError("画像不存在")
        result = "success"
        return row
    finally:
        audit(caller_id, patient_id, transport, result)
