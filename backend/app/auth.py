from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from uuid import uuid4

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import SECRET_KEY
from .db import one

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$")
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def make_token(user_id: str) -> str:
    header = _b64(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64(json.dumps({"sub": user_id, "exp": int(time.time()) + 86400, "jti": str(uuid4())}, separators=(",", ":")).encode())
    body = f"{header}.{payload}"
    signature = _b64(hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{signature}"


def token_identity(token: str) -> tuple[dict, dict]:
    try:
        header, payload, signature = token.split(".")
        expected = _b64(hmac.new(SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        decoded_header = json.loads(_unb64(header))
        if not isinstance(decoded_header, dict) or decoded_header.get("alg") != "HS256":
            raise ValueError("algorithm")
        claims = json.loads(_unb64(payload))
        if not isinstance(claims, dict) or not isinstance(claims.get("exp"), (int, float)):
            raise ValueError("claims")
        if claims.get("exp", 0) <= time.time() or not isinstance(claims.get("sub"), str):
            raise ValueError("expired")
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(401, "登录已失效") from None
    user = one("SELECT id, username FROM users WHERE id=?", (claims["sub"],))
    if not user:
        raise HTTPException(401, "用户不存在")
    return user, claims


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "请先登录")
    return token_identity(credentials.credentials)[0]
