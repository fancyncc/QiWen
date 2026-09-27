from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("QIWEN_DATA_DIR", ROOT / ".data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "qiwen.sqlite3"
CHECKPOINT_PATH = DATA_DIR / "checkpoints.sqlite3"
MODEL_MODE = os.getenv("QIWEN_MODEL_MODE", "mock")
RAG_MODE = os.getenv("QIWEN_RAG_MODE", "hybrid")
API_BASE = os.getenv("QIWEN_MODEL_BASE_URL", "")
API_KEY = os.getenv("QIWEN_MODEL_API_KEY", "")
MODEL_NAME = os.getenv("QIWEN_MODEL_NAME", "")
if os.getenv("QIWEN_ENV") == "production" and not os.getenv("QIWEN_SECRET_KEY"):
    raise RuntimeError("Production requires QIWEN_SECRET_KEY")


def _secret() -> str:
    configured = os.getenv("QIWEN_SECRET_KEY")
    if configured:
        return configured
    file = DATA_DIR / "dev_jwt_secret"
    if file.exists():
        return file.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(48)
    file.write_text(value, encoding="utf-8")
    return value


SECRET_KEY = _secret()
CORS_ORIGINS = os.getenv("QIWEN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
