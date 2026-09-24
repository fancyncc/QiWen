from __future__ import annotations

import io
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from docx import Document
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .auth import check_password, current_user, hash_password, make_token
from .config import CHECKPOINT_PATH, CORS_ORIGINS, MAX_UPLOAD_BYTES, MODEL_MODE, RAG_MODE, UPLOAD_DIR
from .db import all_rows, connection, init_db, now, one
from .workflow import run_flow

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="岐问 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   allow_headers=["Authorization", "Content-Type"])


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class ThreadCreate(BaseModel):
    mode: str = Field(default="fixed", pattern="^(fixed|multi)$")


class ThreadUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ProfilePatch(BaseModel):
    constitution: str | None = Field(default=None, max_length=500)
    allergies: str | None = Field(default=None, max_length=500)
    history: str | None = Field(default=None, max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None
    files: list[str] = Field(default_factory=list, max_length=5)
    mode: str = Field(default="fixed", pattern="^(fixed|multi)$")


def owned_thread(thread_id: str, user_id: str) -> dict:
    thread = one("SELECT * FROM threads WHERE id=? AND user_id=?", (thread_id, user_id))
    if not thread:
        raise HTTPException(404, "会话不存在")
    return thread


def public_thread(thread: dict) -> dict:
    return {key: thread[key] for key in ("id", "title", "mode", "created_at", "updated_at")}


def new_thread(user_id: str, title: str, mode: str) -> dict:
    thread_id = str(uuid4())
    stamp = now()
    with connection() as db:
        db.execute("INSERT INTO threads(id,user_id,title,mode,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                   (thread_id, user_id, title[:80], mode, stamp, stamp))
    return owned_thread(thread_id, user_id)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "rag_mode": RAG_MODE, "model_mode": MODEL_MODE}


@app.post("/api/register", status_code=201)
def register(body: Credentials):
    if one("SELECT id FROM users WHERE username=?", (body.username,)):
        raise HTTPException(409, "用户名已存在")
    user_id = str(uuid4())
    try:
        with connection() as db:
            db.execute("INSERT INTO users(id,username,password_hash,created_at) VALUES(?,?,?,?)",
                       (user_id, body.username, hash_password(body.password), now()))
            db.execute("INSERT INTO profiles(user_id,updated_at) VALUES(?,?)", (user_id, now()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "用户名已存在") from None
    return {"access_token": make_token(user_id), "token_type": "bearer",
            "user": {"id": user_id, "username": body.username}}


@app.post("/api/login")
def login(body: Credentials):
    user = one("SELECT id,username,password_hash FROM users WHERE username=?", (body.username,))
    if not user or not check_password(body.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    return {"access_token": make_token(user["id"]), "token_type": "bearer",
            "user": {"id": user["id"], "username": user["username"]}}


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return user


@app.get("/api/threads")
def list_threads(user: dict = Depends(current_user)):
    return [public_thread(item) for item in all_rows(
        "SELECT * FROM threads WHERE user_id=? ORDER BY updated_at DESC", (user["id"],)
    )]


@app.post("/api/threads", status_code=201)
def create_thread(body: ThreadCreate, user: dict = Depends(current_user)):
    return public_thread(new_thread(user["id"], "新会话", body.mode))


@app.put("/api/threads/{thread_id}")
def rename_thread(thread_id: str, body: ThreadUpdate, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    if not body.title.strip():
        raise HTTPException(422, "会话名称不能为空")
    with connection() as db:
        db.execute("UPDATE threads SET title=?,updated_at=? WHERE id=?",
                   (body.title.strip(), now(), thread_id))
    return public_thread(owned_thread(thread_id, user["id"]))


@app.delete("/api/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: str, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    with connection() as db:
        db.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    if CHECKPOINT_PATH.exists():
        with sqlite3.connect(CHECKPOINT_PATH) as db:
            for table in ("checkpoints", "writes"):
                exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if exists:
                    db.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    rows = all_rows("SELECT id,role,content,intent_type,symptoms,notes,created_at FROM messages WHERE thread_id=? ORDER BY created_at,id",
                    (thread_id,))
    for item in rows:
        item["symptoms"] = json.loads(item["symptoms"] or "[]")
        item["notes"] = json.loads(item["notes"] or "[]")
    return rows


@app.get("/api/profiles/{user_id}")
def get_profile(user_id: str, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权访问该画像")
    return one("SELECT * FROM profiles WHERE user_id=?", (user_id,))


@app.patch("/api/profiles/{user_id}")
def patch_profile(user_id: str, body: ProfilePatch, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权修改该画像")
    changes = body.model_dump(exclude_unset=True)
    if changes:
        assignments = ",".join(f"{field}=?" for field in changes)
        with connection() as db:
            db.execute(f"UPDATE profiles SET {assignments},updated_at=? WHERE user_id=?",
                       (*changes.values(), now(), user_id))
    return get_profile(user_id, user)


def extract_file(filename: str, data: bytes) -> tuple[str, str]:
    extension = Path(filename).suffix.lower()
    try:
        if extension == ".txt":
            return "text/plain", data.decode("utf-8-sig")[:20000]
        if extension == ".pdf" and data.startswith(b"%PDF"):
            pages = PdfReader(io.BytesIO(data)).pages
            return "application/pdf", "\n".join(page.extract_text() or "" for page in pages)[:20000]
        if extension == ".docx" and data.startswith(b"PK"):
            doc = Document(io.BytesIO(data))
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "\n".join(
                p.text for p in doc.paragraphs
            )[:20000]
    except Exception as exc:
        raise HTTPException(400, "文件无法解析") from exc
    raise HTTPException(400, "仅支持有效的 PDF、DOCX、UTF-8 TXT 文件")


@app.post("/api/upload", status_code=201)
async def upload(file: UploadFile = File(...), user: dict = Depends(current_user)):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(400, "文件类型不支持")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件不能超过 10 MB")
    media_type, content = extract_file(filename, data)
    file_id = str(uuid4())
    storage_name = file_id + Path(filename).suffix.lower()
    (UPLOAD_DIR / storage_name).write_bytes(data)
    with connection() as db:
        db.execute("INSERT INTO uploads(id,user_id,original_name,storage_name,media_type,extracted_text,created_at) VALUES(?,?,?,?,?,?,?)",
                   (file_id, user["id"], filename, storage_name, media_type, content, now()))
    return {"id": file_id, "name": filename, "media_type": media_type}


def _prepare_chat(body: ChatRequest, user: dict) -> tuple[dict, dict, str]:
    thread = owned_thread(body.thread_id, user["id"]) if body.thread_id else None
    file_texts = []
    for file_id in body.files:
        item = one("SELECT extracted_text,original_name FROM uploads WHERE id=? AND user_id=?",
                   (file_id, user["id"]))
        if not item:
            raise HTTPException(404, "文件不存在或不属于当前用户")
        file_texts.append(f'{item["original_name"]}: {item["extracted_text"]}')
    if thread is None:
        thread = new_thread(user["id"], body.message.strip()[:28], body.mode)
    profile = one("SELECT constitution,allergies,history FROM profiles WHERE user_id=?", (user["id"],)) or {}
    with connection() as db:
        db.execute(
            "UPDATE threads SET title=? WHERE id=? AND title='新会话' "
            "AND NOT EXISTS (SELECT 1 FROM messages WHERE thread_id=?)",
            (body.message.strip()[:28], thread["id"], thread["id"]),
        )
        db.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
                   (str(uuid4()), thread["id"], "user", body.message, now()))
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now(), thread["id"]))
    return thread, profile, "\n".join(file_texts)[:20000]


def _save_answer(thread_id: str, result: dict):
    with connection() as db:
        db.execute("INSERT INTO messages(id,thread_id,role,content,intent_type,symptoms,notes,created_at) VALUES(?,?,?,?,?,?,?,?)",
                   (str(uuid4()), thread_id, "assistant", result["response"], result.get("intent_type"),
                    json.dumps(result.get("symptoms", []), ensure_ascii=False),
                    json.dumps(result.get("consultation_notes", []), ensure_ascii=False), now()))
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now(), thread_id))


def _sse(item: dict) -> str:
    return "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"


@app.post("/api/chat")
def chat(body: ChatRequest, stream: bool = Query(False), user: dict = Depends(current_user)):
    thread, profile, file_text = _prepare_chat(body, user)
    thread_id = thread["id"]
    started = time.monotonic()
    flow = run_flow(thread_id, thread["mode"], body.message, profile, file_text)
    if not stream:
        try:
            result = None
            for item in flow:
                if item["type"] == "result":
                    result = item["data"]
            if result is None:
                raise RuntimeError("流程未返回结果")
            _save_answer(thread_id, result)
            return {"response": result["response"], "thread_id": thread_id,
                    "intent_type": result.get("intent_type"), "symptoms": result.get("symptoms", []),
                    "notes": result.get("consultation_notes", [])}
        except Exception as exc:
            raise HTTPException(503, f"问答暂不可用：{exc}") from exc

    def events():
        try:
            result = None
            for item in flow:
                if item["type"] == "result":
                    result = item["data"]
                    for index in range(0, len(result["response"]), 18):
                        yield _sse({"token": result["response"][index:index + 18]})
                else:
                    yield _sse(item)
            if result is None:
                raise RuntimeError("流程未返回结果")
            _save_answer(thread_id, result)
            yield _sse({"done": True, "thread_id": thread_id,
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                        "notes": result.get("consultation_notes", [])})
        except Exception as exc:
            yield _sse({"error": f"问答暂不可用：{exc}"})
        finally:
            flow.close()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
