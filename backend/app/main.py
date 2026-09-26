from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from docx import Document
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .auth import check_password, current_user, hash_password, make_token
from .config import CHECKPOINT_PATH, CORS_ORIGINS, MAX_UPLOAD_BYTES, MODEL_MODE, RAG_MODE, UPLOAD_DIR
from .db import all_rows, connection, init_db, now, one
from . import runs
from .prescription import migrate_plan, legacy_plan, review_plan, fingerprint, patient_fingerprint

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    runs.recover_interrupted()
    yield
    for run_id in list(runs.active):
        await runs.cancel(run_id)


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
    run_id: str = Field(default_factory=lambda: str(uuid4()), pattern=r"^[A-Za-z0-9_-]{8,64}$")
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


@app.get("/api/knowledge/sources/{source_id}")
def dosage_source(source_id: str):
    row = one("SELECT id,herb_id,herb_name,min_dose,max_dose,unit,raw_dosage,preparation,source,source_locator,data_version,verification_status,population,reviewed_at FROM dosage_evidence WHERE id=?", (source_id,))
    if row is None:
        raise HTTPException(404, "出处不存在")
    return row


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
            db.execute("INSERT INTO user_permissions(user_id,scope) VALUES(?,'profile:read')", (user_id,))
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
async def delete_thread(thread_id: str, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    run_ids = [row["id"] for row in all_rows("SELECT id FROM chat_runs WHERE thread_id=?", (thread_id,))]
    for run_id in run_ids:
        await runs.cancel(run_id)
    with connection() as db:
        db.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    if CHECKPOINT_PATH.exists():
        with sqlite3.connect(CHECKPOINT_PATH) as db:
            for table in ("checkpoints", "writes"):
                exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if exists:
                    db.executemany(f"DELETE FROM {table} WHERE thread_id=?", [(item,) for item in [thread_id, *run_ids]])


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    rows = all_rows("SELECT id,role,content,intent_type,symptoms,notes,trace,plan,prescription_review,candidate_id,created_at FROM messages WHERE thread_id=? ORDER BY created_at,id",
                    (thread_id,))
    for item in rows:
        item["symptoms"] = json.loads(item["symptoms"] or "[]")
        item["notes"] = json.loads(item["notes"] or "[]")
        item["trace"] = json.loads(item["trace"] or "null")
        item["prescription"] = migrate_plan(json.loads(item.pop("plan"))) if item.get("plan") else None
        item.pop("plan", None)
        if not item["prescription"] and item["role"] == "assistant":
            item["prescription"] = legacy_plan(item["content"])
        item["prescription_review"] = json.loads(item["prescription_review"] or "null")
    return rows


@app.get("/api/profiles/{user_id}")
def get_profile(user_id: str, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权访问该画像")
    profile = one("SELECT * FROM profiles WHERE user_id=?", (user_id,))
    profile["memories"] = all_rows(
        "SELECT thread_id,summary,updated_at FROM profile_memories WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,),
    )
    return profile


@app.delete("/api/profiles/{user_id}/memories/{thread_id}", status_code=204)
def delete_profile_memory(user_id: str, thread_id: str, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权修改该画像")
    with connection() as db:
        db.execute("DELETE FROM profile_memories WHERE user_id=? AND thread_id=?", (user_id, thread_id))


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
    return {"id": file_id, "name": filename, "media_type": media_type, "created_at": now()}


@app.get("/api/uploads")
def list_uploads(user: dict = Depends(current_user)):
    return [{"id": item["id"], "name": item["original_name"],
             "media_type": item["media_type"], "created_at": item["created_at"]}
            for item in all_rows(
                "SELECT id,original_name,media_type,created_at FROM uploads WHERE user_id=? ORDER BY created_at DESC",
                (user["id"],),
            )]


@app.delete("/api/uploads/{file_id}", status_code=204)
def delete_upload(file_id: str, user: dict = Depends(current_user)):
    item = one("SELECT storage_name FROM uploads WHERE id=? AND user_id=?", (file_id, user["id"]))
    if not item:
        raise HTTPException(404, "文件不存在或不属于当前用户")
    path = (UPLOAD_DIR / item["storage_name"]).resolve()
    if path.parent != UPLOAD_DIR.resolve():
        raise HTTPException(500, "文件路径异常")
    with connection() as db:
        db.execute("DELETE FROM uploads WHERE id=? AND user_id=?", (file_id, user["id"]))
    path.unlink(missing_ok=True)


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
        try:
            db.execute("INSERT INTO chat_runs(id,thread_id,user_id,status,created_at,updated_at) VALUES(?,?,?,'running',?,?)",
                       (body.run_id, thread["id"], user["id"], now(), now()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "该会话已有运行中的问答，或 run_id 已被使用") from None
        db.execute(
            "UPDATE threads SET title=? WHERE id=? AND title='新会话' "
            "AND NOT EXISTS (SELECT 1 FROM messages WHERE thread_id=?)",
            (body.message.strip()[:28], thread["id"], thread["id"]),
        )
        db.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
                   (str(uuid4()), thread["id"], "user", body.message, now()))
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now(), thread["id"]))
    return thread, profile, "\n".join(file_texts)[:20000]


def _save_answer(run: runs.Run, result: dict, trace: dict):
    thread_id = run.thread_id

    candidate = result.get("treatment_plan", {})
    safe_plan = None
    review = result.get("prescription_review", {})
    if result.get("intent_type") == "diagnose" and not result.get("safety_errors"):
        checked, latest_review = review_plan(candidate, result.get("profile", {}))
        stored = one("SELECT status,review FROM candidates WHERE id=? AND run_id=?", (run.candidate_id, run.id))
        stored_review = json.loads(stored["review"] or "{}") if stored else {}
        manual = one("SELECT constitution,allergies,history FROM profiles WHERE user_id=?", (run.user_id,)) or {}
        if (not latest_review["approved"] or fingerprint(candidate) != result.get("reviewed_hash")
            or result.get("manual_profile_hash") != patient_fingerprint(manual)
            or not stored or stored["status"] != "approved"
            or stored_review.get("plan_hash") != fingerprint(candidate)
            or stored_review.get("patient_hash") != patient_fingerprint(result.get("profile", {}))
            or checked != candidate):
            raise RuntimeError("候选或画像变化，必须重新生成并审查")
        safe_plan = checked
        review = latest_review
    answer_id = str(uuid4())
    answer = {"id": answer_id, "role": "assistant", "content": result["response"],
              "intent_type": result.get("intent_type"), "symptoms": result.get("symptoms", []),
              "notes": result.get("consultation_notes", []), "trace": trace, "prescription": safe_plan,
              "prescription_review": review, "candidate_id": run.candidate_id}
    final = run.final_event(result, answer, trace)
    with connection() as db:
        changed = db.execute("UPDATE chat_runs SET status='completed',telemetry=?,updated_at=? WHERE id=? AND status='running'",
                             (json.dumps(trace, ensure_ascii=False), now(), run.id)).rowcount
        if not changed or run.context.cancelled:
            raise asyncio.CancelledError
        if result.get("context_summary"):
            db.execute("""INSERT INTO profile_memories(thread_id,user_id,summary,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET summary=excluded.summary,updated_at=excluded.updated_at
                WHERE profile_memories.user_id=excluded.user_id""",
                       (thread_id, run.user_id, result["context_summary"], now()))
        db.execute("INSERT INTO messages(id,thread_id,role,content,intent_type,symptoms,notes,trace,plan,prescription_review,candidate_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (answer_id, thread_id, "assistant", result["response"], result.get("intent_type"),
                    json.dumps(result.get("symptoms", []), ensure_ascii=False),
                    json.dumps(result.get("consultation_notes", []), ensure_ascii=False),
                    json.dumps(trace, ensure_ascii=False),
                    json.dumps(safe_plan, ensure_ascii=False) if safe_plan else None, json.dumps(review, ensure_ascii=False), run.candidate_id, now()))
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now(), thread_id))
        db.execute("UPDATE candidates SET status='final',content=?,plan=?,review=?,updated_at=? WHERE id=? AND run_id=?",
                   (result["response"], json.dumps(safe_plan, ensure_ascii=False), json.dumps(review, ensure_ascii=False), now(), run.candidate_id, run.id))
        db.execute("INSERT INTO run_events(run_id,sequence,candidate_id,type,payload) VALUES(?,?,?,'answer.final',?)",
                   (run.id, final["sequence"], run.candidate_id, json.dumps(final["payload"], ensure_ascii=False)))
    return final


def _sse(item: dict) -> str:
    return "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"



def owned_run(run_id: str, user_id: str):
    row = one("SELECT * FROM chat_runs WHERE id=? AND user_id=?", (run_id, user_id))
    if not row:
        raise HTTPException(404, "运行不存在")
    return row


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, user: dict = Depends(current_user)):
    row = owned_run(run_id, user["id"])
    active = runs.active.get(run_id)
    telemetry = active.telemetry() if active else json.loads(row["telemetry"])
    return {"run_id": run_id, "thread_id": row["thread_id"], "status": row["status"], "telemetry": telemetry}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, user: dict = Depends(current_user)):
    owned_run(run_id, user["id"])
    await runs.cancel(run_id)
    return get_run(run_id, user)



@app.get("/api/runs/{run_id}/candidates")
def get_candidates(run_id: str, user: dict = Depends(current_user)):
    owned_run(run_id, user["id"])
    rows = all_rows("SELECT * FROM candidates WHERE run_id=? ORDER BY created_at", (run_id,))
    for row in rows:
        row["plan"] = json.loads(row["plan"] or "null")
        row["review"] = json.loads(row["review"] or "null")
    return rows


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, user: dict = Depends(current_user)):
    owned_run(run_id, user["id"])
    return [{**row, "payload": json.loads(row["payload"])}
            for row in all_rows("SELECT run_id,candidate_id,sequence,type,payload FROM run_events WHERE run_id=? ORDER BY sequence", (run_id,))]


async def watch_disconnect(request: Request, run: runs.Run):
    while run.terminal is None:
        if await request.is_disconnected():
            await runs.cancel(run.id)
            return
        await asyncio.sleep(0.1)


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request, stream: bool = Query(False), user: dict = Depends(current_user)):
    thread, profile, file_text = _prepare_chat(body, user)
    run = runs.Run(body.run_id, thread["id"], user["id"], thread["mode"])
    runs.active[run.id] = run
    run.task = asyncio.create_task(runs.execute(run, body.message, profile, file_text, _save_answer))
    if not stream:
        watcher = asyncio.create_task(watch_disconnect(request, run))
        try:
            try:
                await asyncio.shield(run.task)
            except asyncio.CancelledError:
                if run.terminal is None:
                    await runs.cancel(run.id)
                    raise
            result = run.terminal or {}
            if result.get("error"):
                raise HTTPException(503, result["error"])
            return result
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


    async def events():
        try:
            while True:
                if await request.is_disconnected():
                    await runs.cancel(run.id)
                    return
                try:
                    item = await asyncio.wait_for(run.queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                yield _sse(item)
                if item["type"] in {"answer.final", "run.cancelled", "run.failed"}:
                    return
        finally:
            await runs.cancel(run.id)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
