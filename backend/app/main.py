from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import date
from uuid import uuid4
from typing import Literal

from docx import Document
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .auth import check_password, current_user, hash_password, make_token
from .config import CORS_ORIGINS, MAX_UPLOAD_BYTES, MODEL_MODE, RAG_MODE, UPLOAD_DIR
from .db import all_rows, connection, init_db, now, one
from . import runs
from . import safety_store
from . import episodes
from .prescription import migrate_plan, legacy_plan, review_plan, fingerprint, patient_fingerprint
from .documents import MAX_FILE_TEXT_CHARS

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    runs.recover_interrupted()
    if (RAG_MODE == "hybrid" and os.getenv("QIWEN_RAG_RERANK_MODE", "cross_encoder") == "cross_encoder"
            and os.getenv("QIWEN_RAG_WARMUP", "true").lower() == "true"):
        from .local_ranking import warm_models
        _app.state.retrieval_warmup = await warm_models()
    else:
        _app.state.retrieval_warmup = {"status": "not_run"}
    yield
    pending = [run.task for run in runs.active.values() if run.task is not None]
    for run_id in list(runs.active):
        await runs.cancel(run_id)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


app = FastAPI(title="岐问 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   allow_headers=["Authorization", "Content-Type"])


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class ThreadCreate(BaseModel):
    mode: str = Field(default="fixed", pattern="^(fixed|multi)$")


class ThreadUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    mode: Literal["fixed", "multi"] | None = None


class ProfilePatch(BaseModel):
    constitution: str | None = Field(default=None, max_length=500)
    allergies: str | None = Field(default=None, max_length=500)
    history: str | None = Field(default=None, max_length=1000)
    revision: int | None = Field(default=None, ge=0)


class MemoryControlPatch(BaseModel):
    enabled: bool
    revision: int = Field(ge=0)


class MedicalRecordInput(BaseModel):
    record_date: date | None = None
    institution: str = Field(default="", max_length=120)
    diagnosis: str = Field(min_length=1, max_length=200)
    treatment: str = Field(default="", max_length=1000)
    outcome: str = Field(default="", max_length=500)
    revision: int = Field(ge=0)
    confirm_self: bool = False


class SuggestionDecision(BaseModel):
    id: str
    action: Literal["accept", "dismiss", "other"]
    confirm_self: bool = False
    replace: bool = False
    entity: str | None = Field(default=None, min_length=1, max_length=80)
    details: str | None = Field(default=None, max_length=500)


class ResolveSuggestions(BaseModel):
    revision: int = Field(ge=0)
    decisions: list[SuggestionDecision] = Field(min_length=1, max_length=30)


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


@app.get("/api/readiness")
async def readiness():
    """Read-only retrieval diagnostics; no provider calls or model downloads."""
    from . import classics, retrieval
    from .blocking import run_blocking
    checks = {}
    checks["startup_models"] = getattr(app.state, "retrieval_warmup", {"status": "not_run"})
    ids = [row["id"] for row in all_rows("SELECT id FROM corpus")]
    checks["corpus"] = {"count": len(ids)}
    if RAG_MODE == "hybrid":
        checks["embedding_configuration"] = {"configured": bool(retrieval.API_BASE and retrieval.API_KEY and os.getenv("QIWEN_EMBEDDING_MODEL"))}
        try:
            async with asyncio.timeout(5):
                def coverage():
                    collection = retrieval._collection()
                    return collection.count(), len(set(collection.get(ids=ids, include=[])["ids"]) & set(ids)) if ids else 0
                count, covered = await run_blocking(coverage)
            checks["vector_index"] = {"count": count, "covered_count": covered,
                                      "status": "covered" if covered == len(ids) > 0 else "empty" if not count else "incomplete"}
        except Exception as exc:
            checks["vector_index"] = {"status": "unavailable", "error_type": type(exc).__name__}
    recent = dict(classics.last_observation)
    if recent and time.time() - recent["observed_at"] <= 300:
        checks["last_retrieval"] = recent
    else:
        checks["last_retrieval"] = {"outcome": "not_probed"}
    unavailable = (checks["startup_models"]["status"] == "unavailable"
                   or not checks["corpus"]["count"] or checks.get("vector_index", {}).get("status", "covered") != "covered"
                   or not checks.get("embedding_configuration", {}).get("configured", True)
                   or checks["last_retrieval"]["outcome"] not in {"success", "no_match", "not_probed"}
                   or checks["last_retrieval"].get("mmr_status") == "failed"
                   or RAG_MODE == "hybrid" and recent and recent.get("vector_status") not in {"success", "disabled"})
    unprobed = (checks["last_retrieval"]["outcome"] == "not_probed"
                or RAG_MODE == "hybrid" and checks["last_retrieval"].get("vector_status") != "success")
    return {"status": "degraded" if unavailable else "not_probed" if unprobed else "ready",
            "scope": "retrieval_prerequisites_and_recent_execution_not_answer_quality", "checks": checks}


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
    return public_thread(new_thread(user["id"], "新对话", body.mode))


@app.put("/api/threads/{thread_id}")
def rename_thread(thread_id: str, body: ThreadUpdate, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    if body.title is None and body.mode is None:
        raise HTTPException(422, "请提供会话名称或问诊模式")
    if body.title is not None and not body.title.strip():
        raise HTTPException(422, "会话名称不能为空")
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if body.mode is not None:
            if db.execute("SELECT 1 FROM chat_runs WHERE thread_id=? AND status IN ('running','cancelling')", (thread_id,)).fetchone():
                raise HTTPException(409, "请等待当前回答结束后再切换问诊模式")
            db.execute("UPDATE threads SET mode=?,updated_at=? WHERE id=?", (body.mode, now(), thread_id))
        if body.title is not None:
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
    # A synchronous busy wait here blocks the aiosqlite writer's event loop,
    # preventing that writer from releasing the very lock deletion needs.
    from .checkpoint_retention import delete_thread_checkpoints
    await asyncio.to_thread(delete_thread_checkpoints, thread_id, run_ids)


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
    with connection() as db:
        profile = safety_store.effective_profile(db, user_id)
        profile["facts"] = [{**json.loads(row["fact"]), "id": row["id"]} for row in db.execute(
            "SELECT id,fact FROM patient_safety_facts WHERE user_id=? AND status='active' ORDER BY created_at", (user_id,))]
    return profile


def change_medical_record(user_id, user, record_id, body=None, *, delete=False):
    if user_id != user["id"]:
        raise HTTPException(403, "无权修改该画像")
    if not delete and (not body.confirm_self or not body.diagnosis.strip()):
        raise HTTPException(422, "请填写诊断或就诊原因，并确认记录属于本人")
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if record_id and not db.execute("SELECT 1 FROM medical_records WHERE id=? AND user_id=?", (record_id, user_id)).fetchone():
            raise HTTPException(404, "诊疗记录不存在")
        current = db.execute("SELECT safety_revision FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
        revision = body if delete else body.revision
        if revision != current:
            raise HTTPException(409, "画像已变化，请刷新后重新保存")
        if delete:
            db.execute("DELETE FROM medical_records WHERE id=? AND user_id=?", (record_id, user_id))
        else:
            values = body.model_dump(mode="json", exclude={"revision", "confirm_self"})
            values = {key: value.strip() if isinstance(value, str) else value for key, value in values.items()}
            if record_id:
                db.execute("UPDATE medical_records SET record_date=?,institution=?,diagnosis=?,treatment=?,outcome=?,updated_at=? WHERE id=? AND user_id=?",
                           (*values.values(), now(), record_id, user_id))
            else:
                db.execute("INSERT INTO medical_records(id,user_id,record_date,institution,diagnosis,treatment,outcome,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                           (str(uuid4()), user_id, *values.values(), now(), now()))
        db.execute("UPDATE profiles SET safety_revision=safety_revision+1,updated_at=? WHERE user_id=?", (now(), user_id))
        from .memory import invalidate_user
        invalidate_user(db, user_id)
    return get_profile(user_id, user)


@app.post("/api/profiles/{user_id}/medical-records", status_code=201)
def create_medical_record(user_id: str, body: MedicalRecordInput, user: dict = Depends(current_user)):
    return change_medical_record(user_id, user, None, body)


@app.put("/api/profiles/{user_id}/medical-records/{record_id}")
def update_medical_record(user_id: str, record_id: str, body: MedicalRecordInput, user: dict = Depends(current_user)):
    return change_medical_record(user_id, user, record_id, body)


@app.delete("/api/profiles/{user_id}/medical-records/{record_id}")
def delete_medical_record(user_id: str, record_id: str, revision: int = Query(ge=0), user: dict = Depends(current_user)):
    return change_medical_record(user_id, user, record_id, revision, delete=True)


@app.delete("/api/profiles/{user_id}/memories/{thread_id}", status_code=204)
def delete_profile_memory(user_id: str, thread_id: str, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权修改该画像")
    owned_thread(thread_id, user_id)
    from .memory import bump_control
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        bump_control(db, user_id, thread_id)


@app.patch("/api/threads/{thread_id}/memory")
def patch_memory_control(thread_id: str, body: MemoryControlPatch, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    from .memory import bump_control, control
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if control(user["id"], thread_id)["revision"] != body.revision:
            raise HTTPException(409, "记忆设置已变化，请刷新")
        return bump_control(db, user["id"], thread_id, body.enabled)


@app.delete("/api/threads/{thread_id}/memory", status_code=204)
def delete_thread_memory(thread_id: str, user: dict = Depends(current_user)):
    owned_thread(thread_id, user["id"])
    from .memory import bump_control
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        bump_control(db, user["id"], thread_id)


@app.patch("/api/profiles/{user_id}")
def patch_profile(user_id: str, body: ProfilePatch, user: dict = Depends(current_user)):
    if user_id != user["id"]:
        raise HTTPException(403, "无权修改该画像")
    changes = body.model_dump(exclude_unset=True)
    revision = changes.pop("revision", None)
    if any(value is None for value in changes.values()):
        raise HTTPException(422, "画像字段不能为 null")
    if changes:
        assignments = ",".join(f"{field}=?" for field in changes)
        with connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT safety_revision FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
            if revision is not None and revision != current:
                raise HTTPException(409, "画像已变化，请刷新")
            for category, field in safety_store.FIELD.items():
                if field in changes:
                    rows = db.execute("SELECT id FROM patient_safety_facts WHERE user_id=? AND category=? AND status='active'",
                                      (user_id, category)).fetchall()
                    for row in rows:
                        db.execute("UPDATE patient_safety_facts SET status='superseded',updated_at=? WHERE id=?", (now(), row["id"]))
                        safety_store.audit(db, user_id, row["id"], "manual_update", current + 1)
            db.execute(f"UPDATE profiles SET {assignments},updated_at=? WHERE user_id=?",
                       (*changes.values(), now(), user_id))
            db.execute("UPDATE profiles SET safety_revision=safety_revision+1 WHERE user_id=?", (user_id,))
            from .memory import invalidate_user
            invalidate_user(db, user_id)
    return get_profile(user_id, user)


@app.get("/api/profile-suggestions")
def profile_suggestions(thread_id: str, user: dict = Depends(current_user)):
    return safety_store.list_suggestions(user["id"], thread_id)


@app.post("/api/profile-suggestions/resolve")
def resolve_profile_suggestions(body: ResolveSuggestions, user: dict = Depends(current_user)):
    return safety_store.resolve(user["id"], body.revision, body.decisions)


@app.delete("/api/profile-facts/{fact_id}")
def revoke_profile_fact(fact_id: str, revision: int = Query(ge=0), user: dict = Depends(current_user)):
    return safety_store.revoke(user["id"], fact_id, revision)


def extract_file(filename: str, data: bytes) -> tuple[str, str]:
    extension = Path(filename).suffix.lower()
    def collect(parts):
        chunks, size = [], 0
        for part in parts:
            size += len(part) + (1 if chunks else 0)
            if size > MAX_FILE_TEXT_CHARS:
                raise HTTPException(413, "附件文本超过 20000 字符，请拆分文件后上传；本次未保存，也未截断处理。")
            chunks.append(part)
        return "\n".join(chunks)

    try:
        if extension == ".txt":
            return "text/plain", collect([data.decode("utf-8-sig")])
        if extension == ".pdf" and data.startswith(b"%PDF"):
            pages = PdfReader(io.BytesIO(data)).pages
            return "application/pdf", collect(page.extract_text() or "" for page in pages)
        if extension == ".docx" and data.startswith(b"PK"):
            doc = Document(io.BytesIO(data))
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", collect(
                block.text if hasattr(block, "text") else "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                for block in doc.iter_inner_content())
    except HTTPException:
        raise
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
    media_type, content = await asyncio.to_thread(extract_file, filename, data)
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
    profile = {}  # Login authorizes access; the workflow resolves the patient first.
    with connection() as db:
        try:
            db.execute("INSERT INTO chat_runs(id,thread_id,user_id,status,created_at,updated_at) VALUES(?,?,?,'running',?,?)",
                       (body.run_id, thread["id"], user["id"], now(), now()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "该会话已有运行中的问答，或 run_id 已被使用") from None
        db.execute(
            "UPDATE threads SET title=? WHERE id=? AND title IN ('新会话','新对话') "
            "AND NOT EXISTS (SELECT 1 FROM messages WHERE thread_id=?)",
            (body.message.strip()[:28], thread["id"], thread["id"]),
        )
        source_message_id = str(uuid4())
        db.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
                   (source_message_id, thread["id"], "user", body.message, now()))
        db.execute("UPDATE chat_runs SET source_message_id=?,source_files=? WHERE id=?",
                   (source_message_id, json.dumps(body.files), body.run_id))
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now(), thread["id"]))
    return thread, profile, "\n".join(file_texts)


def _save_answer(run: runs.Run, result: dict, trace: dict):
    thread_id = run.thread_id
    safe = result.get("safety_state", {})
    personal = result.get("identity_mode") == "clinical" and safe.get("subject") == "self" and safe.get("subject_confirmed") and not result.get("identity_urgent")
    resolved = result.get("identity_mode") == "clinical" and safe.get("subject_confirmed") and safe.get("subject") not in {"unknown", "other:未明确"}
    expected_scope = f'{thread_id}:{safe.get("subject")}:{safe.get("subject_epoch", 0)}' if resolved else ""
    if result.get("identity_scope", "") != expected_scope or result.get("intent_type") == "diagnose" and not resolved:
        raise RuntimeError("问诊对象不一致，需要重新确认")

    candidate = result.get("treatment_plan", {})
    safe_plan = None
    review = result.get("prescription_review", {})
    if result.get("intent_type") == "diagnose" and not result.get("safety_errors"):
        checked, latest_review = review_plan(candidate, result.get("profile", {}))
        stored = one("SELECT status,review FROM candidates WHERE id=? AND run_id=?", (run.candidate_id, run.id))
        stored_review = json.loads(stored["review"] or "{}") if stored else {}
        with connection() as db:
            manual = safety_store.effective_profile(db, run.user_id) if personal else {}
        if (not latest_review["approved"] or fingerprint(candidate) != result.get("reviewed_hash")
            or personal and result.get("manual_profile_hash") != patient_fingerprint(manual)
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
        db.execute("BEGIN IMMEDIATE")
        if personal and result.get("manual_profile_hash") != patient_fingerprint(safety_store.effective_profile(db, run.user_id)):
            raise RuntimeError("画像已变化，本轮需要重新预审")
        changed = db.execute("UPDATE chat_runs SET status='completed',telemetry=?,updated_at=? WHERE id=? AND status='running'",
                             (json.dumps(trace, ensure_ascii=False), now(), run.id)).rowcount
        if not changed or run.context.cancelled:
            raise asyncio.CancelledError
        safe = result.get("safety_state", {})
        if safe and result.get("identity_mode") == "clinical":
            safe["symptom_facts"] = result.get("symptom_facts", [])
            safe.setdefault("clinical", {})[safe["subject"]] = {
                key: result.get(key) for key in ("pending_context", "pending_question", "symptom_pending")}
            safety_store.commit_state(db, run.user_id, thread_id, safe, result.get("safety_state_version", 0))
        if personal:
            safety_store.add_suggestions(db, run.user_id, thread_id, run.id, result.get("profile_suggestions", []))
        episode = result.get("episode_summary", {})
        if result.get("episode_id") and episode and not result.get("episode_pending"):
            db.execute("INSERT INTO consultation_episodes(id,thread_id,user_id,subject,version,summary,state,last_run_id,started_at,updated_at) VALUES(?,?,?,?,1,?,?,?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET version=consultation_episodes.version+1,summary=excluded.summary,state=excluded.state,last_run_id=excluded.last_run_id,updated_at=excluded.updated_at",
                       (result["episode_id"], thread_id, run.user_id, safe["subject"], result.get("context_summary") or episodes.summary(result)[0],
                        json.dumps(episode, ensure_ascii=False), run.id, episode.get("started_at"), now()))
        # Clinical episode summaries stay in this thread; never publish to profiles.
        if result.get("identity_mode") == "knowledge" and result.get("knowledge_state") and not result.get("knowledge_ambiguous"):
            db.execute("INSERT INTO knowledge_topics VALUES(?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET state=excluded.state,updated_at=excluded.updated_at",
                       (thread_id, run.user_id, json.dumps(result["knowledge_state"], ensure_ascii=False), now()))
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
