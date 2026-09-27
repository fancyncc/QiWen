"""Transactional safety state and user-confirmed portrait proposals."""
from __future__ import annotations

import copy
import json
import re
from uuid import uuid4

from fastapi import HTTPException
from .db import connection, now
from .safety_context import empty_state, extract_facts, merge_facts, candidate_key

FIELD = {"allergy": "allergies", "constitution": "constitution", "history": "history"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False)


def active_facts(db, user_id):
    return [json.loads(row["fact"]) for row in db.execute(
        "SELECT fact FROM patient_safety_facts WHERE user_id=? AND status='active' ORDER BY created_at,id", (user_id,))]


def effective_profile(db, user_id):
    row = db.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    profile = dict(row) if row else {"constitution": "", "allergies": "", "history": "", "safety_revision": 0}
    profile["_safety_revision"] = profile.get("safety_revision", 0)
    facts = active_facts(db, user_id)
    for category, field in FIELD.items():
        positives = [f["entity"] for f in facts if f["category"] == category and f["assertion"] == "present"]
        if category == "constitution" and positives:
            profile[field] = positives[-1] + "（用户自述）"
        elif positives:
            base = profile.get(field, "")
            if base.strip() in {"无", "没有", "无过敏", "无过敏史"}:
                base = ""
            profile[field] = "；".join(dict.fromkeys([v for v in [base, *positives] if v]))
    return profile


def load(user_id, thread_id):
    with connection() as db:
        row = db.execute("SELECT * FROM thread_safety_state WHERE thread_id=? AND user_id=?", (thread_id, user_id)).fetchone()
        state = json.loads(row["state"]) if row else empty_state()
        state["version"] = row["version"] if row else 0
        profile = effective_profile(db, user_id)
        facts = active_facts(db, user_id)
        # Time-sensitive facts from another consultation must be revalidated.
        for fact in facts:
            if fact["category"] in {"pregnancy_lactation", "medication"} and fact["source"].get("thread_id") != thread_id:
                fact["temporal_scope"] = "past"
        if state["subject"] == "self":
            raw = db.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
            if raw:
                allergy_text = raw["allergies"].strip()
                if allergy_text and allergy_text not in {"无", "没有", "无过敏", "无过敏史", "未填写"}:
                    for item in re.split(r"[、，,；;]", allergy_text):
                        if item.strip():
                            facts += extract_facts("对" + item.strip() + "过敏", subject="self", kind="manual_profile")
                historical = extract_facts(raw["history"], subject="self", kind="manual_profile")
                for fact in historical:
                    fact["temporal_scope"] = "past"
                facts += historical
        state["facts"] = [f for f in state["facts"] if f["source"].get("kind") not in {"manual_profile", "confirmed_profile"}]
        for fact in facts:
            fact["source"]["kind"] = "confirmed_profile" if fact.get("confirmation") == "confirmed" else fact["source"]["kind"]
        state["facts"] = merge_facts(facts, state["facts"])
        return state, profile


def commit_state(db, user_id, thread_id, state, expected):
    row = db.execute("SELECT version FROM thread_safety_state WHERE thread_id=? AND user_id=?", (thread_id, user_id)).fetchone()
    actual = row["version"] if row else 0
    if actual != expected:
        raise RuntimeError("安全状态变化，本轮需要重新预审")
    state = copy.deepcopy(state)
    state["version"] = actual + 1
    db.execute("""INSERT INTO thread_safety_state(thread_id,user_id,version,state,updated_at) VALUES(?,?,?,?,?)
        ON CONFLICT(thread_id) DO UPDATE SET version=excluded.version,state=excluded.state,updated_at=excluded.updated_at""",
               (thread_id, user_id, actual + 1, dumps(state), now()))


def conflict(db, user_id, fact):
    field = FIELD.get(fact["category"])
    raw = dict(db.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone())
    existing = [f for f in active_facts(db, user_id) if f["category"] == fact["category"]]
    positives = [f["entity"] for f in existing if f["assertion"] == "present"]
    opposite = any(f["entity"] == fact["entity"] and f["assertion"] != fact["assertion"] for f in existing)
    value = raw.get(field, "") if field else ""
    if field and fact["assertion"] == "absent" and fact["entity"] in value:
        opposite = True
    if fact["category"] == "allergy":
        if fact["assertion"] == "absent":
            opposite |= bool(positives or value.strip() not in {"", "无", "没有", "无过敏", "无过敏史"}) if fact["entity"] == "过敏史" else fact["entity"] in value
        else:
            opposite |= value.strip() in {"无", "没有", "无过敏", "无过敏史"}
            opposite |= any(f["entity"] == "过敏史" and f["assertion"] == "absent" for f in existing)
    if fact["category"] == "constitution":
        opposite |= bool(value and fact["entity"] not in value) or any(f["entity"] != fact["entity"] for f in existing)
    return bool(opposite), value or "；".join(positives)


def add_suggestions(db, user_id, thread_id, run_id, candidates):
    profile = effective_profile(db, user_id)
    existing = active_facts(db, user_id)
    for fact in candidates:
        key = candidate_key(fact)
        same = [f for f in existing if all(f.get(k) == fact.get(k) for k in ("category", "entity", "assertion"))]
        field = FIELD.get(fact["category"])
        meaningful_detail = bool(re.search(r"疹|痒|红肿|休克|呼吸|反应", fact.get("details", "")))
        if same and (not meaningful_detail or any(candidate_key(f) == key for f in same)):
            continue
        if field and fact["assertion"] == "present" and fact["entity"] in profile.get(field, "") and not meaningful_detail:
            continue
        db.execute("""INSERT INTO profile_suggestions
            (id,user_id,thread_id,run_id,dedup_key,fact,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',?,?)
            ON CONFLICT(user_id,dedup_key) DO UPDATE SET
            thread_id=excluded.thread_id,run_id=excluded.run_id,fact=excluded.fact,updated_at=excluded.updated_at
            WHERE profile_suggestions.status='pending' AND profile_suggestions.thread_id<>excluded.thread_id""",
                   (str(uuid4()), user_id, thread_id, run_id, key, dumps(fact), now(), now()))


def list_suggestions(user_id, thread_id):
    with connection() as db:
        if not db.execute("SELECT 1 FROM threads WHERE id=? AND user_id=?", (thread_id, user_id)).fetchone():
            raise HTTPException(404, "会话不存在")
        revision = db.execute("SELECT safety_revision FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
        items = []
        for row in db.execute("SELECT * FROM profile_suggestions WHERE user_id=? AND thread_id=? AND status='pending' ORDER BY created_at,id", (user_id, thread_id)):
            fact = json.loads(row["fact"])
            is_conflict, previous = conflict(db, user_id, fact)
            items.append({"id": row["id"], "fact": fact, "conflict": is_conflict, "previous": previous})
        return {"revision": revision, "items": items}


def audit(db, user_id, fact_id, operation, revision):
    db.execute("INSERT INTO safety_fact_events(id,user_id,fact_id,operation,revision,created_at) VALUES(?,?,?,?,?,?)",
               (str(uuid4()), user_id, fact_id, operation, revision, now()))


def resolve(user_id, revision, decisions):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        ids = [item.id for item in decisions]
        if len(ids) != len(set(ids)):
            raise HTTPException(422, "不能重复处理同一建议")
        rows = []
        for item in decisions:
            row = db.execute("SELECT * FROM profile_suggestions WHERE id=? AND user_id=?", (item.id, user_id)).fetchone()
            if not row:
                raise HTTPException(404, "建议不存在")
            if row["status"] != "pending" and row["status"] != item.action:
                raise HTTPException(409, "该建议已处理，请刷新")
            rows.append((item, row))
        current = db.execute("SELECT safety_revision FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
        if all(row["status"] == item.action for item, row in rows):
            return {"revision": current, "status": "unchanged"}
        if current != revision:
            raise HTTPException(409, "画像已变化，请刷新后重新确认")
        next_revision = current + 1
        for item, row in rows:
            if row["status"] != "pending":
                continue
            fact = json.loads(row["fact"])
            if item.action == "accept":
                if not item.confirm_self:
                    raise HTTPException(422, "保存前必须确认资料属于本人")
                if item.entity is not None:
                    fact["entity"] = item.entity.strip()
                    if not fact["entity"]:
                        raise HTTPException(422, "资料内容不能为空")
                if item.details is not None:
                    fact["details"] = item.details.strip()
                is_conflict, _ = conflict(db, user_id, fact)
                if is_conflict and not item.replace:
                    raise HTTPException(409, "存在冲突，请明确确认更正旧资料")
                fact.update(subject="self", confirmation="confirmed")
                fact["confirmed_at"] = now()
                if item.entity is not None or item.details is not None:
                    fact["user_edited"] = True
                fact_id = str(uuid4())
                if item.replace or fact["category"] == "constitution":
                    for old in db.execute("SELECT * FROM patient_safety_facts WHERE user_id=? AND category=? AND status='active'", (user_id, fact["category"])).fetchall():
                        previous = json.loads(old["fact"])
                        if fact["category"] == "constitution" or fact["entity"] == "过敏史" or previous["entity"] == fact["entity"]:
                            db.execute("UPDATE patient_safety_facts SET status='superseded',updated_at=? WHERE id=?", (now(), old["id"]))
                            audit(db, user_id, old["id"], "superseded", next_revision)
                    if fact["category"] == "constitution":
                        db.execute("UPDATE profiles SET constitution='' WHERE user_id=?", (user_id,))
                    if fact["category"] == "history" and fact["assertion"] == "absent":
                        raw = db.execute("SELECT history FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
                        db.execute("UPDATE profiles SET history=? WHERE user_id=?", (raw.replace(fact["entity"], ""), user_id))
                    if fact["category"] == "allergy" and fact["assertion"] == "absent":
                        raw = db.execute("SELECT allergies FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
                        updated = "" if fact["entity"] == "过敏史" else raw.replace(fact["entity"], "")
                        db.execute("UPDATE profiles SET allergies=? WHERE user_id=?", (updated, user_id))
                db.execute("INSERT INTO patient_safety_facts(id,user_id,category,entity,fact,status,created_at,updated_at) VALUES(?,?,?,?,?,'active',?,?)",
                           (fact_id, user_id, fact["category"], fact["entity"], dumps(fact), now(), now()))
                audit(db, user_id, fact_id, "confirmed", next_revision)
                # Explicit confirmation can bind ambiguous facts for the next turn.
                saved = db.execute("SELECT * FROM thread_safety_state WHERE thread_id=? AND user_id=?", (row["thread_id"], user_id)).fetchone()
                if saved:
                    state = json.loads(saved["state"])
                    if item.replace:
                        state["facts"] = [old for old in state["facts"] if not (
                            old["subject"] == "self" and old["category"] == fact["category"]
                            and (fact["category"] == "constitution" or fact["entity"] == "过敏史" or old["entity"] == fact["entity"])
                            and old["id"] != json.loads(row["fact"])["id"])]
                        state["answers"]["conflict"] = "provided"
                        if state.get("pending") and state["pending"]["key"] == "conflict":
                            state["pending"] = None
                    for old in state["facts"]:
                        if old["id"] == json.loads(row["fact"])["id"]:
                            old.update(fact)
                            old.pop("conflict", None)
                    if state.get("pending") and state["pending"]["key"] == "subject":
                        state.update(subject="self", subject_confirmed=True, pending=None)
                    commit_state(db, user_id, row["thread_id"], state, saved["version"])
            elif item.action == "other":
                saved = db.execute("SELECT * FROM thread_safety_state WHERE thread_id=? AND user_id=?", (row["thread_id"], user_id)).fetchone()
                if saved:
                    state = json.loads(saved["state"])
                    for old in state["facts"]:
                        if old["id"] == fact["id"]:
                            old["subject"] = "other:未明确"
                    if state.get("pending") and state["pending"]["key"] == "subject":
                        state.update(subject="other:未明确", subject_confirmed=False)
                    commit_state(db, user_id, row["thread_id"], state, saved["version"])
            db.execute("UPDATE profile_suggestions SET status=?,updated_at=? WHERE id=?", (item.action, now(), item.id))
            audit(db, user_id, item.id, item.action, next_revision)
        db.execute("UPDATE profiles SET safety_revision=?,updated_at=? WHERE user_id=?", (next_revision, now(), user_id))
        return {"revision": next_revision, "status": "saved"}


def revoke(user_id, fact_id, revision):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM patient_safety_facts WHERE id=? AND user_id=?", (fact_id, user_id)).fetchone()
        if not row:
            raise HTTPException(404, "事实不存在")
        current = db.execute("SELECT safety_revision FROM profiles WHERE user_id=?", (user_id,)).fetchone()[0]
        if row["status"] != "active":
            return {"revision": current}
        if revision != current:
            raise HTTPException(409, "画像已变化，请刷新")
        db.execute("UPDATE patient_safety_facts SET status='revoked',updated_at=? WHERE id=?", (now(), fact_id))
        db.execute("UPDATE profiles SET safety_revision=safety_revision+1,updated_at=? WHERE user_id=?", (now(), user_id))
        audit(db, user_id, fact_id, "revoked", current + 1)
        return {"revision": current + 1}
