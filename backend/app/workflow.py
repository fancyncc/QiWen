from __future__ import annotations

import json
import re
import asyncio
import inspect
import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer

from .config import CHECKPOINT_PATH, MODEL_MODE
from .db import one
from .model import acomplete, acomplete_json, CandidatePlan, ReviewResult, invocation, current_node
from .preflight import assess
from .prescription import migrate_plan, names, review_plan, fingerprint, patient_fingerprint, render_plan, knowledge_context
from .classics import passage_excerpt, retrieve_classics
from .terms import CLASSIC_TERMS, search_plan as build_search_plan
from .safety import check_herbs, mentioned_herbs, normalize, contains_dosing
from . import safety_store
from .safety_context import make_context, applicable_facts, gate_context, response_for, profile_candidates, merge_facts, extract_facts, subject_of, uncovered_allergens
from . import symptoms as symptom_parser
from .symptoms import SYMPTOM_ALIASES
from .documents import model_excerpt, normalize_document
from .context import clinical_context
from . import episodes


class FlowState(TypedDict, total=False):
    episode_id: str
    episode_pending: bool
    episode_summary: dict
    knowledge_state: dict
    knowledge_question: str
    knowledge_ambiguous: bool
    knowledge_citations: list[dict]
    identity_mode: str
    identity_current: list[dict]
    identity_previous: dict
    identity_scope: str
    identity_urgent: bool
    identity_resolved_now: bool
    user_id: str
    preflight: dict
    analysis_only: bool
    manual_review_reasons: list[str]
    preliminary_analysis: bool
    analysis_questions: list[str]
    safety_state: dict
    safety_state_version: int
    profile_suggestions: list[dict]
    source_message_id: str
    run_id: str
    prescription_review: dict
    reviewed_hash: str
    reviewer_hash: str
    manual_profile_hash: str
    thread_id: str
    message: str
    file_text: str
    profile: dict
    episode_text: str
    context_summary: str
    context_chars: int
    conversation_turns: int
    symptoms: list[str]
    denied_symptoms: list[str]
    symptom_facts: list[dict]
    symptom_pending: dict | None
    intent_type: str
    pending_question: str
    pending_context: str
    docs: list[dict]
    search_query: str
    search_attempts: int
    retrieval_score: float
    search_required: bool
    retrieval_warning: str
    retrieval_trace: dict
    retrieval_history: list[dict]
    search_plan: dict
    diagnosis: str
    treatment_plan: dict
    active_plan: dict
    bianzheng_note: str
    formula_note: str
    herb_note: str
    consultation_notes: list[dict]
    safety_errors: list[str]
    safety_findings: list[str]
    reviewer_approved: bool
    reviewer_retry_count: int
    safety_retry_count: int
    response: str
    agent_stop: str
    agent_traces: dict
    agent_artifacts: dict
    herb_review: dict


URGENT = ["胸痛", "呼吸困难", "昏厥", "大出血", "意识不清", "自杀",
          "咯血", "咳血", "惊厥", "脑梗", "嘴歪", "半身不遂", "硬得像板"]
MAX_SEARCH_ATTEMPTS = 3
MAX_SAFETY_REVISIONS = 2


def prepare(state: FlowState) -> FlowState:
    # Clear all per-turn fields, including prior expert drafts.
    return {
        "docs": [], "retrieval_warning": "", "diagnosis": "", "treatment_plan": {}, "active_plan": {},
        "bianzheng_note": "", "formula_note": "", "herb_note": "",
        "consultation_notes": [], "safety_errors": [], "safety_findings": [], "response": "",
        "prescription_review": {}, "reviewed_hash": "", "reviewer_hash": "", "reviewer_approved": True, "reviewer_retry_count": 0, "safety_retry_count": 0,
        "episode_text": "", "context_summary": "",
        "episode_summary": {}, "episode_pending": False,
        "knowledge_state": {}, "knowledge_question": "", "knowledge_ambiguous": False, "knowledge_citations": [],
        "context_chars": 0,
        "conversation_turns": 0,
        "search_query": "", "search_attempts": 0, "retrieval_score": 0.0,
        "search_plan": {}, "retrieval_trace": {}, "retrieval_history": [],
        "search_required": False, "intent_type": "", "symptoms": [], "denied_symptoms": [],
        "profile_suggestions": [],
        "analysis_only": False, "manual_review_reasons": [],
        "preliminary_analysis": False, "analysis_questions": [],
        "agent_stop": "", "agent_traces": {}, "agent_artifacts": {}, "herb_review": {},
    }


def document_parse(state: FlowState) -> FlowState:
    # PDF/DOCX/TXT parsing and file ownership checks happen at upload/API time.
    # Normalize the extracted text before it enters a model context.
    return {"file_text": normalize_document(state.get("file_text", ""))}


def attachment_text(state: dict) -> str:
    run = one("SELECT source_files FROM chat_runs WHERE id=? AND user_id=?", (state.get("run_id", ""), state.get("user_id", "")))
    ids = json.loads(run["source_files"]) if run else []
    if not ids:
        return state.get("file_text", "")
    rows = [one("SELECT original_name,extracted_text FROM uploads WHERE id=? AND user_id=?", (id, state["user_id"])) for id in ids]
    return "\n".join(f"{row['original_name']}: {row['extracted_text']}" for row in rows if row)


async def profile_load(state: FlowState) -> FlowState:
    user_id, thread_id = state["user_id"], state["thread_id"]
    safety_state = state["safety_state"]
    personal = state.get("identity_mode") == "clinical" and safety_state.get("subject") == "self" and safety_state.get("subject_confirmed") and not state.get("identity_urgent")
    safety_state, profile = safety_store.load(user_id, thread_id, safety_state) if personal else (safety_state, {})
    source_message_id = state.get("source_message_id", "")
    run_source = one("SELECT source_message_id,source_files FROM chat_runs WHERE id=? AND user_id=?", (state.get("run_id", ""), user_id))
    if run_source and run_source["source_message_id"]:
        source_message_id = run_source["source_message_id"]
    count = one("SELECT COUNT(*) AS total FROM messages WHERE thread_id=? AND role='user'", (thread_id,))
    saved_plan = one(
        "SELECT plan,trace FROM messages WHERE thread_id=? AND role='assistant' AND plan IS NOT NULL "
        "ORDER BY created_at DESC,id DESC LIMIT 1", (thread_id,),
    ) if state.get("identity_mode") == "clinical" and state.get("identity_scope") and not state.get("identity_urgent") else None
    try:
        saved_trace = json.loads(saved_plan["trace"] or "{}") if saved_plan else {}
        active_plan = json.loads(saved_plan["plan"]) if saved_plan and saved_trace.get("identity_scope") == state.get("identity_scope") else {}
    except (TypeError, ValueError):
        active_plan = {}
    return {"profile": profile, "manual_profile_hash": patient_fingerprint(profile) if personal else "",
            "safety_state": safety_state, "safety_state_version": safety_state["version"],
            "symptom_facts": safety_state.get("symptom_facts", []),
            "source_message_id": source_message_id,
            "conversation_turns": count["total"] if count else 0,
            "active_plan": migrate_plan(active_plan) if isinstance(active_plan, dict) and state.get("identity_mode") == "clinical" else {}}


def subject_resolve(state: FlowState) -> FlowState:
    """Resolve current patient before any account medical information is read."""
    previous = safety_store.load_state(state["user_id"], state["thread_id"])
    run = one("SELECT source_message_id,source_files FROM chat_runs WHERE id=? AND user_id=?", (state.get("run_id", ""), state["user_id"]))
    source_id = run["source_message_id"] if run else state.get("source_message_id", "")
    text, attached = state.get("message", ""), attachment_text(state)
    current_symptoms = symptom_parser.extract(text, subject="unknown", message_id=source_id, thread_id=state["thread_id"])
    attached_symptoms = symptom_parser.extract(attached, subject="unknown", message_id=source_id,
                                              thread_id=state["thread_id"], kind="attachment")
    for fact in attached_symptoms:
        fact["source"]["upload_ids"] = json.loads(run["source_files"]) if run else []
    current_symptoms += attached_symptoms
    risks = extract_facts(text, source_id=source_id, thread_id=state["thread_id"])
    urgent = any(f["category"] == "urgent" and f["assertion"] == "present" and f["temporal_scope"] == "current"
                 for f in risks + extract_facts(attached, kind="attachment"))
    general = bool(re.search(r"什么是|是什么意思|怎么理解|请.*(?:科普|介绍|解释)|^(?:科普|介绍|解释)|如果|假如|假设|例如|举例", text))
    personal_report = bool(re.search(r"我(?:现在|目前|最近)?(?:有|出现|感觉|正在|咳嗽|失眠|头痛|腹痛)|(?:妈妈|母亲|爸爸|父亲|孩子|本人).*(?:有|出现|咳嗽|失眠|头痛)", text))
    clinical = bool(attached or current_symptoms or risks or previous.get("pending") or previous.get("episode_confirmation") or previous.get("status") in {"restricted", "needs_info", "blocked"} or
                    symptom_parser.duration_text(text) or
                    (previous.get("subject_confirmed") and mentioned_herbs(text)) or
                    re.search(r"替.{0,10}(?:咨询|问)|帮.{0,10}咨询|换成|改为|我自己|本人|新的问题|新问题|继续上次|还是上次|接着上次|同一个问题", text) or
                    re.search(r"怎么办|不舒服|不适|病历|不方便说|不想说|跳过", text) or
                    previous.get("clinical", {}).get(previous["subject"], {}).get("pending_context"))
    from .knowledge import is_followup
    knowledge = not urgent and ((general and not personal_report and not risks) or not clinical or
                                (is_followup(text) and not personal_report and not risks and not current_symptoms and not attached))
    if knowledge:
        safe, current = make_context({}, text, "", message_id=source_id, thread_id=state["thread_id"])
        safe.update(subject="unknown", subject_confirmed=False, identity_required=False)
        return {"identity_mode": "knowledge", "identity_scope": "", "identity_urgent": False, "identity_resolved_now": False, "identity_previous": previous,
                "identity_current": current, "safety_state": safe, "safety_state_version": previous["version"],
                "source_message_id": source_id, "symptom_facts": [], "symptom_pending": None,
                "pending_context": "", "pending_question": ""}
    safe, current = make_context(previous, text, attached, message_id=source_id, thread_id=state["thread_id"])
    if re.search(r"两个孩子|两位孩子|大宝.*二宝|哥哥.*弟弟", text):
        safe.update(subject="unknown", subject_confirmed=False, pending={"key": "subject", "subject": "unknown"})
    safe["identity_required"] = True
    if not safe.get("subject_confirmed"):
        safe["subject_evidence"] = None
    if safe.get("subject_confirmed") and subject_of(text) == safe["subject"]:
        safe["subject_evidence"] = {"message_id": source_id, "thread_id": state["thread_id"], "quote": text[:400]}
    clinical_states = safe.setdefault("clinical", {})
    prior_clinical = clinical_states.get(previous["subject"], {})
    switched = safe["subject"] != previous["subject"]
    resolving = previous.get("pending", {}).get("key") == "subject" if previous.get("pending") else False
    target_clinical = clinical_states.get(safe["subject"], {})
    facts = safe.get("symptom_facts", [])
    if resolving and safe.get("subject_confirmed"):
        for fact in facts:
            if fact["subject"] == "unknown" or (fact["subject"] == "other:未明确" and safe["subject"].startswith("other:")):
                fact.update(subject=safe["subject"], ownership_confirmed_by=source_id)
        target_clinical = {**prior_clinical, "pending_context": prior_clinical.get("pending_context") or safe.get("episode", ""), "symptom_pending": None}
        # A mixed-person complaint must not be flattened into one person's
        # episode after the ownership answer.
        if any(f.get("subject") not in {safe["subject"], "unknown"} for f in facts + safe.get("facts", [])):
            target_clinical["pending_context"] = symptom_parser.describe(facts, safe["subject"])
            safe["episode"] = target_clinical["pending_context"]
        clinical_states[safe["subject"]] = target_clinical
    elif not safe.get("subject_confirmed"):
        facts = symptom_parser.merge(facts, current_symptoms)
        target_clinical = {"pending_context": safe.get("episode", text), "pending_question": "", "symptom_pending": None}
    if switched:
        safe["subject_epoch"] = safe.get("subject_epoch", 0) + 1
    safe["symptom_facts"] = facts
    scope = f'{state["thread_id"]}:{safe["subject"]}:{safe.get("subject_epoch", 0)}' if safe.get("subject_confirmed") else ""
    return {"identity_mode": "clinical", "identity_scope": scope, "identity_urgent": urgent,
            "identity_resolved_now": bool(resolving and safe.get("subject_confirmed")), "identity_previous": previous,
            "identity_current": current, "safety_state": safe, "safety_state_version": previous["version"],
            "source_message_id": source_id, "symptom_facts": facts,
            **{key: target_clinical.get(key, None if key == "symptom_pending" else "")
               for key in ("pending_context", "pending_question", "symptom_pending")}}


def rule_pre_safety(state: FlowState) -> FlowState:
    text = state.get("message", "")
    attached = attachment_text(state)
    urgent_now = any(f["category"] == "urgent" and f["assertion"] == "present" and f["temporal_scope"] == "current"
                     for f in extract_facts(text) + extract_facts(attached, kind="attachment"))
    if urgent_now or gate_context(state.get("safety_state") or make_context({}, text, attached)[0])["decision"] == "block":
        return {
            "intent_type": "safety_rejected", "symptoms": [],
            "episode_text": text,
            "response": "你提供的信息中出现可能需要及时处理的情况。若症状正在发生，请尽快联系当地急救服务或就近医疗机构，不要等待在线问答。",
        }
    if re.search(r"(忽略|绕过|跳过|关闭|不要).{0,12}(安全|审核|审查|规则)|无论.{0,12}(风险|过敏)", text):
        return {"intent_type": "safety_rejected", "symptoms": [], "episode_text": text,
                "risk_categories": ["bypass"],
                "response": "安全审查不能跳过。本轮不生成方剂或用药建议，请补充咨询对象及实际情况。"}
    dangerous = r"夹竹桃|砒霜|雷公藤|马钱子|生川乌|生草乌|川乌|草乌|附子"
    unsafe_self_use = (
        (contains_dosing(text) and bool(mentioned_herbs(text)))
        or bool(re.search(rf"(?:{dangerous}).{{0,18}}(?:大量|加倍|磨粉|吞服|偏方|煮水|长期服)", text))
        or bool(re.search(r"(?:不认识|不知道名字|来源不明).{0,12}(?:草药|药材).{0,12}(?:一起|混合|煎服|服用)", text))
    )
    if unsafe_self_use:
        return {
            "intent_type": "safety_rejected", "symptoms": [], "episode_text": text,
            "safety_errors": ["高风险药材、剂量或来源不明的自行用药请求"],
            "response": "## 安全警告\n该请求涉及高风险药材、异常剂量或来源不明的自行用药。请勿自行尝试；如已服用或出现不适，请立即联系急救、就近医疗机构或当地中毒咨询机构。",
        }
    if re.search(r"加(?:入|上)?|添加|配(?:伍)?|搭配|合用|同用|同服|一起(?:用|服)|能(?:用|服|配)|可以用|相反|相克|冲突|安全吗", text):
        requested = mentioned_herbs(text)
        if requested:
            plan = state.get("active_plan", {})
            base = names(plan) if isinstance(plan.get("herbs"), list) else []
            removed = {name for name in requested if re.search(
                rf"(?:去掉|去除|删除|不放|不用|撤掉|剔除)\s*{re.escape(name)}", text)}
            base = [name for name in base if normalize(name) not in removed]
            additions = [name for name in requested if name not in removed]
            if additions:
                candidate = base + additions
                errors = check_herbs(candidate) if len(candidate) >= 2 else []
                allergy_text = str(state.get("profile", {}).get("allergies", ""))
                for herb in {normalize(item) for item in candidate}:
                    if herb in allergy_text:
                        errors.append(f"画像提示可能对{herb}过敏")
                where = f'上一轮展示的“{plan.get("name", "")}”知识示例' if base else "你提到的药材"
                if errors:
                    details = "；".join(errors)
                    response = (
                        f"## 安全警告：发现配伍或过敏风险\n{where}与本次加药请求合并检查后命中：**{details}**。"
                        "\n\n请不要自行加药或调整处方；请让开方医师或中药师核对完整处方、病史与用药情况。"
                        "\n\n配伍规则参考：[深圳市《中药饮片处方审核规范》DB4403/T 142—2021 附录 C]"
                        "(https://wjw.sz.gov.cn/attachment/0/758/758943/8708423.pdf)。"
                    )
                elif not base and len(additions) == 1:
                    response = (
                        f"目前只有**{additions[0]}**这一味药的信息，无法判断它能否加入你的处方。"
                        "请提供完整药材清单，并由开方医师或中药师核对；不要自行加药。"
                    )
                else:
                    response = (
                        f"已按内置配伍规则核对{where}与本次提到的药材，暂未命中已收录的十八反、十九畏规则。"
                        "这不代表加药适合你或可以安全服用；请让开方医师或中药师结合完整处方与病史判断。"
                    )
                return {"intent_type": "safety_rejected" if errors else "safety_advisory",
                        "symptoms": [],
                        "episode_text": text, "safety_errors": errors, "response": response}
    return {"intent_type": ""}


async def pre_safety(state: FlowState) -> FlowState:
    text = state.get("message", "") + "；" + attachment_text(state)
    previous = state.get("identity_previous", state["safety_state"])
    safe, current = state["safety_state"], state.get("identity_current", [])
    source_run = one("SELECT source_files FROM chat_runs WHERE id=? AND user_id=?", (state.get("run_id", ""), state["user_id"]))
    for fact in current:
        if fact["source"]["kind"] == "attachment":
            fact["source"]["upload_ids"] = json.loads(source_run["source_files"]) if source_run else []
    # Keep regular inquiry state separately for each consultation subject.
    switched = safe["subject"] != previous["subject"]
    followup = {}
    profile = dict(state.get("profile", {})) if safe["subject"] == "self" and safe.get("subject_confirmed") else {}
    allergies = [f["entity"] for f in applicable_facts(safe)
                 if f["category"] == "allergy" and f["assertion"] == "present" and f["subject"] == safe["subject"]]
    if allergies:
        profile["allergies"] = "；".join([profile.get("allergies", ""), *allergies])
    owned = [f for f in safe.get("facts", []) if f.get("ownership_confirmed_by") == state.get("source_message_id") or
             (state.get("identity_resolved_now") and f.get("subject") == "self" and f.get("source", {}).get("thread_id") == state["thread_id"])]
    base = {"profile": profile, "safety_state": safe, "profile_suggestions": profile_candidates(current + owned),
            **followup}
    state = {**state, **base}
    rules = rule_pre_safety(state)
    if rules.get("intent_type") in {"safety_rejected", "safety_advisory"}:
        safe["status"] = "blocked" if rules["intent_type"] == "safety_rejected" else "restricted"
        return {**base, **rules, "preflight": {
            "decision": "block" if rules["intent_type"] == "safety_rejected" else "clarify",
            "risk_categories": rules.get("risk_categories", ["deterministic"]), "reason": "确定性规则终止治疗路径",
            "required_questions": []}}
    if state.get("identity_mode") == "knowledge":
        return {**base, "preflight": {"decision": "allow", "risk_categories": [], "required_questions": []}, "intent_type": ""}
    identity_gate = gate_context(safe)
    if "subject" in identity_gate.get("question_keys", []):
        response = response_for(safe, identity_gate)
        if re.search(r"跳过|不想说|不方便说", state.get("message", "")):
            response = "可以先了解一般信息；对象未明确时不会结合个人病史生成方剂示例。请留意症状变化，如持续或加重，及时就医。"
        return {**base, "preflight": identity_gate, "intent_type": "safety_advisory", "profile_suggestions": [],
                "episode_text": safe.get("episode", ""), "response": response}
    if state.get("episode_pending"):
        return {**base, "intent_type": "consult", "preflight": {"decision": "allow", "risk_categories": [], "required_questions": []}}
    try:
        result = await assess(text, profile, mock=MODEL_MODE == "mock", context=safe)
        additions = result.pop("observations", [])
        for fact in additions:
            fact["source"].update(id=state.get("source_message_id", ""), thread_id=state["thread_id"], extractor="model")
            if fact["source"]["quote"] in state.get("message", ""):
                fact["source"]["kind"] = "message"
            else:
                fact["source"]["kind"] = "attachment"
        base["profile_suggestions"] += profile_candidates(additions)
        safe["facts"] = merge_facts(safe["facts"], additions)
    except Exception:
        result = {"decision": "block", "risk_categories": ["preflight_unavailable"],
                  "reason": "预审超时、失败或结构无法校验", "required_questions": []}
    analysis_only = bool(result.get("analysis_allowed") and identity_gate.get("analysis_allowed")
                         and result["decision"] == "clarify" and not result.get("question_keys"))
    if result["decision"] != "allow" and not analysis_only:
        response = response_for(safe, result)
        return {**base, "safety_state": safe, "preflight": result,
                "intent_type": "safety_rejected" if result["decision"] == "block" else "safety_advisory",
                "episode_text": safe.get("episode", state.get("message", "")), "response": response}
    # Resume the original complaint after safety clarification without losing it.
    if previous.get("status") in {"needs_info", "restricted", "blocked"} and not state.get("pending_context"):
        base["pending_context"] = safe.get("episode", "")
    reasons = (["过敏原“" + "、".join(uncovered_allergens(safe)) + "”尚未覆盖自动药材风险校验，个人用药适用性需人工复核"]
               if analysis_only else [])
    safe.update(status="restricted" if analysis_only else "resolved", pending=None, episode="")
    return {**base, "safety_state": safe, "preflight": result, "intent_type": "",
            "analysis_only": analysis_only, "manual_review_reasons": reasons,
            "analysis_questions": result.get("review_questions", [])[:3] if analysis_only else []}


def pre_safety_route(state: FlowState) -> str:
    return "blocked" if state.get("intent_type") in {"safety_rejected", "safety_advisory"} else "allowed"


def symptom_observations(text: str) -> tuple[list[str], list[str]]:
    """Legacy list interface, derived from source-backed current observations."""
    positive, denied, _ = symptom_parser.view(symptom_parser.merge([], symptom_parser.extract(text)), "self")
    return positive, denied


def followup_question(symptoms: list[str], denied: list[str]) -> str:
    questions = ["这种情况持续多久了？"]
    if any(item in symptoms for item in ("腹胀", "腹痛", "胃痛", "大便稀")):
        missing = [name for name in ("腹痛", "口干", "口苦") if name not in symptoms and name not in denied]
        if missing:
            questions.append("是否伴有" + "、".join(missing) + "？")
        if "大便稀" in symptoms:
            questions.append("大便情况还有哪些变化？")
        if "舌苔白" not in symptoms and "舌苔白" not in denied:
            questions.append("若方便，请用文字描述舌苔的颜色和厚薄。")
    elif "失眠" in symptoms or "睡眠" in symptoms:
        questions.append("主要是入睡困难，还是容易醒来？是否伴有其他不适？")
    elif "咳嗽" in symptoms:
        questions.append("是否有痰或其他不适？")
    else:
        questions.append("是否伴有其他不适？")
    return "".join(questions)


def inquire(state: FlowState) -> FlowState:
    text = state.get("message", "")
    if state.get("episode_pending"):
        return {"intent_type": "consult", "response": episodes.QUESTION, "pending_question": episodes.QUESTION,
                "symptoms": [], "denied_symptoms": [], "symptom_facts": state.get("symptom_facts", [])}
    if state.get("identity_mode") == "knowledge":
        return {"intent_type": "chat", "symptom_facts": [], "symptoms": [], "pending_context": "", "pending_question": "",
                "response": "可以讨论一般中医知识。若需要结合具体不适整理信息，请说明为谁咨询、主要症状及持续时间。"}
    previous = state.get("pending_context") or ""
    current = text + ("；附件：" + state["file_text"][:800] if state.get("file_text") else "")
    combined = f"{previous}；{current}" if previous else current
    subject = state["safety_state"]["subject"]
    facts = state.get("symptom_facts", [])
    # Migrate an unfinished old inquiry once, without labelling it confirmed.
    if previous and not facts:
        facts = symptom_parser.extract(previous, subject=subject, kind="legacy", thread_id=state["thread_id"])
        for fact in facts:
            fact["subject"] = subject
    source = {"subject": subject, "message_id": state.get("source_message_id", ""), "thread_id": state["thread_id"]}
    pending = state.get("symptom_pending")
    new = symptom_parser.extract(text, **source)
    new += symptom_parser.bind_answer(facts, pending, text, **source)
    if not new:
        for clause in re.finditer(r"[^，,。；;！？!?\n]+", text):
            new += symptom_parser.bind_details(facts, clause[0], **source, start=clause.start())
    new += symptom_parser.extract(attachment_text(state), **source, kind="attachment")
    for fact in new:
        fact["episode_id"] = state.get("episode_id", "")
    if state.get("file_text"):
        uploaded = one("SELECT source_files FROM chat_runs WHERE id=? AND user_id=?", (state.get("run_id", ""), state["user_id"]))
        for fact in new:
            if fact["source"]["kind"] == "attachment":
                fact["source"]["upload_ids"] = json.loads(uploaded["source_files"]) if uploaded else []
    if pending and pending["kind"] in {"assertion", "time", "conflict", "mapping"}:
        for fact in new:
            if (fact["symptom"] == pending["symptom"] and fact["subject"] == subject
                    and fact["assertion"] in {"present", "absent"} and fact["time_scope"] == "current"):
                fact["correction"] = True
    # Resolving patient ownership requires an explicit consultation target. It
    # binds unknown records only; other people's named observations stay separate.
    ownership_pending = pending or state["safety_state"].get("clinical", {}).get("unknown", {}).get("symptom_pending")
    if (subject != "unknown" and subject_of(text) == subject and
            ownership_pending and ownership_pending["kind"] == "subject"):
        facts = json.loads(json.dumps(facts))
        for fact in facts:
            if fact["subject"] == "unknown":
                fact["subject"] = subject
                fact["ownership_confirmed_by"] = source["message_id"]
    facts = symptom_parser.merge(facts, new)
    symptoms, denied_symptoms, unresolved = symptom_parser.view(facts, subject)
    if unresolved:
        unresolved["id"] = (pending["id"] if pending and pending.get("id") and
                            (pending["kind"], pending["symptom"]) == (unresolved["kind"], unresolved["symptom"])
                            else str(uuid4()))
    base = {"symptom_facts": facts, "symptom_pending": unresolved,
            "symptoms": symptoms, "denied_symptoms": denied_symptoms, "episode_text": combined}
    if unresolved:
        return {**base, "intent_type": "consult", "pending_context": combined,
                "pending_question": unresolved["question"], "response": unresolved["question"]}
    if previous and re.fullmatch(r"(?:不知道|不清楚|跳过|不是|是|是的|有|没有|没有了)[。！! ]*", text.strip()) and not new:
        question = state.get("pending_question") or followup_question(symptoms, denied_symptoms)
        return {**base, "intent_type": "consult", "pending_context": combined,
                "pending_question": question, "response": "这条回答还不能补全资料。" + question}
    if not symptoms and (denied_symptoms or facts):
        return {**base, "intent_type": "consult", "pending_context": "", "pending_question": "",
                "response": "已记录你对当前症状的说明。若还有其他不适，请具体描述；既往记录仍保留。"}
    if not symptoms and not previous:
        return {
            **base, "intent_type": "chat", "pending_context": "",
            "pending_question": "",
            "episode_text": combined,
            "response": "你好，我可以帮助整理症状信息，并展示中医知识检索与会诊流程。请描述主要不适、持续时间，以及有无过敏史。内容仅供学习研究参考。",
        }
    has_duration = bool(symptom_parser.duration_text(combined)) or any(
        f.get("duration_text") for f in facts if f["subject"] == subject and f["time_scope"] == "current" and not f.get("superseded_by"))
    has_detail = bool(re.search(
        r"伴|同时|并|没有|无|加重|减轻|反复|常|不用查古籍|不需要查文献|无需检索",
        combined,
    ))
    has_time_of_day = bool(re.search(r"白天|早上|晨起|中午|下午|晚上|夜间|夜里|睡前", combined))
    has_enough_context = (
        (has_duration and (len(symptoms) >= 2 or has_detail))
        or (len(symptoms) >= 2 and has_time_of_day)
    )
    identity_only = state.get("identity_resolved_now") and not new
    if (not previous or identity_only) and len((previous if identity_only else combined).strip()) < 18 and not has_enough_context:
        question = followup_question(symptoms, denied_symptoms)
        return {
            **base, "intent_type": "consult",
            "episode_text": combined,
            "pending_context": combined, "pending_question": question,
            "response": "为了更好地整理信息，请补充：" + question,
        }
    return {
        **base, "intent_type": "diagnose",
        "denied_symptoms": denied_symptoms,
        "episode_text": combined,
        "pending_context": "", "pending_question": "",
    }


def main_route(state: FlowState) -> str:
    if state.get("identity_mode") == "knowledge" and state.get("intent_type") == "chat":
        return "knowledge"
    return "diagnose" if state["intent_type"] == "diagnose" else "end"


def rag_decision(state: FlowState) -> FlowState:
    explicit_skip = any(term in state.get("message", "") for term in ("不用查古籍", "不需要查文献", "无需检索"))
    required = not explicit_skip and bool(state.get("symptoms") or re.search(r"古籍|文献|出处|依据", state.get("message", "")))
    return {"search_required": required}


def rag_route(state: FlowState) -> str:
    return "search" if state.get("search_required") else "skip"


async def search(state: FlowState) -> FlowState:
    plan = state.get("search_plan") or build_search_plan(state)
    docs, trace = await retrieve_classics(plan)
    return {"docs": docs, "retrieval_warning": trace["warning"], "search_query": plan["keyword_query"],
            "search_plan": plan, "retrieval_trace": trace,
            "retrieval_history": [*state.get("retrieval_history", []), trace],
            "search_attempts": state.get("search_attempts", 0) + 1}


def evaluate_retrieval(state: FlowState) -> FlowState:
    docs = state.get("docs", [])
    # Each passage already passed its own admission check. Do not discard
    # semantically relevant vector-only evidence for lacking an exact symptom word.
    score = 1.0 if docs else 0.0
    warning = state.get("retrieval_warning", "")
    if (score < 0.5 and state.get("retrieval_trace", {}).get("outcome", "no_match") == "no_match"
            and (state.get("search_attempts", 0) >= MAX_SEARCH_ATTEMPTS or next_search_plan(state) is None
                 or not state.get("retrieval_trace", {}).get("retryable", True))):
        warning = (warning + " " if warning else "") + "检索结果相关性不足，未引用古籍片段。"
        docs = []
    return {"retrieval_score": score, "docs": docs, "retrieval_warning": warning}


def next_search_plan(state: FlowState):
    previous = state.get("search_plan", {})
    for attempt in range(state.get("search_attempts", 0), MAX_SEARCH_ATTEMPTS):
        plan = build_search_plan(state, attempt=attempt)
        if (plan["keyword_query"], plan["semantic_query"]) != (previous.get("keyword_query"), previous.get("semantic_query")):
            return plan
    return None


def retrieval_route(state: FlowState) -> str:
    trace = state.get("retrieval_trace", {})
    if (state.get("retrieval_score", 0.0) < 0.5 and state.get("search_attempts", 0) < MAX_SEARCH_ATTEMPTS
            and trace.get("outcome", "no_match") == "no_match" and trace.get("retryable", True)):
        if next_search_plan(state) is not None:
            return "rewrite"
    return "ready"


def rewrite_query(state: FlowState) -> FlowState:
    plan = next_search_plan(state) or state["search_plan"]
    return {"search_query": plan["keyword_query"], "search_plan": plan}


def evidence_ready(_state: FlowState) -> FlowState:
    return {}


def _context(state: FlowState) -> str:
    return clinical_context(state)


async def bianzheng(state: FlowState) -> FlowState:
    if MODEL_MODE == "mock":
        # Exercise the same source/budget contract in offline evaluations.
        _context(state)
        symptoms = "、".join(state.get("symptoms", [])) or "所述不适"
        denied = "、".join(state.get("denied_symptoms", []))
        diagnosis = (f"根据已提供信息，用户描述主要涉及{symptoms}。"
                     + (f"用户表示没有{denied}。" if denied else "")
                     + "信息不足以确定证型，以下仅展示分析流程。")
    else:
        diagnosis = await acomplete(
            "你是中医知识辅助系统。区分用户报告的症状与明确否认的症状；后者不能作为阳性依据。"
            "历史自述不是本轮已确认症状，保留时间和否定含义，不得覆盖当前明确说明。"
            "只解释可能的辨证思路和不确定性，不作确诊，不编造古籍出处或药物剂量。",
            _context(state), role="bianzheng",
        )

    context = invocation.get()
    if MODEL_MODE == "mock" and context and context.emit:
        for offset in range(0, len(diagnosis), 24):
            context.check()
            context.emit("candidate.delta", {"channel": "body", "text": diagnosis[offset:offset + 24]})
            await asyncio.sleep(0)
    return {"diagnosis": diagnosis, "bianzheng_note": diagnosis}


async def _make_plan(state: FlowState, *, revision: bool = False) -> dict:
    if MODEL_MODE == "mock":
        _context(state)
        if revision:
            # A demo must not invent a replacement prescription after a failed check.
            return {"name": "安全校验后未提供候选方剂", "herbs": []}
        symptoms = state.get("symptoms", [])
        if any(item in symptoms for item in ["失眠", "睡眠"]):
            name = "酸枣仁汤"
        elif any(item in symptoms for item in ["怕冷", "鼻塞"]):
            name = "桂枝汤"
        else:
            name = ""
        if name:
            import json
            row = one("SELECT name,structured_herbs FROM formulas WHERE name=?", (name,))
            plan = {"name": row["name"], "herbs": json.loads(row["structured_herbs"])} if row else {"name": "基础方剂数据缺失", "herbs": []}
        else:
            plan = {"name": "暂无匹配的示例方剂", "herbs": []}
    else:
        retry_context = ("\n上次候选未通过安全校验：" + "；".join(state.get("safety_errors", []))) if revision else ""
        evidence_names = names(state.get("treatment_plan", {})) + mentioned_herbs(state.get("message", "") + state.get("diagnosis", ""))
        for doc in state.get("docs", []):
            evidence_names += mentioned_herbs(passage_excerpt(doc))
        plan = await acomplete_json(
            "你是中医知识演示系统。返回候选方剂 JSON，按 Schema 返回 name 和结构化 herbs。每味药包含 herb_id、name、dosage、preparation、source_ids、verification_status。剂量只能依据提供的知识记录，缺少依据时 min/max 为空；所有 verification_status 均为 unverified，不要自行声称核验。禁止自行抓药指令。",
            _context(state) + "\n辨证思路：" + state.get("diagnosis", "") + retry_context + "\n可用剂量出处（未列出的药材必须留空剂量）：" + knowledge_context(evidence_names), role="formula", schema=CandidatePlan,
        )
        if not isinstance(plan.get("name"), str) or not isinstance(plan.get("herbs"), list):
            plan = {"name": "未能解析候选方剂", "herbs": []}
    plan = migrate_plan(plan)
    for herb in plan.get("herbs", []):
        herb["verification_status"] = "unverified"
        if state.get("analysis_only"):
            herb["dosage"].update(min=None, max=None)
    return plan


async def formula(state: FlowState) -> FlowState:
    plan = await _make_plan(state)
    return {"treatment_plan": plan, "formula_note": f'候选方剂：{plan["name"]}'}


async def revise_formula(state: FlowState) -> FlowState:
    plan = await _make_plan(state, revision=True)
    note = f'调整后候选方剂：{plan["name"]}'
    notes = [{**item, "content": note} if item.get("expert") == "方剂" else item
             for item in state.get("consultation_notes", [])]
    return {"treatment_plan": plan, "formula_note": note, "consultation_notes": notes,
            "safety_retry_count": state.get("safety_retry_count", 0) + 1}


def herb_expert(state: FlowState) -> FlowState:
    allergies = state.get("profile", {}).get("allergies", "")
    return {"herb_note": f"已检查基础配伍规则。画像中的过敏信息：{allergies or '未填写'}。"}


def synthesize(state: FlowState) -> FlowState:
    return {"consultation_notes": [
        {"expert": "辨证", "content": state.get("bianzheng_note", "")},
        {"expert": "方剂", "content": state.get("formula_note", "")},
        {"expert": "本草", "content": state.get("herb_note", "")},
    ]}


async def _review(plan: dict, context: str) -> tuple[bool, str]:
    if MODEL_MODE == "mock":
        return True, ""
    result = await acomplete_json(
        "你是独立安全审查者。审查候选方剂核对药材标识、剂量范围与单位、炮制、出处及患者过敏/孕哺/年龄/合并用药条件。缺少依据的剂量不可声称已核验，不提供自行用药指令。"
        "返回 JSON：approved 布尔值、reason 字符串。确定性配伍校验由程序另行执行。",
        context + "\n候选方案：" + str(plan), role="reviewer", schema=ReviewResult,
    )
    return result.get("approved") is True, str(result.get("reason", "审查未通过"))


async def reviewer(state: FlowState) -> FlowState:
    approved, reason = await _review(state.get("treatment_plan", {}), _context(state))
    # A rejection explanation can repeat the rejected prescription: never publish it.
    return {"reviewer_approved": approved, "reviewer_hash": fingerprint(state.get("treatment_plan", {})), "safety_errors": [] if approved else ["独立审查未通过"]}



def safety(state: FlowState) -> FlowState:
    plan, review = review_plan(state.get("treatment_plan", {}), state.get("profile", {}))
    errors = review["errors"]
    if not state.get("reviewer_approved", True):
        errors = state.get("safety_errors", []) + errors
    if state.get("reviewer_hash") != fingerprint(state.get("treatment_plan", {})):
        errors.append("候选数据发生变化，必须重新审查")
    if contains_dosing(state.get("diagnosis", "")):
        errors.append("正文剂量必须来自结构化药材数据")
    # Clearing an unsupported dose changes the reviewed material: reject and regenerate.
    if fingerprint(plan) != state.get("reviewer_hash"):
        errors.append("剂量已因缺少依据清空，候选需重新审查")
    errors += state.get("manual_review_reasons", [])
    review = {**review, "approved": not errors, "errors": errors}
    findings = list(dict.fromkeys(state.get("safety_findings", []) + errors))
    return {"treatment_plan": plan, "prescription_review": review,
            "reviewed_hash": fingerprint(plan) if not errors else "",
            "safety_errors": errors, "safety_findings": findings}


def safety_route(state: FlowState) -> str:
    if state.get("analysis_only"):
        return "finish"
    if state.get("safety_errors") and state.get("safety_retry_count", 0) < MAX_SAFETY_REVISIONS:
        return "revise"
    return "finish"


def context_budget(state: FlowState) -> FlowState:
    # Informational stage; every actual model call enforces its own input budget.
    return {"context_chars": len(_context(state))}


def episode_write(state: FlowState) -> FlowState:
    if state.get("identity_mode") == "clinical" and state.get("safety_state", {}).get("subject_confirmed"):
        text, data = episodes.summary(state)
        return {"context_summary": text, "episode_summary": data}
    return {"context_summary": "", "episode_summary": {}}


def formula_references(state: FlowState) -> list[dict]:
    """Public source entries, separate from an unapproved personal candidate."""
    candidates = [state.get("treatment_plan", {}).get("name", "")]
    for observation in state.get("agent_traces", {}).get("formula", []):
        if observation.get("tool") == "search_formulas" and observation.get("status") == "success":
            candidates += [row.get("name", "") for row in observation.get("result", {}).get("formulas", [])]
    rows = []
    for name in dict.fromkeys(candidates):
        row = one("SELECT name,source,indication FROM formulas WHERE name=?", (name,))
        if row and not contains_dosing(row["name"] + row["source"]):
            if contains_dosing(row["indication"]):
                row["indication"] = "原条目含用药内容，本轮省略。"
            rows.append(row)
        if len(rows) == 3:
            break
    return rows


def herb_references(state: FlowState) -> list[dict]:
    """Show only public entries actually looked up, never an executable herb list."""
    rows = []
    seen = set()
    for role in ("herb", "formula"):
        for observation in state.get("agent_traces", {}).get(role, []):
            if observation.get("tool") != "lookup_herb" or observation.get("status") != "success":
                continue
            result = observation.get("result", {})
            name = result.get("name") if result.get("found") else None
            if not name or name in seen:
                continue
            row = one("SELECT name,properties,effects,contraindications,source FROM herbs WHERE name=?", (name,))
            if not row or contains_dosing(row["name"] + row["source"]):
                continue
            for field in ("properties", "effects", "contraindications"):
                if contains_dosing(row[field] or ""):
                    row[field] = "涉及用药内容，本轮省略。"
            rows.append(row)
            seen.add(name)
            if len(rows) == 4:
                return rows
    return rows


def herb_reference_text(row: dict) -> str:
    return (row["name"] + "：" + "；".join(label + row[key] for label, key in
            (("性味归经：", "properties"), ("功效：", "effects"), ("条目禁忌：", "contraindications")) if row.get(key))
            + "；知识来源：" + row["source"] + "。")


def public_safety_warnings(state: FlowState) -> list[str]:
    warnings = []
    allergens = list(dict.fromkeys(f["entity"] for f in applicable_facts(state.get("safety_state", {}))
                                  if f["category"] == "allergy" and f["assertion"] == "present")) if state.get("safety_state") else []
    if allergens:
        warnings.append("已记录过敏史：" + "、".join(allergens) + "。拟用药物的相关风险需医师或药师核对。")
    for error in state.get("safety_findings", []) or state.get("safety_errors", []):
        if re.search(r"十八反|十九畏|画像提示可能对", error) and not contains_dosing(error):
            warnings.append(error[:250])
    if state.get("analysis_only") or state.get("preliminary_analysis"):
        warnings.append("当前辨证为初步分析，个人用药适用性尚未确认。")
    if state.get("safety_errors"):
        warnings.append("本轮用药建议尚未完成安全核验，需由执业中医师或药师复核。")
    warnings.append("请勿依据本次分析自行抓药或调整用药。")
    return list(dict.fromkeys(warnings))


def final_consultation_notes(state: FlowState) -> list[dict]:
    """Create display notes only after the reviewer and deterministic checks finish."""
    if not state.get("consultation_notes"):
        return []
    errors = (state.get("safety_findings", []) or state.get("safety_errors", []))
    if not state.get("safety_errors"):
        errors = []
    plan = state.get("treatment_plan", {})
    symptoms = "、".join(state.get("symptoms", [])) or "未提取到明确症状词"
    denied = "、".join(state.get("denied_symptoms", []))
    docs = state.get("docs", [])
    source = (f'本轮检索片段：《{docs[0]["title"]}》{docs[0]["chapter"]}'
              if docs else "本轮没有可引用的古籍片段")
    allergies = state.get("profile", {}).get("allergies", "")
    formula_source = "内置方剂知识种子" if MODEL_MODE == "mock" else "已配置的模型生成候选"
    notes = [
        {
            "expert": "辨证", "status": "资料有限",
            "content": state.get("diagnosis", "现有信息不足以分析。"),
            "evidence": (f"本轮识别到的描述：{symptoms}；"
                         + (f"用户表示没有：{denied}；" if denied else "") + f"{source}。"),
            "reasoning": "仅整理现有描述与检索信息；缺少面诊资料，不能据此确定证型。",
        },
        {
            "expert": "方剂", "status": "需人工复核" if errors else "知识示例",
            "content": ("候选方剂未通过安全审查，名称与药材清单已隐藏。" if errors
                        else f'候选知识示例：{plan.get("name", "暂无候选")}。'),
            "evidence": ("候选内容未展示。" if errors else f"候选来源：{formula_source}。"),
            "reasoning": ("安全校验未通过，不提供替代用药方案。" if errors
                          else "该候选用于知识对照；现有资料不能证明其适合当前用户。"),
            "recommendation": ("请由执业医师和中药师审核完整处方。" if errors
                               else "剂量与出处按结构化数据展示，未核验项需医师和药师复核，不可自行服用。"),
        },
        {
            "expert": "本草", "status": "发现风险" if errors else "规则已核对",
            "content": ("确定性校验发现需人工处理的项目。" if errors
                        else "已核对已收录药材名称、十八反十九畏规则和画像中的过敏信息。"),
            "evidence": f"画像过敏信息：{allergies or '未填写'}。",
            "reasoning": ("当前候选不能作为用药依据。" if errors
                          else "未命中内置规则不等于临床安全，仍需结合病史和完整处方审核。"),
            "warning": "；".join(errors) if errors else "",
            "recommendation": "请勿自行抓药、加药或调整处方。",
        },
    ]
    if state.get("analysis_only"):
        references = formula_references(state)
        notes[1].update(status="知识对照·待复核",
                        content=("\n".join(row["name"] + "：" + row["indication"] for row in references) if references else "本轮未取得可展示的方剂知识条目。")
                                + "这些是文献知识对照，未确认适用于当前咨询对象。",
                        evidence="；".join(row["name"] + "：" + row["source"] for row in references),
                        reasoning="过敏资料已传给各专家；当前未批准个人用药候选。")
        herb_rows = herb_references(state)
        notes[2].update(status="知识对照·待复核", content=("\n".join(herb_reference_text(row) for row in herb_rows)
                        if herb_rows else "本轮未取得可展示的本草条目，不能声称已完成具体药材知识核对。") + "\n个人用药适用性尚未确认。",
                        warning="；".join(errors))
    artifact = state.get("agent_artifacts", {}).get("bianzheng", {})
    notes[0].update(syndrome=artifact.get("syndrome", ""), confidence=artifact.get("confidence"),
                    summary=artifact.get("summary") or notes[0]["content"])
    comparison = state.get("agent_artifacts", {}).get("formula", {}).get("comparison", "")
    notes[1]["summary"] = notes[1]["content"] + ("\n" + comparison if comparison and not contains_dosing(comparison) else "")
    if errors:
        advice = "当前尚不能形成可执行的个人处方，需由执业中医师结合面诊资料复核。"
        if state.get("analysis_only"):
            references = formula_references(state)
            if references:
                advice += "\n\n可供医师核对的方剂知识条目：" + "、".join(row["name"] for row in references) + "。这些条目不表示已为你选方。"
    else:
        advice = render_plan(plan) + "\n\n请由执业中医师确认个人适用性，勿自行抓药。"
    notes[1]["prescription_advice"] = advice
    herb_rows = herb_references(state)
    notes[2]["summary"] = "\n".join(herb_reference_text(row).split("；知识来源：")[0] for row in herb_rows) or notes[2]["content"]
    notes[2]["public_warnings"] = public_safety_warnings(state)
    return notes


def finish(state: FlowState) -> FlowState:
    if state.get("intent_type") != "diagnose":
        response = state.get("response", "")
        if state.get("analysis_only"):
            response = "已记录过敏史，后续分析会保留这条风险信息。可以继续问诊与知识分析，个人用药适用性仍需复核。\n\n" + response
        return {"response": response}
    is_consultation = bool(state.get("consultation_notes"))
    diagnosis = state.get("diagnosis", "资料不足。")
    if contains_dosing(diagnosis):
        diagnosis = "生成内容包含不允许展示的用药指令，本轮仅保留受限答复，请由医师核对。"
        state = {**state, "diagnosis": diagnosis, "safety_errors": ["答复检查未通过：包含剂量或用法"]}
    lines = ["## 初步辨证（资料待补全）" if state.get("preliminary_analysis") else "## 会诊综合" if is_consultation else "## 辨证思路", diagnosis]
    docs = state.get("docs", [])
    if state.get("retrieval_warning"):
        lines.extend(["", state["retrieval_warning"]])
    if docs:
        lines.extend(["", "## 检索出处"])
        lines.extend(f'- 《{d["title"]}》{d["chapter"]}：'
                     + ("原文涉及古籍用药内容，摘录已省略。" if contains_dosing(passage_excerpt(d))
                        else passage_excerpt(d)) for d in docs)
    else:
        lines.extend(["", "本轮按要求未检索古籍。" if not state.get("search_required") else "当前没有可引用的古籍片段。"])
    errors = state.get("safety_errors", [])
    display_errors = (state.get("safety_findings", []) or errors) if errors else []
    plan = state.get("treatment_plan", {})
    if errors:
        if state.get("analysis_only"):
            lines.extend(["", "## 方剂知识对照（适用性待复核）"])
            references = formula_references(state)
            lines.extend(f'- **{row["name"]}**：条目记载：{row["indication"]}；知识来源：{row["source"]}。' for row in references)
            comparison = state.get("agent_artifacts", {}).get("formula", {}).get("comparison", "")
            if comparison and not contains_dosing(comparison):
                lines.extend(["", comparison])
            lines.append("以上仅为检索到的文献条目，不表示已选为个人处方；本轮不展示剂量或用法。" if references
                         else "本轮尚未取得可展示的方剂知识条目。")
            lines.extend(["", "## 本草知识对照"])
            herb_rows = herb_references(state)
            lines.extend("- " + herb_reference_text(row) for row in herb_rows)
            if not herb_rows:
                lines.append("本轮未取得可展示的本草条目，无法提供有来源的具体药材分析。")
        lines.extend(["", "## 安全校验",
                      "本轮个人用药适用性尚待复核，不生成可执行处方。" if state.get("analysis_only")
                      else "候选方剂未通过确定性校验，具体药材清单已隐藏。请由执业医师人工复核。"])
        lines.extend(f"- {error}" for error in display_errors)
    else:
        lines.extend(["", "## 方剂知识示例", render_plan(plan)])
        lines.append("配伍规则通过不等于剂量已核验，未核验项须人工复核；此为知识演示，不能作为个人处方。")
    if state.get("analysis_questions"):
        lines.extend(["", "## 仍待补充"])
        lines.extend("- " + question for question in state["analysis_questions"] if not contains_dosing(question))
    lines.extend(["", "> 仅供学习与研究参考，不能代替执业医师面诊，也不应作为自行抓药的依据。"])
    notes = final_consultation_notes(state)
    response = "\n".join(lines)
    # prescription_advice is constructed above from the reviewed structured plan,
    # never from model prose. Other note fields must remain dose-free.
    prose_notes = [{key: value for key, value in note.items() if key != "prescription_advice"} for note in notes]
    if contains_dosing(json.dumps(prose_notes, ensure_ascii=False)):
        return {"response": "答复检查未通过，本轮不展示具体治疗内容，请由医师核对。",
                "consultation_notes": [], "safety_errors": ["最终答复包含剂量或用法"]}
    return {"response": response, "consultation_notes": notes,
            "safety_errors": state.get("safety_errors", [])}



def instrument(name: str, function):
    async def execute(state):
        context = invocation.get()
        if context:
            context.check()
        writer = get_stream_writer()
        started = time.monotonic()
        node_token = current_node.set(name)
        writer({"type": "stage", "stage": name, "label": LABELS.get(name, name), "status": "start"})

        try:
            if context and context.emit:
                if name == "revise_formula" or (name == "formula" and state.get("safety_errors")):
                    context.emit("candidate.started", {"text": state.get("diagnosis", ""), "revision": True})
                if name == "reviewer":
                    context.emit("candidate.completed", {"text": state.get("diagnosis", "") + "\n\n## 候选方剂\n" + render_plan(state["treatment_plan"]),
                                                         "plan": state["treatment_plan"]})
                    context.emit("review.started", {})
            if inspect.iscoroutinefunction(function):
                result = await function(state)
            elif name == "search":
                # Retrieval is read-only. Cancelling detaches its worker; it cannot dispatch nodes or write portraits.
                result = await asyncio.to_thread(function, state)
            else:
                result = function(state)

            if context:
                context.check()
                if context.emit and name == "safety":
                    context.emit("review.approved" if result["prescription_review"]["approved"] else "review.rejected",
                                 {"review": result["prescription_review"], "plan": result["treatment_plan"]})
            writer({"type": "stage", "stage": name, "label": LABELS.get(name, name), "status": "done",
                    "elapsed_ms": round((time.monotonic() - started) * 1000)})
            return result
        except BaseException as exc:
            writer({"type": "stage", "stage": name, "label": LABELS.get(name, name),
                    "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    "elapsed_ms": round((time.monotonic() - started) * 1000)})
            raise
        finally:
            current_node.reset(node_token)
    return execute


def build_graph(mode: str):
    from .knowledge import search_knowledge, answer_knowledge
    graph = StateGraph(FlowState)
    graph.add_node("prepare", instrument("prepare", prepare))
    graph.add_node("document_parse", instrument("document_parse", document_parse))
    graph.add_node("profile_load", instrument("profile_load", profile_load))
    graph.add_node("subject_resolve", instrument("subject_resolve", subject_resolve))
    graph.add_node("episode_resolve", instrument("episode_resolve", episodes.resolve))
    graph.add_node("pre_safety", instrument("pre_safety", pre_safety))
    graph.add_node("inquiry", instrument("inquiry", inquire))
    graph.add_node("main_route", instrument("main_route", evidence_ready))
    graph.add_node("evidence_ready", instrument("evidence_ready", evidence_ready))
    graph.add_node("context_budget", instrument("context_budget", context_budget))
    graph.add_node("episode_write", instrument("episode_write", episode_write))
    graph.add_node("finish", instrument("finish", finish))
    graph.add_node("knowledge_search", instrument("knowledge_search", search_knowledge))
    graph.add_node("knowledge_answer", instrument("knowledge_answer", answer_knowledge))
    graph.add_edge("knowledge_search", "knowledge_answer")
    graph.add_edge("knowledge_answer", "finish")
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "document_parse")
    graph.add_edge("document_parse", "subject_resolve")
    graph.add_edge("subject_resolve", "episode_resolve")
    graph.add_edge("episode_resolve", "profile_load")
    graph.add_edge("profile_load", "pre_safety")
    graph.add_conditional_edges("pre_safety", pre_safety_route,
                                {"blocked": "context_budget", "allowed": "inquiry"})
    graph.add_edge("inquiry", "main_route")
    graph.add_node("safety", instrument("safety", safety))
    if mode == "multi":
        from .agents.multi import (bianzheng_agent, bianzheng_route, formula_agent, formula_route,
                                   herb_agent, review_agents, synthesize_agents)
        graph.add_node("bianzheng_agent", instrument("bianzheng", bianzheng_agent))
        graph.add_node("formula_agent", instrument("formula", formula_agent))
        graph.add_node("herb_agent", instrument("herb", herb_agent))
        graph.add_node("synthesis", instrument("synthesis", synthesize_agents))
        graph.add_node("multi_reviewer", instrument("reviewer", review_agents))
        graph.add_conditional_edges("main_route", main_route,
                                    {"diagnose": "evidence_ready", "knowledge": "knowledge_search", "end": "episode_write"})
        graph.add_edge("evidence_ready", "bianzheng_agent")
        graph.add_conditional_edges("bianzheng_agent", bianzheng_route,
                                    {"continue": "formula_agent", "stop": "context_budget"})
        graph.add_conditional_edges("formula_agent", formula_route,
                                    {"continue": "herb_agent", "stop": "context_budget"})
        graph.add_edge("herb_agent", "synthesis")
        graph.add_edge("synthesis", "multi_reviewer")
        graph.add_edge("multi_reviewer", "safety")
        graph.add_conditional_edges("safety", safety_route,
                                    {"revise": "formula_agent", "finish": "context_budget"})
    else:
        graph.add_node("rag_decision", instrument("rag_decision", rag_decision))
        graph.add_node("search", instrument("search", search))
        graph.add_node("evaluate_retrieval", instrument("evaluate_retrieval", evaluate_retrieval))
        graph.add_node("rewrite_query", instrument("rewrite_query", rewrite_query))
        graph.add_node("bianzheng", instrument("bianzheng", bianzheng))
        graph.add_node("formula", instrument("formula", formula))
        graph.add_node("revise_formula", instrument("revise_formula", revise_formula))
        graph.add_node("reviewer", instrument("reviewer", reviewer))
        graph.add_conditional_edges("main_route", main_route,
                                    {"diagnose": "rag_decision", "knowledge": "knowledge_search", "end": "episode_write"})
        graph.add_conditional_edges("rag_decision", rag_route,
                                    {"search": "search", "skip": "evidence_ready"})
        graph.add_edge("search", "evaluate_retrieval")
        graph.add_conditional_edges("evaluate_retrieval", retrieval_route,
                                    {"rewrite": "rewrite_query", "ready": "evidence_ready"})
        graph.add_edge("rewrite_query", "search")
        graph.add_edge("evidence_ready", "bianzheng")
        graph.add_edge("bianzheng", "formula")
        graph.add_edge("formula", "reviewer")
        graph.add_edge("reviewer", "safety")
        graph.add_edge("revise_formula", "reviewer")
        graph.add_conditional_edges("safety", safety_route,
                                    {"revise": "revise_formula", "finish": "context_budget"})
    graph.add_edge("context_budget", "episode_write")
    graph.add_edge("episode_write", "finish")
    graph.add_edge("finish", END)
    return graph


LABELS = {
    "episode_resolve": "确认本次病程",
    "knowledge_search": "知识检索", "knowledge_answer": "知识解答",
    "prepare": "资料准备", "document_parse": "文档解析", "subject_resolve": "确认问诊对象", "profile_load": "加载本人画像",
    "pre_safety": "前置安全审查", "inquiry": "问诊与意图识别", "main_route": "主路由",
    "rag_decision": "检索决策", "search": "古籍检索", "evaluate_retrieval": "检索评估",
    "rewrite_query": "查询重写", "evidence_ready": "证据整理", "bianzheng": "辨证",
    "formula": "方剂建议", "revise_formula": "方剂调整", "herb": "本草",
    "synthesis": "会诊综合", "reviewer": "独立审查", "safety": "处方安全校验",
    "context_budget": "上下文预算",
    "episode_write": "准备会话摘要", "finish": "最终答复检查",
    "bianzheng_agent": "辨证 Agent", "formula_agent": "方剂 Agent", "herb_agent": "本草 Agent",
}


async def run_flow(thread_id: str, mode: str, message: str, profile: dict, file_text: str, user_id: str, run_id: str):
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        graph = build_graph(mode).compile(checkpointer=saver)
        previous = one("SELECT id FROM chat_runs WHERE thread_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1", (thread_id,))
        prior = (await graph.aget_state({"configurable": {"thread_id": previous["id"]}})).values if previous else {}
        # Import legacy thread checkpoints only when there is no completed run.
        if not previous:
            prior = (await graph.aget_state({"configurable": {"thread_id": thread_id}})).values
        # Import only current-thread checkpoint fields declared by this workflow.
        # Retired cross-session summary channels cannot survive legacy checkpoints.
        prior = {key: value for key, value in prior.items() if key in FlowState.__annotations__}
        state = {**prior, "message": message, "profile": profile, "file_text": model_excerpt(file_text),
                 "thread_id": thread_id, "user_id": user_id, "run_id": run_id}
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
        attempts = {}
        async for kind, data in graph.astream(state, config=config, stream_mode=["custom", "updates"]):
            if kind == "custom":
                if data["type"] != "stage":
                    yield data
                    continue
                name = data["stage"]
                if data["status"] == "start":
                    attempts[name] = attempts.get(name, 0) + 1
                yield {**data, "attempt": attempts.get(name, 1)}
        final = (await graph.aget_state(config)).values
        yield {"type": "result", "data": final}
