from __future__ import annotations

import json
import re
import asyncio
import inspect
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer

from .config import CHECKPOINT_PATH, MODEL_MODE
from .db import all_rows, one
from .model import acomplete, acomplete_json, CandidatePlan, ReviewResult, invocation, current_node
from .preflight import assess
from .prescription import migrate_plan, legacy_plan, names, review_plan, fingerprint, patient_fingerprint, render_plan, knowledge_context
from .retrieval import retrieve_with_status
from .safety import check_herbs, mentioned_herbs, normalize, contains_dosing


class FlowState(TypedDict, total=False):
    user_id: str
    preflight: dict
    prescription_review: dict
    reviewed_hash: str
    reviewer_hash: str
    manual_profile_hash: str
    thread_id: str
    message: str
    file_text: str
    profile: dict
    memory_context: str
    episode_text: str
    context_summary: str
    context_chars: int
    conversation_turns: int
    compression_required: bool
    symptoms: list[str]
    denied_symptoms: list[str]
    intent_type: str
    pending_question: str
    pending_context: str
    docs: list[dict]
    search_query: str
    search_attempts: int
    retrieval_score: float
    search_required: bool
    retrieval_warning: str
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


SYMPTOM_ALIASES = {
    "失眠": ("失眠", "睡不好"), "睡眠": ("睡眠",), "咳嗽": ("咳嗽",),
    "头痛": ("头痛",), "腹胀": ("腹胀", "肚子胀", "肚胀", "胃胀"),
    "腹痛": ("腹痛", "肚子痛", "肚子疼"), "胃痛": ("胃痛",),
    "大便稀": ("大便稀", "便稀", "稀便", "便溏"),
    "怕冷": ("怕冷",), "发热": ("发热",), "疲劳": ("疲劳",),
    "乏力": ("乏力",), "口干": ("口干", "嘴干"), "口苦": ("口苦",),
    "舌苔白": ("舌苔发白", "舌苔白", "白苔"),
    "月经": ("月经",), "眩晕": ("眩晕",), "鼻塞": ("鼻塞",), "食欲": ("食欲",),
}
URGENT = ["胸痛", "呼吸困难", "昏厥", "大出血", "意识不清", "自杀"]
MAX_SEARCH_ATTEMPTS = 3
MAX_SAFETY_REVISIONS = 2
CONTEXT_CHAR_LIMIT = 6000
MAX_CONVERSATION_TURNS = 10
CLASSIC_TERMS = {
    "失眠": "不寐 不得眠", "睡眠": "不寐", "咳嗽": "咳 嗽", "疲劳": "倦怠 乏力",
    "乏力": "倦怠", "怕冷": "恶寒", "发热": "发热 身热", "头痛": "头痛 头项强痛",
    "腹痛": "腹痛", "胃痛": "心下痛", "眩晕": "眩晕 头眩",
}


def prepare(state: FlowState) -> FlowState:
    # Clear all per-turn fields, including prior expert drafts.
    return {
        "docs": [], "retrieval_warning": "", "diagnosis": "", "treatment_plan": {}, "active_plan": {},
        "bianzheng_note": "", "formula_note": "", "herb_note": "",
        "consultation_notes": [], "safety_errors": [], "safety_findings": [], "response": "",
        "prescription_review": {}, "reviewed_hash": "", "reviewer_hash": "", "reviewer_approved": True, "reviewer_retry_count": 0, "safety_retry_count": 0,
        "memory_context": "", "episode_text": "", "context_summary": "",
        "context_chars": 0, "compression_required": False,
        "conversation_turns": 0,
        "search_query": "", "search_attempts": 0, "retrieval_score": 0.0,
        "search_required": False, "intent_type": "", "symptoms": [], "denied_symptoms": [],
    }


def document_parse(state: FlowState) -> FlowState:
    # PDF/DOCX/TXT parsing and file ownership checks happen at upload/API time.
    # Normalize the extracted text before it enters a model context.
    return {"file_text": re.sub(r"\s+", " ", state.get("file_text", "")).strip()[:4000]}


def memory_load(state: FlowState) -> FlowState:
    user_id, thread_id = state["user_id"], state["thread_id"]
    profile = one("SELECT constitution,allergies,history FROM profiles WHERE user_id=?", (user_id,)) or state.get("profile", {})
    memories = all_rows(
        "SELECT summary FROM profile_memories WHERE user_id=? ORDER BY updated_at DESC LIMIT 20",
        (user_id,),
    )
    recent = all_rows(
        "SELECT content FROM messages WHERE thread_id=? AND role='user' ORDER BY created_at DESC,id DESC LIMIT 5",
        (thread_id,),
    )
    prior = [row["content"][:180] for row in recent]
    if prior and prior[0] == state.get("message", ""):
        prior.pop(0)
    query = state.get("message", "") + state.get("file_text", "")[:300]
    def terms(value: str) -> set[str]:
        chars = re.findall(r"[\u4e00-\u9fff]", value)
        return {chars[index] + chars[index + 1] for index in range(len(chars) - 1)}
    relevant = terms(query)
    ranked = sorted(enumerate(memories), key=lambda item: (-len(relevant & terms(item[1]["summary"])), item[0]))
    parts = [item["summary"] for _, item in ranked[:3]] + list(reversed(prior))
    count = one("SELECT COUNT(*) AS total FROM messages WHERE thread_id=? AND role='user'", (thread_id,))
    saved_plan = one(
        "SELECT plan FROM messages WHERE thread_id=? AND role='assistant' AND plan IS NOT NULL "
        "ORDER BY created_at DESC,id DESC LIMIT 1", (thread_id,),
    )
    try:
        active_plan = json.loads(saved_plan["plan"]) if saved_plan else {}
    except (TypeError, ValueError):
        active_plan = {}
    if not active_plan:
        # Older conversations stored the visible example but not its structure.
        legacy = one(
            "SELECT content FROM messages WHERE thread_id=? AND role='assistant' "
            "AND content LIKE '%## 方剂知识示例%' ORDER BY created_at DESC,id DESC LIMIT 1",
            (thread_id,),
        )
        active_plan = legacy_plan(legacy["content"] if legacy else "") or {}
    return {"profile": profile, "manual_profile_hash": patient_fingerprint(profile), "memory_context": "\n".join(parts)[:1200],
            "conversation_turns": count["total"] if count else 0,
            "active_plan": migrate_plan(active_plan) if isinstance(active_plan, dict) else {}}


def rule_pre_safety(state: FlowState) -> FlowState:
    text = state.get("message", "")
    attached = state.get("file_text", "")
    if any(word in text or word in attached for word in URGENT):
        return {
            "intent_type": "safety_rejected", "symptoms": [],
            "pending_context": "", "pending_question": "", "episode_text": text,
            "response": "你提供的信息中出现可能需要及时处理的情况。若症状正在发生，请尽快联系当地急救服务或就近医疗机构，不要等待在线问答。",
        }
    if re.search(r"加(?:入|上)?|添加|配伍|搭配|合用|同用|一起(?:用|服)|能(?:用|服)|可以用|相反|相克|冲突", text):
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
                        "symptoms": [], "pending_context": "", "pending_question": "",
                        "episode_text": text, "safety_errors": errors, "response": response}
    return {"intent_type": ""}


async def pre_safety(state: FlowState) -> FlowState:
    # Current explicit allergy statements apply to this round without mutating the portrait.
    text = state.get("message", "") + "；" + state.get("file_text", "")
    profile = dict(state.get("profile", {}))
    allergies = []
    for clause in re.split(r"[，,。；;\n]", text):
        if "过敏" in clause and not re.search(r"没有|不过敏|无过敏|否认", clause):
            allergies.extend(mentioned_herbs(clause))
    if allergies:
        profile["allergies"] = str(profile.get("allergies", "")) + "；" + "、".join(allergies)
    state = {**state, "profile": profile}
    rules = rule_pre_safety(state)
    if rules.get("intent_type") in {"safety_rejected", "safety_advisory"}:
        return {**rules, "profile": profile, "preflight": {
            "decision": "block" if rules["intent_type"] == "safety_rejected" else "clarify",
            "risk_categories": ["deterministic"], "reason": "确定性规则终止治疗路径",
            "required_questions": []}}
    try:
        result = await assess(text, profile, mock=MODEL_MODE == "mock")
    except Exception:
        result = {"decision": "block", "risk_categories": ["preflight_unavailable"],
                  "reason": "预审超时、失败或结构无法校验", "required_questions": []}
    if result["decision"] != "allow":
        # Never publish model-generated safety text, which may contain an unsafe instruction.
        response = ("安全预审未通过，本轮不生成具体治疗建议。请由医师核对风险资料。"
                    if result["decision"] == "block" else
                    "需要先补充安全资料，本轮不生成方剂或治疗建议。请说明年龄、孕哺状态、过敏史和当前全部用药。")
        return {"profile": profile, "preflight": result, "intent_type": "safety_rejected" if result["decision"] == "block" else "safety_advisory",
                "pending_context": "", "pending_question": "", "episode_text": state.get("message", ""),
                "response": response}
    return {"profile": profile, "preflight": result, "intent_type": ""}


def pre_safety_route(state: FlowState) -> str:
    return "blocked" if state.get("intent_type") in {"safety_rejected", "safety_advisory"} else "allowed"


def symptom_observations(text: str) -> tuple[list[str], list[str]]:
    """Separate self-reported symptoms from explicitly denied symptoms."""
    reported: set[str] = set()
    denied: set[str] = set()
    for clause in re.split(r"[，,。；;！？!?\n]|但是|不过|但", text):
        if not clause.strip():
            continue
        negated_clause = bool(re.search(r"没有|无|否认|不伴|未见|并无", clause))
        for symptom, aliases in SYMPTOM_ALIASES.items():
            if any(alias in clause for alias in aliases):
                explicitly_negated = negated_clause or any(f"不{alias}" in clause for alias in aliases)
                (denied if explicitly_negated else reported).add(symptom)
    reported.difference_update(denied)
    return ([name for name in SYMPTOM_ALIASES if name in reported],
            [name for name in SYMPTOM_ALIASES if name in denied])


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
    previous = state.get("pending_context", "")
    current = text + ("；附件：" + state["file_text"][:800] if state.get("file_text") else "")
    combined = f"{previous}；{current}" if previous else current
    earlier_positive, earlier_denied = symptom_observations(previous)
    current_positive, current_denied = symptom_observations(current)
    positive = (set(earlier_positive) - set(earlier_denied) - set(current_denied)) | set(current_positive)
    denied = (set(earlier_denied) | set(current_denied)) - set(current_positive)
    symptoms = [name for name in SYMPTOM_ALIASES if name in positive]
    denied_symptoms = [name for name in SYMPTOM_ALIASES if name in denied]
    if not symptoms and not previous:
        return {
            "intent_type": "chat", "symptoms": [], "denied_symptoms": denied_symptoms, "pending_context": "",
            "pending_question": "",
            "episode_text": combined,
            "response": "你好，我可以帮助整理症状信息，并展示中医知识检索与会诊流程。请描述主要不适、持续时间，以及有无过敏史。内容仅供学习研究参考。",
        }
    if not previous and len(combined.strip()) < 18 and not re.search(r"\d|天|周|月|年|伴|同时", combined):
        question = followup_question(symptoms, denied_symptoms)
        return {
            "intent_type": "consult", "symptoms": symptoms, "denied_symptoms": denied_symptoms,
            "episode_text": combined,
            "pending_context": combined, "pending_question": question,
            "response": "为了更好地整理信息，请补充：" + question,
        }
    return {
        "intent_type": "diagnose", "symptoms": symptoms,
        "denied_symptoms": denied_symptoms,
        "episode_text": combined,
        "pending_context": "", "pending_question": "",
    }


def main_route(state: FlowState) -> str:
    return "diagnose" if state["intent_type"] == "diagnose" else "end"


def rag_decision(state: FlowState) -> FlowState:
    explicit_skip = any(term in state.get("message", "") for term in ("不用查古籍", "不需要查文献", "无需检索"))
    required = not explicit_skip and bool(state.get("symptoms") or re.search(r"古籍|文献|出处|依据", state.get("message", "")))
    return {"search_required": required}


def rag_route(state: FlowState) -> str:
    return "search" if state.get("search_required") else "skip"


def search(state: FlowState) -> FlowState:
    # Negative findings remain in the model context, but must not become search terms.
    query = state.get("search_query") or (" ".join(state.get("symptoms", [])) or state["message"])
    docs, warning = retrieve_with_status(query)
    return {"docs": docs, "retrieval_warning": warning, "search_query": query,
            "search_attempts": state.get("search_attempts", 0) + 1}


def evaluate_retrieval(state: FlowState) -> FlowState:
    docs = state.get("docs", [])
    symptoms = state.get("symptoms", [])
    if not docs:
        score = 0.0
    elif not symptoms:
        score = 0.5
    else:
        contents = "\n".join(doc["body"] for doc in docs)
        matches = sum(any(term in contents for term in (symptom, *CLASSIC_TERMS.get(symptom, "").split()))
                      for symptom in symptoms)
        score = matches / len(symptoms)
    warning = state.get("retrieval_warning", "")
    if score < 0.5 and state.get("search_attempts", 0) >= MAX_SEARCH_ATTEMPTS:
        warning = (warning + " " if warning else "") + "检索结果相关性不足，未引用古籍片段。"
        docs = []
    return {"retrieval_score": score, "docs": docs, "retrieval_warning": warning}


def retrieval_route(state: FlowState) -> str:
    if state.get("retrieval_score", 0.0) < 0.5 and state.get("search_attempts", 0) < MAX_SEARCH_ATTEMPTS:
        return "rewrite"
    return "ready"


def rewrite_query(state: FlowState) -> FlowState:
    terms = state.get("symptoms", [])
    if state.get("search_attempts") == 1:
        query = " ".join(terms + [CLASSIC_TERMS.get(term, "") for term in terms])
    else:
        query = " ".join(CLASSIC_TERMS.get(term, term) for term in terms)
        if not query.strip():
            query = state.get("episode_text", state.get("message", ""))[:100]
    return {"search_query": query.strip()}


def evidence_ready(_state: FlowState) -> FlowState:
    return {}


def _context(state: FlowState) -> str:
    docs = "\n".join(f'{d["title"]}：{d["body"][:240]}' for d in state.get("docs", []))
    return (
        f'用户描述：{state.get("message","")}\n'
        f'本轮问诊整合：{state.get("episode_text", "")[:600]}\n'
        f'症状：{",".join(state.get("symptoms",[]))}\n'
        f'用户明确否认的症状：{",".join(state.get("denied_symptoms",[])) or "无"}\n'
        f'病历摘要：{state.get("file_text","")[:1600]}\n'
        f'患者画像：{state.get("profile",{})}\n'
        f'过往自述摘要：{state.get("memory_context", "")[:900]}\n'
        f'检索片段：{docs or "无可用古籍片段"}'
    )


async def bianzheng(state: FlowState) -> FlowState:
    if MODEL_MODE == "mock":
        symptoms = "、".join(state.get("symptoms", [])) or "所述不适"
        denied = "、".join(state.get("denied_symptoms", []))
        diagnosis = (f"根据已提供信息，用户描述主要涉及{symptoms}。"
                     + (f"用户表示没有{denied}。" if denied else "")
                     + "信息不足以确定证型，以下仅展示分析流程。")
    else:
        diagnosis = await acomplete(
            "你是中医知识辅助系统。区分用户报告的症状与明确否认的症状；后者不能作为阳性依据。"
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
        plan = await acomplete_json(
            "你是中医知识演示系统。返回候选方剂 JSON，按 Schema 返回 name 和结构化 herbs。每味药包含 herb_id、name、dosage、preparation、source_ids、verification_status。剂量只能依据提供的知识记录，缺少依据时 min/max 为空；所有 verification_status 均为 unverified，不要自行声称核验。禁止自行抓药指令。",
            _context(state) + "\n辨证思路：" + state.get("diagnosis", "") + retry_context + "\n可用剂量出处：" + knowledge_context(), role="formula", schema=CandidatePlan,
        )
        if not isinstance(plan.get("name"), str) or not isinstance(plan.get("herbs"), list):
            plan = {"name": "未能解析候选方剂", "herbs": []}
    plan = migrate_plan(plan)
    for herb in plan.get("herbs", []):
        herb["verification_status"] = "unverified"
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
    review = {**review, "approved": not errors, "errors": errors}
    findings = list(dict.fromkeys(state.get("safety_findings", []) + errors))
    return {"treatment_plan": plan, "prescription_review": review,
            "reviewed_hash": fingerprint(plan) if not errors else "",
            "safety_errors": errors, "safety_findings": findings}


def safety_route(state: FlowState) -> str:
    if state.get("safety_errors") and state.get("safety_retry_count", 0) < MAX_SAFETY_REVISIONS:
        return "revise"
    return "finish"


def context_budget(state: FlowState) -> FlowState:
    context_chars = len(_context(state)) + len(state.get("memory_context", ""))
    return {"context_chars": context_chars,
            "compression_required": context_chars > CONTEXT_CHAR_LIMIT
            or state.get("conversation_turns", 0) > MAX_CONVERSATION_TURNS}


def compression_route(state: FlowState) -> str:
    return "compress" if state.get("compression_required") else "write"


def compress_context(state: FlowState) -> FlowState:
    # Preserve the current statement and manually entered profile; reduce only
    # supporting context that may grow across turns.
    summary = "；".join(part for part in (
        "症状：" + "、".join(state.get("symptoms", [])) if state.get("symptoms") else "",
        "近期自述：" + state.get("episode_text", state.get("message", ""))[:400],
        "病历摘录：" + state.get("file_text", "")[:400] if state.get("file_text") else "",
    ) if part)[:1000]
    return {"context_summary": summary, "memory_context": state.get("memory_context", "")[-500:],
            "file_text": state.get("file_text", "")[:800]}


def profile_write(state: FlowState) -> FlowState:
    if state.get("intent_type") not in {"diagnose", "consult", "safety_rejected", "safety_advisory"}:
        return {}
    summary = state.get("context_summary") or "；".join(part for part in (
        "症状：" + "、".join(state.get("symptoms", [])) if state.get("symptoms") else "",
        "用户自述：" + state.get("episode_text", state.get("message", ""))[:450],
    ) if part)[:1000]
    return {"context_summary": summary}


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
    return notes


def finish(state: FlowState) -> FlowState:
    if state.get("intent_type") != "diagnose":
        return {"response": state.get("response", "")}
    is_consultation = bool(state.get("consultation_notes"))
    diagnosis = state.get("diagnosis", "资料不足。")
    if contains_dosing(diagnosis):
        diagnosis = "生成内容包含不允许展示的用药指令，本轮仅保留受限答复，请由医师核对。"
        state = {**state, "diagnosis": diagnosis, "safety_errors": ["答复检查未通过：包含剂量或用法"]}
    lines = ["## 会诊综合" if is_consultation else "## 辨证思路", diagnosis]
    docs = state.get("docs", [])
    if state.get("retrieval_warning"):
        lines.extend(["", state["retrieval_warning"]])
    if docs:
        lines.extend(["", "## 检索出处"])
        lines.extend(f'- 《{d["title"]}》{d["chapter"]}：'
                     + ("原文涉及古籍用药内容，摘录已省略。" if contains_dosing(d["body"][:110])
                        else d["body"][:110] + "…") for d in docs)
    else:
        lines.extend(["", "本轮按要求未检索古籍。" if not state.get("search_required") else "当前没有可引用的古籍片段。"])
    errors = state.get("safety_errors", [])
    display_errors = (state.get("safety_findings", []) or errors) if errors else []
    plan = state.get("treatment_plan", {})
    if errors:
        lines.extend(["", "## 安全校验", "候选方剂未通过确定性校验，具体药材清单已隐藏。请由执业医师人工复核。"])
        lines.extend(f"- {error}" for error in display_errors)
    else:
        lines.extend(["", "## 方剂知识示例", render_plan(plan)])
        lines.append("配伍规则通过不等于剂量已核验，未核验项须人工复核；此为知识演示，不能作为个人处方。")
    lines.extend(["", "> 仅供学习与研究参考，不能代替执业医师面诊，也不应作为自行抓药的依据。"])
    notes = final_consultation_notes(state)
    response = "\n".join(lines)
    if contains_dosing(json.dumps(notes, ensure_ascii=False)):
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
                if name == "revise_formula":
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
    graph = StateGraph(FlowState)
    graph.add_node("prepare", instrument("prepare", prepare))
    graph.add_node("document_parse", instrument("document_parse", document_parse))
    graph.add_node("memory_load", instrument("memory_load", memory_load))
    graph.add_node("pre_safety", instrument("pre_safety", pre_safety))
    graph.add_node("inquiry", instrument("inquiry", inquire))
    graph.add_node("main_route", instrument("main_route", evidence_ready))
    graph.add_node("rag_decision", instrument("rag_decision", rag_decision))
    graph.add_node("search", instrument("search", search))
    graph.add_node("evaluate_retrieval", instrument("evaluate_retrieval", evaluate_retrieval))
    graph.add_node("rewrite_query", instrument("rewrite_query", rewrite_query))
    graph.add_node("evidence_ready", instrument("evidence_ready", evidence_ready))
    graph.add_node("context_budget", instrument("context_budget", context_budget))
    graph.add_node("compress_context", instrument("compress_context", compress_context))
    graph.add_node("profile_write", instrument("profile_write", profile_write))
    graph.add_node("finish", instrument("finish", finish))
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "document_parse")
    graph.add_edge("document_parse", "memory_load")
    graph.add_edge("memory_load", "pre_safety")
    graph.add_conditional_edges("pre_safety", pre_safety_route,
                                {"blocked": "context_budget", "allowed": "inquiry"})
    graph.add_edge("inquiry", "main_route")
    graph.add_conditional_edges("main_route", main_route,
                                {"diagnose": "rag_decision", "end": "finish"})
    graph.add_conditional_edges("rag_decision", rag_route,
                                {"search": "search", "skip": "evidence_ready"})
    graph.add_edge("search", "evaluate_retrieval")
    graph.add_conditional_edges("evaluate_retrieval", retrieval_route,
                                {"rewrite": "rewrite_query", "ready": "evidence_ready"})
    graph.add_edge("rewrite_query", "search")
    graph.add_node("bianzheng", instrument("bianzheng", bianzheng))
    graph.add_node("formula", instrument("formula", formula))
    graph.add_node("revise_formula", instrument("revise_formula", revise_formula))
    graph.add_node("safety", instrument("safety", safety))
    graph.add_node("reviewer", instrument("reviewer", reviewer))
    if mode == "multi":
        graph.add_node("herb", instrument("herb", herb_expert))
        graph.add_node("synthesis", instrument("synthesis", synthesize))
        graph.add_edge("evidence_ready", "bianzheng")
        graph.add_edge("evidence_ready", "formula")
        graph.add_edge("evidence_ready", "herb")
        graph.add_edge(["bianzheng", "formula", "herb"], "synthesis")
        graph.add_edge("synthesis", "reviewer")
        graph.add_edge("reviewer", "safety")
        graph.add_edge("revise_formula", "reviewer")
    else:
        graph.add_edge("evidence_ready", "bianzheng")
        graph.add_edge("bianzheng", "formula")
        graph.add_edge("formula", "reviewer")
        graph.add_edge("reviewer", "safety")
        graph.add_edge("revise_formula", "reviewer")
    graph.add_conditional_edges("safety", safety_route,
                                {"revise": "revise_formula", "finish": "context_budget"})
    graph.add_conditional_edges("context_budget", compression_route,
                                {"compress": "compress_context", "write": "profile_write"})
    graph.add_edge("compress_context", "profile_write")
    graph.add_edge("profile_write", "finish")
    graph.add_edge("finish", END)
    return graph


LABELS = {
    "prepare": "资料准备", "document_parse": "文档解析", "memory_load": "记忆与画像",
    "pre_safety": "前置安全审查", "inquiry": "问诊与意图识别", "main_route": "主路由",
    "rag_decision": "检索决策", "search": "古籍检索", "evaluate_retrieval": "检索评估",
    "rewrite_query": "查询重写", "evidence_ready": "证据整理", "bianzheng": "辨证",
    "formula": "方剂建议", "revise_formula": "方剂调整", "herb": "本草",
    "synthesis": "会诊综合", "reviewer": "独立审查", "safety": "处方安全校验",
    "context_budget": "上下文预算", "compress_context": "上下文压缩",
    "profile_write": "准备会话摘要", "finish": "最终答复检查",
}


async def run_flow(thread_id: str, mode: str, message: str, profile: dict, file_text: str, user_id: str, run_id: str):
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        graph = build_graph(mode).compile(checkpointer=saver)
        previous = one("SELECT id FROM chat_runs WHERE thread_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1", (thread_id,))
        prior = (await graph.aget_state({"configurable": {"thread_id": previous["id"]}})).values if previous else {}
        # Import legacy thread checkpoints only when there is no completed run.
        if not previous:
            prior = (await graph.aget_state({"configurable": {"thread_id": thread_id}})).values
        state = {**prior, "message": message, "profile": profile, "file_text": file_text,
                 "thread_id": thread_id, "user_id": user_id}
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
