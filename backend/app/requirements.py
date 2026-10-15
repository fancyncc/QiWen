"""Read existing scoped consultation requirements without calling a provider."""
import json
import re

from .db import all_rows
from .safety import contains_dosing
from .safety_context import applicable_facts
from . import symptoms


def is_requirements_query(state: dict) -> bool:
    text = state.get("message", "").strip()
    if state.get("file_text") or state.get("identity_current"):
        return False
    if symptoms.extract(text, subject=state["safety_state"]["subject"]) or contains_dosing(text):
        return False
    return bool(re.search(r"(?:还|仍|再)?(?:需要|需|缺|要).{0,8}(?:补充|提供|确认).{0,6}(?:什么|哪些|哪几)|"
                          r"(?:还|仍)?缺(?:少)?(?:什么|哪些|哪几).{0,8}(?:资料|信息|内容|条件)|"
                          r"(?:补充|提供)(?:什么|哪些|哪几).{0,8}(?:资料|信息|内容|才能|才可以)", text))


def requirements_reply(state: dict, gate: dict) -> str:
    safe = state["safety_state"]
    current, _, _ = symptoms.view(state.get("symptom_facts", []), safe["subject"])
    lines = ["## 已有资料"]
    if current:
        lines.append("已记录的本次症状：" + "、".join(current) + "。")
    allergens = list(dict.fromkeys(f["entity"] for f in applicable_facts(safe)
                                  if f["category"] == "allergy" and f["assertion"] == "present"))
    if allergens and not contains_dosing("、".join(allergens)):
        lines.append("已记录过敏史：" + "、".join(allergens) + "。")
    lines.append("已提供的内容无需重复；有变化时再说明。")
    lines.append("\n## 需要补充什么")
    questions = list(gate.get("required_questions", []))
    if not questions and state.get("pending_question"):
        questions = [state["pending_question"]]
    questions = [question for question in questions if not contains_dosing(question)][:3]
    if questions:
        lines.extend("- " + question for question in questions)
    else:
        lines.append("当前记录没有保存新的必答问题。不能仅凭“未形成处方”，就认定你还缺少病情资料。")
    rows = all_rows("SELECT trace,prescription_review FROM messages WHERE thread_id=? AND role='assistant' ORDER BY created_at DESC,id DESC LIMIT 12",
                    (state["thread_id"],))
    same_scope = []
    for row in rows:
        trace = json.loads(row["trace"] or "{}")
        if (trace and trace.get("identity_scope") == state.get("identity_scope")
                and trace.get("preflight", {}).get("scope") != "requirements_only"):
            same_scope.append((trace, json.loads(row["prescription_review"] or "{}")))
    lines.append("\n## 当前流程限制")
    if same_scope and "preflight_unavailable" in same_scope[0][0].get("preflight", {}).get("risk_categories", []):
        lines.append("最近一轮停在模型服务不可用，安全预审没有完成。这属于系统调用故障，继续补充病情不能解决；服务恢复后需重新完成预审。")
    if any(review.get("dosage_status") in {"unverified", "rejected"} for _, review in same_scope):
        lines.append("既往候选的药材或剂量依据仍有未核验项，需要补足知识依据并复核；这不是要求你重复回答症状或过敏史。")
    lines.append("补齐病情资料不保证形成个人处方；文献、药材、剂量和个人适用性仍需分别核验。")
    lines.append("\n本轮仅查询当前会话的已有记录和待答问题，未执行新的辨证或处方生成，也不代表安全预审已通过。")
    return "\n\n".join(lines)
