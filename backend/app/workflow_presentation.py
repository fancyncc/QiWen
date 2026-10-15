"""Public workflow state built from persisted intent, events and validation results."""
from __future__ import annotations

import re

from .safety import contains_dosing
from .prescription import render_plan
from .prescription_reference import reference_prescription, render_reference
from .db import one
from .safety_context import applicable_facts
from .clinical_output import selection_advice


FIXED_STEPS = [("inquiry", "问诊"), ("search", "古籍检索（按需）"), ("bianzheng", "辨证"),
               ("formula", "调理方案"), ("safety", "配伍与用药校验")]
MULTI_STEPS = [("inquiry", "问诊"), ("bianzheng", "辨证"), ("formula", "方剂"), ("herb", "本草"),
               ("reviewer", "独立模型审查"), ("safety", "配伍与用药校验")]


def safe_guidance(values: list[str]) -> str:
    return "\n".join("- " + item for item in values if isinstance(item, str) and item.strip() and not contains_dosing(item))


def fixed_sections(state: dict, *, legacy: bool = False) -> tuple[list[dict], str]:
    diagnosis = state.get("diagnosis", "")
    if legacy:
        match = re.search(r"^## (?:辨证思路|会诊综合|初步辨证[^\n]*)\n([\s\S]*?)(?=^## |\Z)", state.get("response", ""), re.M)
        diagnosis = match[1].strip() if match else "这条历史答复未单独保存辨证分析。"
    if contains_dosing(diagnosis):
        diagnosis = "辨证内容含未核验的用药信息，需由医师复核。"
    summary = (state.get("diagnosis_summary") or "\n\n".join(diagnosis.split("\n\n")[:2])) if legacy else diagnosis
    if contains_dosing(summary):
        summary = "辨证摘要含未核验的用药内容，请由医师复核。"
    if legacy and len(summary) > 240:
        summary = summary[:240] + "…"
    plan, review = state.get("treatment_plan", {}), state.get("prescription_review", {}) or {}
    approved = bool(review.get("approved") and not state.get("safety_errors") and not state.get("analysis_only"))
    medication = (render_plan(plan) + ("\n\n剂量仍有未核验项，以上仅为知识参考，需由执业中医师或药师核对。" if review.get("dosage_status") != "verified"
                                     else "\n\n此为已校验的知识参考，个人适用性仍需执业中医师确认。")
                  if approved and plan.get("herbs") else "当前候选未完成安全校验，具体药材清单已隐藏。本轮不形成可执行的个人处方，药材、剂量和煎服方法需由执业中医师结合面诊资料核对。")
    if state.get("analysis_only") and plan.get("name"):
        entry = one("SELECT name,indication FROM formulas WHERE name=?", (plan["name"],))
        if entry and not contains_dosing(entry["name"] + entry["indication"]):
            medication += f'\n\n方剂知识对照：**{entry["name"]}**。条目记载：{entry["indication"]}。此为文献条目，未确认适用于当前问诊对象。'
    reference = reference_prescription(state)
    if reference:
        medication = selection_advice(state, reference)
    elif not legacy:
        medication = selection_advice(state, None)
    allergens = list(dict.fromkeys(f["entity"] for f in applicable_facts(state["safety_state"])
                                  if f["category"] == "allergy" and f["assertion"] == "present")) if state.get("safety_state") else []
    if allergens:
        medication += "\n\n过敏提示：已记录" + "、".join(allergens) + "过敏，拟用药物风险仍需医师或药师核对。"
    sections = [
        {"id": "diagnosis", "title": "辨证结论", "content": summary or "现有资料尚不足以明确辨证方向。"},
        {"id": "medication", "title": "选方与处方建议", "content": medication},
        {"id": "non_drug", "title": "非药物建议", "content": safe_guidance(state.get("non_drug_advice", [])) or "本轮未生成独立的非药物建议，可在补充问诊后进一步讨论。"},
        {"id": "medical_guidance", "title": "就医指引", "content": safe_guidance(state.get("medical_guidance", [])) or "请携带症状、过敏史及既往诊疗资料，由执业中医师面诊评估。"},
        {"id": "disclaimer", "title": "参考声明", "content": "本内容仅供学习与健康信息参考，不能替代执业医师面诊，请勿自行抓药或调整用药。"},
    ]
    if not legacy and not safe_guidance(state.get("non_drug_advice", [])):
        sections = [item for item in sections if item["id"] != "non_drug"]
    return sections, diagnosis if diagnosis != summary else ""


def build_presentation(state: dict, trace: dict, *, legacy: bool = False) -> dict:
    mode = trace.get("mode", state.get("workflow_mode", "fixed"))
    intent = state.get("intent_type", "")
    preflight = state.get("preflight") or trace.get("preflight", {})
    unavailable = "preflight_unavailable" in preflight.get("risk_categories", [])
    status = ("cancelled" if trace.get("status") == "cancelled" else "failed" if trace.get("status") == "failed"
              else "unavailable" if unavailable
              else "awaiting_input" if intent in {"consult", "safety_advisory"}
              else "blocked" if intent == "safety_rejected" else "answered" if intent in {"chat", "requirements"}
              else "restricted" if state.get("safety_errors") or (state.get("prescription_review") or {}).get("approved") is False
              or (state.get("prescription_review") or {}).get("dosage_status") in {"unverified", "rejected"}
              else "completed")
    latest = {}
    for event in trace.get("events", []):
        latest[event["stage"]] = event
    steps = []
    planned = ([] if intent == "requirements" else [("inquiry", "问诊与意图判断"), ("knowledge_search", "知识检索"), ("knowledge_answer", "知识解答")]
               if intent == "chat" else MULTI_STEPS if mode == "multi" else FIXED_STEPS)
    clarification = state.get("clarification_stage") or ("pre_safety" if intent == "safety_advisory" else "inquiry")
    if legacy and mode == "multi" and status == "awaiting_input":
        # Old sequential runs lack the explicit pause field, and could even
        # turn a completed/null artifact into a question. Use their last executed
        # expert, never question words in the prose, to restore that pause point.
        clarification = next((event["stage"] for event in reversed(trace.get("events", []))
                              if event["stage"] in {"bianzheng", "formula", "herb"}), clarification)
    if mode == "multi" and state.get("agent_stop") == "need_user_input" and not state.get("clarification_stage"):
        clarification = next((stage for stage in ("herb", "formula", "bianzheng") if stage in latest), clarification)
    waiting_index = next((index for index, (stage, _) in enumerate(planned) if stage == clarification), -1)
    for stage, label in planned:
        event = latest.get(stage)
        actual = event.get("status") if event else "not_run"
        if stage == "search" and not event and "rag_decision" in latest:
            actual = "skipped"
        if status == "awaiting_input" and stage == clarification:
            actual = "waiting"
        if status == "awaiting_input" and waiting_index >= 0 and len(steps) > waiting_index:
            # Later stages from a previous candidate are not approval of this revision.
            actual = "superseded" if event else "not_run"
        steps.append({"stage": stage, "label": label, "status": actual,
                      **({"outcome": event["outcome"]} if event and event.get("outcome") else {})})
    # Old fixed runs really did call a separate reviewer. Do not rewrite their history.
    if mode == "fixed" and "reviewer" in latest:
        steps.insert(-1, {"stage": "reviewer", "label": "独立审查（历史流程）", "status": latest["reviewer"]["status"]})
    if status in {"blocked", "awaiting_input", "unavailable"} and "pre_safety" in latest:
        steps.insert(0, {"stage": "pre_safety", "label": "前置安全检查",
                         "status": "unavailable" if unavailable else "blocked" if status == "blocked" else "waiting" if clarification == "pre_safety" else latest["pre_safety"]["status"]})
    compatibility = state.get("compatibility_check") or {"status": "unknown" if "safety" in latest else "not_checked"}
    if status == "awaiting_input":
        compatibility = {"status": "not_checked"}
    output = {"mode": mode, "intent": intent, "status": status, "steps": steps,
              "compatibility": compatibility, "legacy": legacy}
    if intent == "diagnose" and status not in {"cancelled", "failed", "blocked", "unavailable"}:
        from .workflow import public_safety_warnings
        output["safety_warnings"] = public_safety_warnings(state)
        reference = reference_prescription(state)
        if reference:
            output["prescription_reference"] = reference
    if mode == "fixed" and intent == "diagnose":
        sections, details = fixed_sections(state, legacy=legacy)
        output.update(sections=sections, diagnosis_details=details)
    return output
