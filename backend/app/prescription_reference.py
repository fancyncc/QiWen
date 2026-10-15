"""Evidence-backed prescription references remain distinct from approved orders."""
from __future__ import annotations

import json
import re

from .db import one
from .prescription import CandidatePlan, fingerprint, names, review_plan, dose_label
from .safety import normalize, contains_dosing, mentioned_herbs
from .safety_context import applicable_facts
from .dosage_references import literature_dose, CATALOG
from .formula_risks import toxic_herbs


def dose_free(text: str) -> str:
    return text.strip() if isinstance(text, str) and not contains_dosing(text) else ""


def reference_prescription(state: dict) -> dict | None:
    """Recheck the candidate and its source; uncertainty never overrides a conflict.

    This does not approve an order, alter a candidate, or recover one from prose.
    A reference may expose verified composition while optional facts remain
    unknown. Candidate doses require independent verification; literature
    ranges are separately labelled and never promote candidate approval.
    """
    safe = state.get("safety_state", {})
    preflight = state.get("preflight", {})
    if (state.get("intent_type") != "diagnose" or not safe.get("subject_confirmed")
            or safe.get("subject") in {None, "unknown", "other:未明确"}
            or preflight.get("decision") not in {"allow", "clarify"} or preflight.get("question_keys")
            or state.get("reviewer_approved") is False):
        return None
    plan, review = state.get("treatment_plan", {}), state.get("prescription_review", {})
    if not plan.get("herbs") or review.get("plan_hash") != fingerprint(plan):
        return None
    if toxic_herbs(plan):
        return None
    facts = [f for f in applicable_facts(safe) if not f.get("superseded") and f.get("temporal_scope") != "ended"
             and (f.get("category") in {"allergy", "history"} or f.get("temporal_scope") != "past")]
    allergies = [f["entity"] for f in facts if f["category"] == "allergy" and f["assertion"] == "present"]
    allergy_text = "、".join(dict.fromkeys([*allergies, state.get("profile", {}).get("allergies", "")])).strip("、")
    if set(mentioned_herbs(allergy_text)) & {normalize(name) for name in names(plan)}:
        return None
    profile = dict(state.get("profile", {}))
    conditions = [(f["entity"] + "岁" if f["category"] == "age" and f["entity"].isdigit() else f["entity"])
                  for f in facts if f["category"] in {"age", "pregnancy_lactation", "medication", "history"}
                  and f["assertion"] == "present"]
    profile["history"] = "；".join([profile.get("history", ""), *conditions])
    try:
        CandidatePlan.model_validate(plan)
        checked, latest = review_plan(plan, profile)
    except (ValueError, TypeError):
        return None
    if (latest["errors"] or checked != plan or latest["compatibility_status"] != "passed"
            or state.get("reviewer_hash") and state["reviewer_hash"] != fingerprint(plan)):
        return None
    # Any remaining failure must be an explicitly recorded manual review need.
    manual = set(state.get("manual_review_reasons", []))
    if any(error not in manual for error in state.get("safety_errors", [])):
        return None
    source = one("SELECT name,structured_herbs,indication,source FROM formulas WHERE name=?", (plan["name"],))
    if not source or not source["source"] or not dose_free(source["name"] + source["source"]):
        return None
    try:
        source_herbs = json.loads(source["structured_herbs"] or "[]")
        expected = {normalize(item["name"] if isinstance(item, dict) else item) for item in source_herbs}
    except (ValueError, TypeError, KeyError):
        return None
    if not expected or expected != {normalize(name) for name in names(plan)}:
        return None
    rows = [one("SELECT name,effects,contraindications FROM herbs WHERE name IN (?,?)", (item["name"], normalize(item["name"])))
            for item in checked["herbs"]]
    if any(row is None for row in rows):
        return None
    warnings = [*state.get("manual_review_reasons", []), *latest["warnings"], *state.get("treatment_safety_notes", [])]
    no_allergy = any(f["category"] == "allergy" and f["entity"] == "过敏史" and f["assertion"] == "absent" for f in facts)
    warnings.append("过敏风险：已记录" + allergy_text + "，与候选药材的相关风险仍需核对。" if allergy_text
                    else "过敏风险：用户自述无过敏史；仍需注意新出现的过敏反应。" if no_allergy
                    else "过敏风险：过敏史尚未明确，处方参考不能证明个人用药安全。")
    medication = [f["entity"] for f in facts if f["category"] == "medication" and f["assertion"] == "present"]
    no_medication = any(f["category"] == "medication" and f["assertion"] == "absent"
                        and re.fullmatch(r"服药|用药|吃药|服用任何药(?:物|品)?|使用任何药(?:物|品)?", f["entity"]) for f in facts)
    warnings.append("合并用药：已记录" + "、".join(dict.fromkeys(medication)) + "，药物相互作用尚需医师或药师核对。" if medication
                    else "合并用药：用户自述目前未用药；如有用药变化，应重新核对相互作用。" if no_medication
                    else "合并用药：未取得完整用药及相互作用核验资料，需核对正在使用的药物。")
    special = [("年龄" + f["entity"] + "岁" if f["category"] == "age" and f["entity"].isdigit() else f["entity"])
               for f in facts if f["category"] in {"age", "pregnancy_lactation", "history"}
               and f["assertion"] != "absent"]
    warnings.append("特殊人群与适用条件：已记录" + "、".join(dict.fromkeys(special)) + "，需结合这些资料核对候选的适用性。" if special
                    else "特殊人群：年龄、妊娠或哺乳、肝肾功能等适用条件需核对，不能直接套用一般人群资料。")
    warnings.append("配伍：未命中已收录的冲突规则，不等于所有配伍、相互作用及个人禁忌均已排除。")
    warnings.append("剂量：文献参考范围与方案拟用量分列；文献范围不表示适合个人，具体每剂用量、每日剂数及疗程需医师确认。")
    for row in rows:
        warning = dose_free(row["contraindications"] or "")
        warnings.append(row["name"] + "条目禁忌：" + (warning or "禁忌资料未完整核验，需医师核对。"))
    uncertainties = [*state.get("analysis_questions", []),
                     *state.get("agent_artifacts", {}).get("bianzheng", {}).get("uncertainties", []),
                     *state.get("agent_artifacts", {}).get("formula", {}).get("uncertainties", [])]
    rationale = state.get("formula_reasoning") or state.get("agent_artifacts", {}).get("formula", {}).get("comparison", "")
    return {"name": source["name"], "source": source["source"], "indication": dose_free(source["indication"]),
            "rationale": dose_free(rationale),
            "herbs": [{**item, "dosage": item["dosage"] if item["verification_status"] == "verified"
                       else {"min": None, "max": None, "unit": item["dosage"]["unit"]},
                       "preparation": item["preparation"] if item["verification_status"] == "verified" else "",
                       "reference_dose": literature_dose(item["name"]),
                       "effect": dose_free(row["effects"] or ""), "cautions": dose_free(row["contraindications"] or "")}
                      for item, row in zip(checked["herbs"], rows)],
            "dose_notice": CATALOG["notice"],
            "uncertainties": list(dict.fromkeys(filter(None, map(dose_free, uncertainties)))),
            "safety_warnings": list(dict.fromkeys(filter(None, map(dose_free, warnings)))),
            "approved": False}


def render_reference(reference: dict) -> str:
    lines = ["### 处方参考（待医师复核）", "", "**" + reference["name"] + "**"]
    if reference["rationale"]:
        lines.extend(["", "选方依据与条件：" + reference["rationale"]])
    elif reference["indication"]:
        lines.extend(["", "方剂条目适应描述：" + reference["indication"] + "。需结合上述初步辨证核对个人适用性。"])
    lines.extend(["", "参考组成："])
    for item in reference["herbs"]:
        dose = dose_label(item) + "（记录已核验，个人用量仍需确认）" if item["dosage"]["min"] is not None else "剂量待核验，未提供"
        entry = item.get("reference_dose")
        literature = ("；文献参考范围：" + dose_label({"dosage": entry}) + "，" + entry["basis"]
                      + "；[" + entry["source"] + "](" + entry["url"] + ")") if entry else ""
        lines.append("- " + item["name"] + "：方案拟用量" + dose + literature)
    if reference.get("dose_notice"):
        lines.extend(["", reference["dose_notice"]])
    lines.extend(["", "方剂出处：" + reference["source"]])
    if reference["uncertainties"]:
        lines.extend(["", "尚未确认的依据：", *("- " + value for value in reference["uncertainties"])])
    lines.extend(["", "### 安全说明", "", *("- " + value for value in reference["safety_warnings"]),
                  "", "以上为待复核的处方参考，未批准个人用药；请勿自行抓药、服用或调整用药。"])
    return "\n".join(lines)
