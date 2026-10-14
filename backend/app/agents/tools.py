from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from .runtime import EmptyArgs, ToolSpec
from ..blocking import run_blocking
from ..classics import passage_excerpt, retrieve_classics, retrieval_cache_key
from ..db import all_rows, one
from ..safety import check_herbs, normalize
from ..safety_context import uncovered_allergens
from ..terms import search_plan, CATALOG
from ..query_plan import knowledge_plan
from ..retrieval_budget import timeout_seconds


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=300)


class LookupHerbArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=100)


class HerbListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    herbs: list[str] = Field(min_length=1, max_length=40)


async def search_classics(state: dict, args: SearchArgs) -> dict:
    if (args.query.strip() == str(state.get("search_query", "")).strip()
            and (state.get("docs") or state.get("search_attempts", 0) >= 3)):
        return {"documents": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                                "excerpt": passage_excerpt(row)} for row in state.get("docs", [])],
                "status": "cached_workflow_evidence", "warning": state.get("retrieval_warning", "")}
    # Clinical searches use resolved facts, never unrestricted model-written symptoms.
    from ..model import invocation
    context = invocation.get()
    clinical = state.get("identity_mode") == "clinical" or state.get("symptoms")
    plan = knowledge_plan(args.query) if not clinical else search_plan(state)
    if clinical:
        for attempt in range(min(2, state.get("search_attempts", 0)), 3):
            plan = search_plan(state, attempt=attempt)
            key = retrieval_cache_key(plan)
            previous = context.retrieval_cache.get(key) if context else None
            if not previous or previous[1].get("outcome") != "no_match" or not previous[1].get("retryable", True):
                break
    rows, trace = await retrieve_classics(plan)
    return {"documents": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                             "excerpt": passage_excerpt(row), "retrieval": row.get("retrieval", {})} for row in rows],
            "status": trace.get("outcome", trace.get("rerank", {}).get("status", "unknown")),
            "warning": trace["warning"], "retrieval_trace": trace}


def get_patient_profile(state: dict, _args: EmptyArgs) -> dict:
    safe = state.get("safety_state", {})
    subject = safe.get("subject", "unknown")
    resolved = safe.get("subject_confirmed") and subject not in {"unknown", "other:未明确"} and state.get("identity_mode") == "clinical" and not state.get("identity_urgent")
    return {"subject": subject, "status": "available" if resolved else "not_applicable",
            "profile": state.get("profile", {}) if resolved and subject == "self" else {},
            "confirmed_facts": [fact for fact in safe.get("facts", [])
                                if resolved and fact.get("subject") == subject and fact.get("confirmation") == "confirmed"
                                and fact.get("source", {}).get("kind") != "summary"]}


def search_formulas(state: dict, args: SearchArgs) -> dict:
    def related_terms(values):
        terms = list(values)
        for entry in CATALOG.values():
            if any(alias in value for value in values for alias in entry["aliases"]):
                terms += entry["aliases"] + [item["term"] for item in entry["expansions"]]
        return list(dict.fromkeys(term.strip() for term in terms if len(term.strip()) >= 2))[:30]

    def matches(terms):
        if not terms:
            return []
        params = []
        for term in terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params += [f"%{escaped}%"] * 2
        return all_rows("SELECT name,structured_herbs,indication,source FROM formulas WHERE "
                        + " OR ".join("(name LIKE ? ESCAPE '\\' OR indication LIKE ? ESCAPE '\\')" for _ in terms)
                        + " ORDER BY name LIMIT 10", tuple(params))

    terms = related_terms([args.query, *re.split(r"[\s，,、；;：:（）()]+", args.query)])
    rows = matches(terms)
    fallback = False
    if not rows and state.get("symptoms"):
        terms = related_terms(state["symptoms"])
        rows = matches(terms)
        fallback = bool(rows)
    return {"matched_terms": terms, "symptom_fallback": fallback,
            "warning": "原查询无匹配，以下按本轮明确症状检索，仅作知识对照，不代表个人选方。" if fallback else "",
            "formulas": [{"name": row["name"], "indication": row["indication"], "source": row["source"],
                            "herbs": json.loads(row["structured_herbs"] or "[]")} for row in rows]}


def lookup_herb(_state: dict, args: LookupHerbArgs) -> dict:
    row = one("SELECT name,herb_id,properties,effects,indications,dosage,processing,contraindications,source,"
              "structured_dosage,source_ids,verification_status FROM herbs WHERE name=?", (args.name.strip(),))
    if not row:
        return {"found": False, "name": args.name}
    for key, default in (("structured_dosage", "{}"), ("source_ids", "[]")):
        row[key] = json.loads(row[key] or default)
    return {"found": True, **row}


def get_dosage_evidence(_state: dict, args: LookupHerbArgs) -> dict:
    rows = all_rows("SELECT id,herb_id,herb_name,min_dose,max_dose,unit,raw_dosage,preparation,source,"
                    "source_locator,population,verification_status,reviewed_at FROM dosage_evidence "
                    "WHERE herb_name=? ORDER BY id LIMIT 20", (args.name.strip(),))
    return {"herb": args.name, "evidence": rows}


def check_herb_compatibility(_state: dict, args: HerbListArgs) -> dict:
    violations = check_herbs(args.herbs)
    return {"status": "failed" if violations else "passed", "violations": violations,
            "warning": "未命中内置规则不代表临床安全。"}


def check_user_risks(state: dict, args: HerbListArgs) -> dict:
    profile = state.get("profile", {})
    text = " ".join(str(profile.get(key, "")) for key in ("constitution", "allergies", "history"))
    allergies = [name for name in args.herbs if normalize(name) in text and normalize(name) in str(profile.get("allergies", ""))]
    special = [label for label, pattern in (
        ("pregnancy_lactation", r"怀孕|孕期|孕妇|妊娠|备孕|哺乳"),
        ("minor", r"儿童|未成年|婴儿|幼儿"),
        ("liver_kidney", r"肝功能|肾功能|肝病|肾病|肝损|肾损"),
        ("anticoagulant", r"华法林|抗凝"),
    ) if re.search(pattern, text)]
    uncovered = uncovered_allergens({"subject": "unknown", "facts": [], **state.get("safety_state", {})})
    return {"allergy_risks": allergies, "special_risks": special, "uncovered_allergens": uncovered,
            "requires_human_review": bool(uncovered),
            "medication_interaction_status": "requires_human_review" if uncovered or re.search(r"服用|用药|药物|华法林|抗凝", text) else "unknown",
            "safe": not allergies and not special and not uncovered,
            "warning": "当前仅检查已确认画像和已收录规则，未知药物相互作用必须由医师或药师复核。"}


PROFILE_TOOL = ToolSpec("get_patient_profile", "读取当前咨询对象已经确认的画像和安全事实。",
                        EmptyArgs, get_patient_profile, 3)
CLASSICS_TOOL = ToolSpec("search_classics", "检索古籍并返回经过相关性门槛的可追溯片段。",
                         SearchArgs, search_classics, timeout_seconds() + 1)
FORMULA_TOOL = ToolSpec("search_formulas", "按名称或适应描述查询结构化方剂知识。",
                        SearchArgs, search_formulas, 5)
HERB_TOOL = ToolSpec("lookup_herb", "查询单味药材的属性、来源、炮制和禁忌信息。",
                     LookupHerbArgs, lookup_herb, 5)
DOSAGE_TOOL = ToolSpec("get_dosage_evidence", "查询单味药材的结构化剂量证据；没有记录时必须保持未核验。",
                       LookupHerbArgs, get_dosage_evidence, 5)
COMPATIBILITY_TOOL = ToolSpec("check_herb_compatibility", "按内置十八反十九畏规则检查药材组合。",
                              HerbListArgs, check_herb_compatibility, 5)
USER_RISK_TOOL = ToolSpec("check_user_risks", "结合当前确认画像检查过敏和已收录的特殊风险。",
                          HerbListArgs, check_user_risks, 5)
