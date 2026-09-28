from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from .runtime import EmptyArgs, ToolSpec
from ..classics import passage_excerpt, retrieve_classics
from ..db import all_rows, one
from ..memory import rerank_memories, retrieve_memories
from ..safety import check_herbs, normalize
from ..terms import search_plan


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
    base = search_plan(state, attempt=min(2, state.get("search_attempts", 0)))
    terms = list(dict.fromkeys([*base.get("lexical_terms", []), *re.findall(r"[\u3400-\u9fff]{2,}|[A-Za-z0-9]+", args.query)]))
    plan = {**base, "keyword_query": args.query, "semantic_query": args.query,
            "lexical_terms": terms or base.get("lexical_terms", [])}
    rows, trace = await retrieve_classics(plan)
    return {"documents": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                             "excerpt": passage_excerpt(row)} for row in rows],
            "status": trace["rerank"]["status"], "warning": trace["warning"]}


def get_patient_profile(state: dict, _args: EmptyArgs) -> dict:
    safe = state.get("safety_state", {})
    return {"subject": safe.get("subject", "self"), "profile": state.get("profile", {}),
            "confirmed_facts": [fact for fact in safe.get("facts", [])
                                if fact.get("confirmation") == "confirmed"]}


async def search_memories(state: dict, args: SearchArgs) -> dict:
    rows, info = retrieve_memories(args.query, state["user_id"], state["thread_id"], 10)
    rows, info = await rerank_memories(args.query, rows, info, state["user_id"])
    return {"memories": [{"thread_id": row["thread_id"], "updated_at": row["updated_at"],
                           "summary": row["summary"][:500]} for row in rows],
            "mode": info.get("active_mode"), "rerank": info.get("rerank", {})}


def search_formulas(_state: dict, args: SearchArgs) -> dict:
    escaped = args.query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = all_rows(
        "SELECT name,structured_herbs,indication,source FROM formulas "
        "WHERE name LIKE ? ESCAPE '\\' OR indication LIKE ? ESCAPE '\\' ORDER BY name LIMIT 10",
        (f"%{escaped}%", f"%{escaped}%"),
    )
    return {"formulas": [{"name": row["name"], "indication": row["indication"], "source": row["source"],
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
    return {"allergy_risks": allergies, "special_risks": special,
            "medication_interaction_status": "requires_human_review" if re.search(r"服用|用药|药物|华法林|抗凝", text) else "unknown",
            "safe": not allergies and not special,
            "warning": "当前仅检查已确认画像和已收录规则，未知药物相互作用必须由医师或药师复核。"}


PROFILE_TOOL = ToolSpec("get_patient_profile", "读取当前咨询对象已经确认的画像和安全事实。",
                        EmptyArgs, get_patient_profile, 3)
MEMORY_TOOL = ToolSpec("search_memories", "检索当前用户其他会话中的相关历史自述。",
                       SearchArgs, search_memories, 15)
CLASSICS_TOOL = ToolSpec("search_classics", "检索古籍并返回经过相关性门槛的可追溯片段。",
                         SearchArgs, search_classics, 30)
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
