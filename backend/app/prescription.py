from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .db import all_rows, one
from .safety import check_herbs, normalize, mentioned_herbs


class Dosage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)
    unit: Literal["g", "mg", "ml", "unconverted"] = "g"

    @model_validator(mode="after")
    def interval(self):
        if (self.min is None) != (self.max is None):
            raise ValueError("剂量上下限需同时提供或同时为空")
        if self.min is not None and (self.min <= 0 or self.max < self.min):
            raise ValueError("剂量范围无效")
        return self


class Herb(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    herb_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    dosage: Dosage = Field(default_factory=Dosage)
    preparation: str = Field(default="", max_length=500)
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    verification_status: Literal["unverified", "verified", "rejected"] = "unverified"


class CandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=100)
    herbs: list[Herb] = Field(max_length=40)


def herb_id(name: str) -> str:
    return "herb:" + hashlib.sha256(normalize(name).encode()).hexdigest()[:20]


def parse_seed_dosage(text: str) -> dict:
    # Retain historical units; never convert ancient measures into modern grams.
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(?:\s*[-～~]\s*(\d+(?:\.\d+)?))?\s*(g|克|mg|毫克|ml|毫升)\s*", text)
    if not match:
        return {"min": None, "max": None, "unit": "unconverted" if text else "g"}
    return {"min": float(match[1]), "max": float(match[2] or match[1]),
            "unit": {"克": "g", "毫克": "mg", "毫升": "ml"}.get(match[3], match[3])}


def structured_herb(name: str, *, raw_dosage="", preparation="", source_ids=None) -> dict:
    return Herb(herb_id=herb_id(name), name=name, dosage=Dosage(**parse_seed_dosage(raw_dosage)),
                preparation=preparation, source_ids=source_ids or []).model_dump()


def migrate_plan(plan: dict) -> dict:
    """Historical string lists become empty-dose, unverified entries."""
    if not isinstance(plan, dict):
        return {}
    return {**plan, "herbs": [structured_herb(item) if isinstance(item, str) else item for item in plan.get("herbs", [])]}


def legacy_plan(content: str) -> dict | None:
    """Recover only names from the old stored example; never infer doses or approval."""
    match = re.search(r"## 方剂知识示例\s+\*\*([^*\n]+)\*\*：([^\n]+)", content)
    if not match:
        return None
    herbs = mentioned_herbs(match.group(2))
    if not herbs or check_herbs(herbs):
        return None
    return migrate_plan({"name": match.group(1), "herbs": herbs})


def names(plan: dict) -> list[str]:
    return [item["name"] if isinstance(item, dict) else item for item in plan.get("herbs", [])]


def fingerprint(plan: dict) -> str:
    # Exclude derived statuses: the hash represents the proposed clinical data.
    material = {"name": plan.get("name"), "herbs": [
        {key: item.get(key) for key in ("herb_id", "name", "dosage", "preparation", "source_ids")}
        for item in plan.get("herbs", []) if isinstance(item, dict)]}
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def patient_fingerprint(profile: dict) -> str:
    return hashlib.sha256(json.dumps({key: profile.get(key, "") for key in ("constitution", "allergies", "history")},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def review_plan(plan: dict, profile: dict) -> tuple[dict, dict]:
    plan = CandidatePlan.model_validate(plan).model_dump()
    errors = check_herbs(names(plan))
    dosage_errors = []
    warnings = []
    seen = set()
    statuses = []
    for item in plan["herbs"]:
        item["verification_status"] = "unverified"  # Only this verifier can promote a status.
        canonical = normalize(item["name"])
        if item["herb_id"] != herb_id(canonical) or not one("SELECT name FROM herbs WHERE name IN (?,?)", (canonical, item["name"])):
            errors.append("药材标识与药名不一致或未收录")
        if item["herb_id"] in seen:
            errors.append("候选存在重复药材")
        seen.add(item["herb_id"])
        if canonical in str(profile.get("allergies", "")):
            errors.append(f"画像提示可能对{item['name']}过敏")
        source_rows = [one("SELECT * FROM dosage_evidence WHERE id=? AND herb_id=?", (source_id, item["herb_id"]))
                       for source_id in item["source_ids"]]
        invalid_source = any(row is None for row in source_rows)
        if invalid_source:
            dosage_errors.append("剂量出处不存在或与药材不一致")
            item["verification_status"] = "rejected"
        dose = item["dosage"]
        if dose["min"] is not None:
            if not source_rows:
                item["dosage"] = {"min": None, "max": None, "unit": dose["unit"]}
                warnings.append(f"{item['name']}未提供剂量出处，已清空剂量")
                statuses.append("unverified")
                continue
            matching = [row for row in source_rows if row and row["unit"] == dose["unit"]
                        and row["min_dose"] is not None and row["max_dose"] is not None
                        and row["min_dose"] <= dose["min"] <= dose["max"] <= row["max_dose"]
                        and row["preparation"] == item["preparation"]]
            reviewed = [row for row in matching if row["verification_status"] == "verified"
                        and row["reviewed_by"] and row["reviewed_at"] and row["population"] == "general_adult"]
            patient_text = " ".join(str(profile.get(key, "")) for key in ("constitution", "allergies", "history"))
            adult_confirmed = re.search(r"成人|成年|(?:1[89]|[2-9]\d)\s*岁", patient_text)
            if reviewed and not invalid_source and adult_confirmed and not re.search(r"怀孕|孕期|孕妇|妊娠|备孕|哺乳|儿童|未成年|华法林|抗凝|肝|肾", patient_text):
                item["verification_status"] = "verified"
            elif source_rows and not matching:
                item["verification_status"] = "rejected"
                dosage_errors.append(f"{item['name']}的剂量、单位或炮制与所引出处不一致")
            else:
                warnings.append(f"{item['name']}剂量尚未核验，需人工复核")
        else:
            warnings.append(f"{item['name']}缺少可核验剂量")
        statuses.append(item["verification_status"])
    review = {"compatibility_status": "failed" if errors else "passed",
              "dosage_status": "rejected" if dosage_errors or "rejected" in statuses else "verified" if statuses and all(value == "verified" for value in statuses) else "unverified",
              "errors": list(dict.fromkeys(errors + dosage_errors)), "warnings": warnings, "plan_hash": fingerprint(plan),
              "patient_hash": patient_fingerprint(profile), "approved": not errors and not dosage_errors}
    return plan, review


def dose_label(item: dict) -> str:
    dose = item["dosage"]
    if dose["min"] is None:
        return "剂量未提供"
    low, high = f'{dose["min"]:g}', f'{dose["max"]:g}'
    return (low if low == high else low + "–" + high) + " " + dose["unit"]


def render_plan(plan: dict) -> str:
    return f'**{plan["name"]}**：' + "、".join(
        item["name"] + "（" + dose_label(item) + "；" + ("已核验" if item["verification_status"] == "verified" else "未核验")
        + ("；" + item["preparation"] if item["preparation"] else "") + "）" for item in plan["herbs"])


def knowledge_context() -> str:
    rows = all_rows("SELECT id,herb_id,herb_name,min_dose,max_dose,unit,preparation,source,verification_status FROM dosage_evidence LIMIT 120")
    return json.dumps(rows, ensure_ascii=False)
