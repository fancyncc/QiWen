"""Repair unsupported generated mixtures by asking for a sourced choice, never choosing for the model."""
from __future__ import annotations

import copy
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from .consultation_followup import needs_patient_answer
from .context import clinical_context
from .prescription import names
from .safety import normalize
from .formula_risks import toxic_herbs


def matches_source(plan: dict, sources: list[dict]) -> bool:
    source = next((row for row in sources if row["name"] == plan.get("name")), None)
    proposed = [normalize(name) for name in names(plan)]
    expected = [normalize(name) for name in names(source or {})]
    if not (proposed and len(proposed) == len(set(proposed)) and set(proposed) == set(expected)):
        return False
    expected_rows = {normalize(item["name"]): item for item in source["herbs"]}
    return all(isinstance(item, dict) and item.get("herb_id") == expected_rows[normalize(item["name"])].get("herb_id")
               and bool(item.get("source_ids"))
               and set(item["source_ids"]) <= set(expected_rows[normalize(item["name"])].get("source_ids", []))
               for item in plan["herbs"])


class SourceChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected_name: str
    information_gap: Literal["optional", "evidence"]
    rationale: str = Field(max_length=2000)
    safety_notes: list[str] = Field(default_factory=list, max_length=8)


async def repair_source_choice(state: dict, plan: dict, sources: list[dict], generate) -> dict:
    if needs_patient_answer(plan) or not sources:
        return plan
    # A literature entry is not a suitable provisional prescription merely
    # because symptoms overlap. Toxic ingredients require clinical supervision,
    # reliable preparation and dose review; do not propose them through this
    # unapproved reference path. The independent approval path is unchanged.
    sources = [row for row in {row["name"]: row for row in sources}.values() if not toxic_herbs(row)]
    if not sources:
        return {**plan, "name": "本轮未取得可供参考的合适方剂", "herbs": [], "information_gap": "evidence",
                "rationale": "已取得的候选涉及有毒药材，炮制、剂量与临床适用性尚待医师核对，不能作为本轮处方参考。",
                "safety_notes": ["涉及有毒药材的方案须由医师面诊评估，不能直接按文献组成自行用药。"]}
    if matches_source(plan, sources):
        return plan
    sources = sources[:4]
    # The empty option is essential: available source rows do not prove a
    # candidate fits this patient. No first-result or symptom-to-formula rule.
    options = tuple(dict.fromkeys(["", *(row["name"] for row in sources)]))
    schema = create_model("RetrievedFormulaChoice", __base__=SourceChoice,
                          selected_name=(Literal[options], Field(...)))
    payload = [{"name": row["name"], "indication": row.get("indication", ""), "source": row.get("source", ""),
                "composition": names(row), "contraindications": row.get("contraindications", "")} for row in sources]
    choice = await generate(
        "原候选的名称或组成未对应实际取得的方剂来源。请重新比较已知支持、反证和未确认条件，"
        "只选择实际来源中有临床依据的一个标准方作为待医师复核的参考。selected_name 必须是列出的方名或空字符串；"
        "不能为了有结果而勉强选方，不适合或不足以支持任何候选时选空并说明具体原因。只有部分症状重合不等于方证相符；"
        "条目记载的重症表现、典型症状不能反写成用户已报告，也不能忽略条目禁忌。"
        "rationale 说明选择与鉴别条件，safety_notes 说明过敏、合并用药、人群、禁忌及剂量核验问题。"
        "程序从该条目复制原组成并清空所有剂量；不生成自拟组合、剂量或用法，不批准个人用药。返回 Schema JSON。",
        clinical_context(state) + "\n初步辨证：" + state.get("diagnosis", "")
        + "\n实际检索来源：" + json.dumps(payload, ensure_ascii=False)
        + "\n原候选未通过来源对应：" + str(plan.get("name", "")),
        role="formula", schema=schema,
    )
    choice = schema.model_validate(choice).model_dump()
    if not choice["selected_name"]:
        return {**plan, "name": "未找到有依据的方剂候选", "herbs": [], "information_gap": "evidence",
                "rationale": choice["rationale"], "safety_notes": choice["safety_notes"]}
    source = next(row for row in sources if row["name"] == choice["selected_name"])
    herbs = copy.deepcopy(source["herbs"])
    for item in herbs:
        item.update(dosage={"min": None, "max": None, "unit": "g"}, preparation="", verification_status="unverified")
    return {**plan, "name": source["name"], "herbs": herbs, "information_gap": choice["information_gap"],
            "rationale": choice["rationale"], "safety_notes": choice["safety_notes"]}
