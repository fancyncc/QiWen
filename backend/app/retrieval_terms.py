"""Content constraints shared by recall, windows and evidence admission.

These checks preserve explicit query requirements; they are not semantic labels.
"""
from __future__ import annotations

import re

TERM_VERSION = 1
# Orthographic variants only. Processing names (e.g. 炙黄芪) stay distinct.
NEGATIVE_SWEAT = re.compile(r"(?:不可|不得|不宜|勿|禁|慎勿|无令)(?:发其汗|发汗|汗之|汗)")
POSITIVE_SWEAT = re.compile(r"(?:可|宜|当)(?:发其汗|发汗|汗之)")
FORMULA_STATEMENT = re.compile(r"[\u3400-\u9fff]{2,14}[汤丸散饮膏丹](?:主之|方|。|，)|(?:宜|服|用).{0,16}[汤丸散饮膏丹]")


def normalize_search(text: str) -> str:
    # Use a whole-name replacement: 耆 in other words is not a drug alias.
    return (text.replace("黃耆", "黄芪").replace("黃芪", "黄芪").replace("黄耆", "黄芪")
            .replace("癥病", "症病").replace("發汗", "发汗").replace("發其汗", "发其汗").casefold())


def matching_terms(text: str, terms: list[str]) -> list[str]:
    normalized = normalize_search(text)
    return [term for term in terms if normalize_search(term) in normalized]


def content_check(text: str, plan: dict, *, heading: str = "") -> dict:
    normalized = normalize_search(text)
    missing = []
    for group in plan.get("required_term_groups", []):
        if not any(normalize_search(term) in normalized for term in group):
            missing.append(group)
    entities = plan.get("required_entities", [])
    if entities and not all(normalize_search(entity) in normalize_search(heading + "\n" + text)
                            for entity in entities):
        return {"passed": False, "reason": "target_entity_missing", "missing_groups": missing}
    polarity = plan.get("action_polarity")
    if polarity == "negative_sweat" and not NEGATIVE_SWEAT.search(normalized):
        return {"passed": False, "reason": "negative_action_missing", "missing_groups": missing}
    if polarity == "positive_sweat" and not POSITIVE_SWEAT.search(NEGATIVE_SWEAT.sub("", normalized)):
        return {"passed": False, "reason": "positive_action_missing", "missing_groups": missing}
    if polarity == "contrast_sweat" and not (NEGATIVE_SWEAT.search(normalized) or POSITIVE_SWEAT.search(normalized)):
        return {"passed": False, "reason": "action_missing", "missing_groups": missing}
    if plan.get("requested_property") == "formula" and not FORMULA_STATEMENT.search(normalized):
        return {"passed": False, "reason": "formula_statement_missing", "missing_groups": missing}
    if plan.get("requested_property") == "disease_definition" and not re.search(r"之为病|之爲病|提纲", normalized):
        return {"passed": False, "reason": "disease_definition_missing", "missing_groups": missing}
    return {"passed": not missing, "reason": "required_content_missing" if missing else "",
            "missing_groups": missing}


def content_priority(row: dict, plan: dict) -> tuple:
    body = row["body"]
    heading = row.get("chapter", "")
    checked = content_check(body, plan, heading=heading)
    groups = plan.get("required_term_groups", [])
    covered = sum(bool(matching_terms(body, group)) for group in groups)
    # Constrained evidence precedes incidental word mentions before pool truncation.
    return (int(checked["passed"]), covered) if (groups or plan.get("required_entities") or plan.get("requested_property") or
                                               plan.get("action_polarity")) else (0, 0)
