"""Versioned term catalog. Search expansions are not clinical equivalences."""
from __future__ import annotations

import json
import re
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[1] / "resources" / "symptom_terms.json"


def load_catalog(path=CATALOG_PATH):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != 1:
        raise ValueError("Unsupported symptom catalog version")
    entries, aliases = {}, {}
    for entry in data["entries"]:
        name = entry["canonical"]
        if not name or name in entries or name not in entry["aliases"]:
            raise ValueError("Invalid canonical term")
        for alias in entry["aliases"]:
            if not alias or alias in aliases:
                raise ValueError("Duplicate symptom alias")
            aliases[alias] = name
        for expansion in entry["expansions"]:
            if (expansion["relation"] != "retrieval_related" or
                    expansion["term"] not in expansion["evidence"]["quote"] or
                    not re.fullmatch(r"[a-f0-9]{64}", expansion["evidence"]["id"])):
                raise ValueError("Search expansion requires source evidence")
        entries[name] = entry
    return entries


CATALOG = load_catalog()
SYMPTOM_ALIASES = {name: tuple(entry["aliases"]) for name, entry in CATALOG.items()}
CLASSIC_TERMS = {name: " ".join(e["term"] for e in entry["expansions"])
                 for name, entry in CATALOG.items() if entry["expansions"]}


def unique(items):
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def search_plan(state: dict, attempt=0) -> dict:
    primary = unique(state.get("symptoms", []))
    aliases, expansions, unknown = [], [], []
    for term in primary:
        entry = CATALOG.get(term)
        if not entry:
            unknown.append(term)
            continue
        aliases.extend(entry["aliases"])
        expansions.extend(entry["expansions"])
    expanded = unique([*primary, *(aliases if attempt >= 1 else []),
                       *(e["term"] for e in expansions if attempt >= 2)])
    subject = state.get("safety_state", {}).get("subject", "self")
    descriptors = []
    for fact in state.get("symptom_facts", []):
        if (not fact.get("superseded_by") and fact["subject"] == subject and
                fact["time_scope"] == "current" and fact["assertion"] == "present" and
                fact["symptom"] in primary):
            descriptors.append(" ".join(unique([fact["symptom"], fact.get("degree_text", ""),
                                                  fact.get("duration_text", ""), fact.get("trend_text", "")])))
    query = " ".join(expanded) or state.get("message", "")[:300]
    return {"keyword_query": query, "semantic_query": "；".join(unique([*primary, *descriptors])) or query,
            "primary": primary, "lexical_terms": expanded,
            "expansions": expansions if attempt >= 2 else [], "unmapped": unknown,
            "attempt": attempt + 1, "catalog_version": 1}
