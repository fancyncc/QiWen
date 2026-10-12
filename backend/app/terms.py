"""Versioned term catalog. Search expansions are not clinical equivalences."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from .query_plan import make_plan

CATALOG_PATH = Path(__file__).resolve().parents[1] / "resources" / "symptom_terms.json"


def load_catalog(path=CATALOG_PATH):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != 1:
        raise ValueError("Unsupported symptom catalog version")
    entries, aliases = {}, {}
    source_cache = {}
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
            evidence = expansion["evidence"]
            book = evidence["book"]
            if book not in source_cache:
                source_path = CATALOG_PATH.parent / "classics_segments" / f"{book}.json"
                if source_path.parent != CATALOG_PATH.parent / "classics_segments":
                    raise ValueError("Invalid expansion source name")
                segments = json.loads(source_path.read_text(encoding="utf-8"))
                source_cache[book] = {hashlib.sha256(f'{book}\n{row["content"].strip()}'.encode()).hexdigest():
                                      row["content"].strip() for row in segments}
            if evidence["quote"] not in source_cache[book].get(evidence["id"], ""):
                raise ValueError("Expansion source quote does not match bundled corpus")
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
    # Related classical terms participate in the first pass, never in patient facts.
    expanded = unique([*primary, *aliases, *(e["term"] for e in expansions)])
    subject = state.get("safety_state", {}).get("subject", "self")
    descriptors = []
    for fact in state.get("symptom_facts", []):
        if (not fact.get("superseded_by") and fact["subject"] == subject and
                fact["time_scope"] == "current" and fact["assertion"] == "present" and
                fact["symptom"] in primary):
            descriptors.append(" ".join(unique([fact["symptom"], fact.get("degree_text", ""),
                                                  fact.get("duration_text", ""), fact.get("trend_text", "")])))
    query = " ".join(expanded) or state.get("message", "")[:300]
    semantic = "；".join(unique([*primary, *descriptors])) or query
    plan = make_plan(query, semantic, expanded, primary=primary, expansions=expansions,
                     unmapped=unknown, attempt=attempt + 1)
    plan["evidence_kind"] = "clinical_classics"
    plan["recall_queries"] = [{"kind": "original", "keyword_query": " ".join(unique([*primary, *aliases])) or query,
                               "semantic_query": semantic}]
    classical = unique([e["term"] for e in expansions])
    if classical:
        plan["recall_queries"].append({"kind": "classical_related", "keyword_query": " ".join(classical),
                                       "semantic_query": " ".join(classical)})
        plan["rerank_query"] = " ".join(unique([*primary, *classical]))
    return plan
