"""Public conceptual sources are separate from classical and patient corpora."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

CONCEPT_PATH = Path(__file__).resolve().parents[1] / "resources" / "concept_evidence.json"


def concept_rows(path=None):
    data = json.loads((path or CONCEPT_PATH).read_text(encoding="utf-8"))
    if data["version"] != 1:
        raise ValueError("Unsupported concept evidence version")
    rows = []
    for entry in data["entries"]:
        parsed = urlsplit(entry["source_url"])
        if (entry["verification_status"] != "source_verified" or parsed.scheme != "https"
                or parsed.username or parsed.password or not entry["source_locator"]
                or not entry["source_quote"] or not entry["body"]):
            raise ValueError("Concept evidence requires verified public provenance")
        # Include provenance in identity, so an edited locator invalidates old evidence too.
        identity = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        rows.append({"id": hashlib.sha256(identity.encode()).hexdigest(), "title": entry["title"],
                     "chapter": entry["chapter"], "body": entry["body"],
                     "metadata": json.dumps({**entry, "evidence_kind": "concept_definition"}, ensure_ascii=False)})
    return rows


def matching_concepts(plan):
    if plan.get("evidence_kind") != "concept_definition" or plan.get("source_hints"):
        return []
    # A scope note supports a definition, not a comparison or relationship claim.
    if re.search(r"关系|区别|比较|联系|原因|治疗|用药|以及|还有|和|与", plan["semantic_query"]):
        return []
    terms = {term.casefold() for term in plan.get("lexical_terms", [])}
    return [row for row in concept_rows()
            if terms.intersection(alias.casefold() for alias in json.loads(row["metadata"])["aliases"])]


def source_lookup(classical_rows):
    return {row["id"]: row for row in [*classical_rows, *concept_rows()]}
