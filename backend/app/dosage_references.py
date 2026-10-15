"""Attributed literature ranges. Never fed into personal-order approval.

Source transcription is separate from the human-reviewed dosage_evidence
table. In particular these ranges must not promote verification_status.
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = json.loads((Path(__file__).resolve().parents[1] / "resources" / "dosage_references.json").read_text(encoding="utf-8"))


def literature_dose(name: str) -> dict | None:
    # Exact preparation identity, deliberately not safety.normalize (which
    # collapses red/white peony and raw/honey-prepared liquorice).
    row = next((item for item in CATALOG["herbs"] if item["name"] == name), None)
    if not row:
        return None
    return {"min": row["min"], "max": row["max"], "unit": row["unit"],
            "kind": "literature_range", "basis": CATALOG["basis"], "preparation": row["preparation"],
            "source": CATALOG["source"] + "（" + row["edition"] + "年版）",
            "url": "https://shennongalpha.westlake.edu.cn/knowledge/" + row["locator"],
            "quote": row["quote"], "checked_at": CATALOG["version"]}
