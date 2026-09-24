"""Load the bundled, attributed herb and formula seed set."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SEED_FILE = Path(__file__).resolve().parents[1] / "resources" / "knowledge_seed.json"


def _column(db: sqlite3.Connection, table: str, name: str) -> None:
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if name not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")


def seed_bundled_knowledge(db: sqlite3.Connection) -> None:
    for column in ("aliases", "meridians", "indications", "dosage", "processing", "contraindications"):
        _column(db, "herbs", column)
    for column in ("composition", "modifications", "contraindications", "category"):
        _column(db, "formulas", column)
    db.execute("""CREATE TABLE IF NOT EXISTS herb_cautions (
        herb TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL,
        severity TEXT NOT NULL, source TEXT NOT NULL,
        PRIMARY KEY (herb, kind, detail)
    )""")
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    for herb in data["herbs"]:
        db.execute("""INSERT INTO herbs(
            name, properties, effects, source, aliases, meridians, indications,
            dosage, processing, contraindications
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            properties=excluded.properties, effects=excluded.effects,
            source=excluded.source, aliases=excluded.aliases,
            meridians=excluded.meridians, indications=excluded.indications,
            dosage=excluded.dosage, processing=excluded.processing,
            contraindications=excluded.contraindications
        WHERE herbs.source IN ('内置演示种子', 'LingYi 基础种子')""", (
            herb["name"], herb["nature_flavor"], herb["efficacy"], "LingYi 基础种子",
            json.dumps(herb["aliases"], ensure_ascii=False),
            json.dumps(herb["meridians"], ensure_ascii=False),
            json.dumps(herb["indications"], ensure_ascii=False),
            herb["dosage"], herb["processing"], herb["contraindications"],
        ))
    for formula in data["formulas"]:
        db.execute("""INSERT INTO formulas(
            name, herbs, indication, source, composition, modifications, contraindications, category
        ) VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            herbs=excluded.herbs, indication=excluded.indication,
            source=excluded.source, composition=excluded.composition,
            modifications=excluded.modifications,
            contraindications=excluded.contraindications, category=excluded.category
        WHERE formulas.source IN ('内置演示种子', 'LingYi 基础种子')""", (
            formula["name"],
            json.dumps([item["herb"] for item in formula["composition"]], ensure_ascii=False),
            formula["indication"], formula["source"],
            json.dumps(formula["composition"], ensure_ascii=False),
            formula["modifications"], formula["contraindications"], formula["category"],
        ))
    db.executemany(
        "INSERT OR IGNORE INTO herb_cautions(herb,kind,detail,severity,source) VALUES(?,?,?,?,?)",
        [(item["herb"], item["type"], item["detail"], item["severity"], "LingYi 基础种子")
         for item in data["contraindications"]],
    )
