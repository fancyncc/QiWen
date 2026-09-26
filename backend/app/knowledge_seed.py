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


def seed_structured_dosage(db: sqlite3.Connection) -> None:
    from .prescription import herb_id, parse_seed_dosage, structured_herb
    for column in ("herb_id", "structured_dosage", "source_ids", "verification_status"):
        _column(db, "herbs", column)
    _column(db, "formulas", "structured_herbs")
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    def evidence(identifier, name, raw, preparation, source, locator):
        dosage = parse_seed_dosage(raw)
        db.execute("""INSERT OR IGNORE INTO dosage_evidence(id,herb_id,herb_name,min_dose,max_dose,unit,raw_dosage,
            preparation,source,source_locator,data_version) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (identifier, herb_id(name), name, dosage["min"], dosage["max"], dosage["unit"], raw,
             preparation, source, locator, "LingYi-cb6b49d-seed-v1"))
        return structured_herb(name, raw_dosage=raw, preparation=preparation, source_ids=[identifier])
    for index, item in enumerate(data["herbs"]):
        identifier = "seed:" + herb_id(item["name"])
        herb = evidence(identifier, item["name"], item["dosage"], item["processing"], "LingYi 基础种子",
                        f"knowledge_seed.json#/herbs/{index}")
        db.execute("UPDATE herbs SET herb_id=?,structured_dosage=?,source_ids=?,verification_status='unverified' WHERE name=? AND structured_dosage=''",
                   (herb["herb_id"], json.dumps(herb["dosage"]), json.dumps(herb["source_ids"]), item["name"]))
    for index, item in enumerate(data["formulas"]):
        herbs = []
        for position, entry in enumerate(item["composition"]):
            identifier = "seed:formula:" + str(index) + ":" + str(position)
            herbs.append(evidence(identifier, entry["herb"], entry.get("dosage", ""), entry.get("preparation", ""),
                                  item["source"], f"knowledge_seed.json#/formulas/{index}/composition/{position}"))
        db.execute("UPDATE formulas SET structured_herbs=? WHERE name=? AND structured_herbs=''",
                   (json.dumps(herbs, ensure_ascii=False), item["name"]))
    for name, in db.execute("SELECT name FROM herbs WHERE herb_id=''").fetchall():
        herb = structured_herb(name)
        db.execute("UPDATE herbs SET herb_id=?,structured_dosage=?,source_ids='[]',verification_status='unverified' WHERE name=?",
                   (herb["herb_id"], json.dumps(herb["dosage"]), name))
