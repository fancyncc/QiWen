"""Rebuild or verify the bundled classics data from preserved sources.

Run from backend: python -m scripts.rebuild_classics [--write]
The default is a read-only comparison; --write replaces the derived JSON files.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from data_pipeline.ingest import TCM_FILES, chunk_all_books, save_chunks_json

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "resources" / "classics_source"
CHUNK_DIR = ROOT / "resources" / "classics_segments"
KNOWLEDGE_SOURCE = ROOT / "data_pipeline" / "seed_knowledge.py"
KNOWLEDGE_FILE = ROOT / "resources" / "knowledge_seed.json"
FIELDS = {
    "Herb": ("name", "aliases", "nature_flavor", "meridians", "efficacy", "indications", "dosage", "processing", "contraindications"),
    "FormulaComponent": ("herb", "dosage"),
    "Formula": ("name", "source", "composition", "indication", "modifications", "contraindications", "category"),
    "Contraindication": ("herb", "type", "detail", "severity"),
}


def _literal(node: ast.AST):
    """Evaluate only source literals and the four known data constructors."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FIELDS:
        if len(node.args) > len(FIELDS[node.func.id]) or any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("Unexpected constructor arguments")
        values = {field: _literal(arg) for field, arg in zip(FIELDS[node.func.id], node.args)}
        for keyword in node.keywords:
            if keyword.arg in values:
                raise ValueError(f"Duplicate field: {keyword.arg}")
            values[keyword.arg] = _literal(keyword.value)
        unknown = values.keys() - FIELDS[node.func.id]
        if unknown:
            raise ValueError(f"Unknown {node.func.id} fields: {unknown}")
        return values
    raise ValueError(f"Unexpected expression in knowledge seed: {ast.dump(node)[:120]}")


def rebuild_knowledge() -> dict[str, list[dict]]:
    """Read upstream seed declarations without importing its unavailable runtime."""
    tree = ast.parse(KNOWLEDGE_SOURCE.read_text(encoding="utf-8"))
    values: dict[str, list[dict]] = {}
    targets = {"HERBS", "FORMULAS", "CONTRAINDICATIONS"}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in targets:
            values[node.target.id] = _literal(node.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "extend"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "CONTRAINDICATIONS"):
                if len(call.args) != 1 or call.keywords:
                    raise ValueError("Unexpected contraindication extension")
                values["CONTRAINDICATIONS"].extend(_literal(call.args[0]))
    if set(values) != targets:
        raise ValueError(f"Missing knowledge sets: {targets - values.keys()}")
    return {
        "herbs": values["HERBS"],
        "formulas": values["FORMULAS"],
        "contraindications": values["CONTRAINDICATIONS"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the bundled JSON files")
    args = parser.parse_args()
    missing = [name for name in TCM_FILES.values() if not (RAW_DIR / name).is_file()]
    if missing:
        parser.error(f"Missing original texts: {', '.join(missing)}")

    chunks = chunk_all_books(str(RAW_DIR))
    knowledge = rebuild_knowledge()
    chunk_json = {
        book: [{"id": item.id, "content": item.content, "metadata": item.metadata} for item in items]
        for book, items in chunks.items()
    }
    if args.write:
        save_chunks_json(chunks, str(CHUNK_DIR))
        KNOWLEDGE_FILE.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        for book, expected in chunk_json.items():
            actual = json.loads((CHUNK_DIR / f"{book}.json").read_text(encoding="utf-8"))
            if actual != expected:
                raise SystemExit(f"Derived chunks differ: {book}")
        if json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8")) != knowledge:
            raise SystemExit("Derived knowledge differs")
    print("Verified" if not args.write else "Rebuilt", sum(map(len, chunks.values())),
          "chunks in", len(chunks), "books,", len(knowledge["herbs"]), "herbs,",
          len(knowledge["formulas"]), "formulas,", len(knowledge["contraindications"]), "cautions")


if __name__ == "__main__":
    main()
