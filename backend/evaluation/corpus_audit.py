"""Audit bundled public corpus cleaning and similar passages without deleting variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from app.corpus_seed import BOOKS, CHUNK_DIR, RAW_CHUNK_DIR
from data_pipeline.cleaning import clean_source_text


def audit():
    sources, changed, empty = [], [], []
    raw_dir = RAW_CHUNK_DIR if RAW_CHUNK_DIR.exists() else CHUNK_DIR
    for book in BOOKS:
        for chunk in json.loads((raw_dir / f"{book}.json").read_text(encoding="utf-8")):
            raw = chunk["content"].strip()
            body, provenance = clean_source_text(raw, book)
            if not body:
                empty.append({"book": book, "raw_source_chunk_id": chunk["id"]})
                continue
            if body != raw:
                changed.append({"book": book, "raw_source_chunk_id": chunk["id"], **provenance})
    for book in BOOKS:
        for chunk in json.loads((CHUNK_DIR / f"{book}.json").read_text(encoding="utf-8")):
            body = chunk["content"].strip()
            key = hashlib.sha256(f"{book}\n{body}".encode()).hexdigest()
            normalized = re.sub(r"[\W_]+", "", body)
            sources.append({"id": key, "book": book, "length": len(normalized),
                            "shingles": {normalized[i:i + 5] for i in range(max(0, len(normalized) - 4))}})
    duplicates = []
    by_book = defaultdict(list)
    for row in sources:
        by_book[row["book"]].append(row)
    for book, rows in by_book.items():
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if min(a["length"], b["length"]) / max(a["length"], b["length"], 1) < .85:
                    continue
                union = a["shingles"] | b["shingles"]
                similarity = len(a["shingles"] & b["shingles"]) / len(union) if union else 0
                if similarity >= .85:
                    duplicates.append({"book": book, "ids": [a["id"], b["id"]],
                                       "shingle_jaccard": round(similarity, 4), "status": "pending_source_variant_review"})
    return {"version": 1, "scope": "bundled_public_sources_only", "raw_sources_preserved": True,
            "canonical_chunk_count": len(sources),
            "changed_chunk_count": len(changed), "empty_chunk_count": len(empty),
            "changed_chunks": changed, "empty_chunks": empty, "near_duplicate_count": len(duplicates),
            "near_duplicates": duplicates, "automatic_near_duplicate_deletion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a corpus audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), **{k: report[k] for k in
        ("changed_chunk_count", "empty_chunk_count", "near_duplicate_count")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
