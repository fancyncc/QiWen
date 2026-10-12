"""Check family splits, duplicate prompts and label provenance before freezing data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).with_name("datasets")


def audit(folder: Path = DATA) -> dict:
    families, prompts, files, errors = defaultdict(set), defaultdict(list), {}, []
    for path in sorted(folder.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        files[path.name] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if len({r["id"] for r in rows}) != len(rows):
            errors.append(path.name + ":duplicate_id")
        for row in rows:
            if not all(row.get(k) for k in ("family", "split", "version", "label_status")):
                errors.append(path.name + ":" + row["id"] + ":missing_provenance")
                continue
            if row["split"] not in {"development", "tuning", "frozen"}:
                errors.append(row["id"] + ":invalid_split")
            families[row["family"]].add(row["split"])
            if "qrels" in row and (not isinstance(row["qrels"], dict) or any(type(g) is not int or not 0 <= g <= 3 for g in row["qrels"].values())):
                errors.append(row["id"] + ":invalid_qrels")
            if "passage_labels" in row:
                labels = row["passage_labels"]
                valid = isinstance(labels, dict) and all(isinstance(entries, list) and all(
                    isinstance(label, dict) and re.fullmatch(r"[a-f0-9]{64}", str(label.get("quote_sha256", ""))) and
                    re.fullmatch(r"[a-f0-9]{64}", str(label.get("body_sha256", ""))) and
                    type(label.get("relevance")) is int and 0 <= label["relevance"] <= 3 and
                    type(label.get("supports_question")) is bool and
                    (not label["supports_question"] or label["relevance"] > 0) and
                    isinstance(label.get("supported_claims", []), list) and
                    all(isinstance(claim, str) and claim for claim in label.get("supported_claims", [])) and
                    label.get("label_status") == "expert_reviewed" and
                    all(label.get(key) for key in ("reviewer", "reviewed_at", "rationale", "supported_scope"))
                    for label in entries) and len({label["quote_sha256"] for label in entries}) == len(entries)
                    for entries in labels.values())
                if not valid:
                    errors.append(row["id"] + ":invalid_passage_support_labels")
            if "expected_evidence" in row and type(row["expected_evidence"]) is not bool:
                errors.append(row["id"] + ":invalid_expected_evidence")
            if row["split"] == "frozen" and (row["label_status"] not in {"expert_reviewed", "engineering_reviewed"} or not row.get("reviewer") or not row.get("label_reason")):
                errors.append(row["id"] + ":frozen_without_review")
            texts = row.get("turns", []) + [s["input"] for s in row.get("steps", []) if "input" in s]
            texts += [row[k] for k in ("input", "question", "query") if k in row]
            for text in texts:
                key = re.sub(r"[\s，。！？、,.!?]", "", text)
                prompts[key].append({"file": path.name, "id": row["id"], "split": row["split"], "family": row["family"]})
    for family, splits in families.items():
        if len(splits) > 1:
            errors.append(family + ":family_split_leakage")
    duplicates = [rows for rows in prompts.values() if len({(r["file"], r["id"]) for r in rows}) > 1]
    for rows in duplicates:
        if len({r["split"] for r in rows}) > 1:
            errors.append(rows[0]["id"] + ":prompt_split_leakage")
    return {"status": "failed" if errors else "passed", "errors": errors, "files": files,
            "unique_families": len(families), "duplicate_prompt_groups": duplicates,
            "similarity_scope": "exact_normalized_prompts; paraphrase_families_require_manual_review",
            "frozen_ready": bool(families) and all(s == {"frozen"} for s in families.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.folder)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": report["errors"], "families": report["unique_families"],
                      "duplicate_groups": len(report["duplicate_prompt_groups"]), "frozen_ready": report["frozen_ready"]}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
