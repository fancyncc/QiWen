"""Offline threshold proposals, frozen development/validation labels required.

This tool never changes runtime settings or interprets sigmoid as clinical confidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from .evidence_audit import fingerprint


def metrics(rows, threshold):
    tp = sum(row["positive"] and row["score"] >= threshold for row in rows)
    fp = sum(not row["positive"] and row["score"] >= threshold for row in rows)
    fn = sum(row["positive"] and row["score"] < threshold for row in rows)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "hard_negative_admissions": sum(row["hard_negative"] and row["score"] >= threshold for row in rows)}


def calibrate(audit, labels, min_precision=.95, min_per_class=10):
    if not 0 < min_precision <= 1 or min_per_class < 1:
        raise ValueError("Invalid calibration requirements")
    tasks = {task["task_id"]: task for task in audit["tasks"]}
    if len(tasks) != len(audit["tasks"]):
        raise ValueError("Duplicate audit task")
    for task in tasks.values():
        if task["binding_sha256"] != fingerprint({k: v for k, v in task.items() if k != "binding_sha256"}):
            raise ValueError("Audit source/score binding changed")
    seen, rows, reasons = set(), [], []
    for label in labels:
        key = label["task_id"]
        if key in seen or key not in tasks:
            raise ValueError("Duplicate or unknown passage label")
        seen.add(key)
        task = tasks[key]
        if label.get("binding_sha256") != task["binding_sha256"]:
            raise ValueError("Label source/score binding mismatch")
        if (label.get("label_status") != "expert_reviewed" or not label.get("reviewer")
                or not label.get("reviewed_at") or not label.get("rationale")
                or not label.get("supported_scope") or label.get("split") not in {"development", "validation"}
                or any(type(label.get(field)) is not bool for field in
                       ("passage_relevant", "supports_question", "hard_negative"))):
            reasons.append("independent_review_incomplete")
            continue
        if label["supports_question"] and not label["passage_relevant"]:
            raise ValueError("Supporting evidence must be relevant")
        positive = label["passage_relevant"] and label["supports_question"]
        if positive and label["hard_negative"]:
            raise ValueError("Positive evidence cannot be a hard negative")
        # Unscored recalled passages must be reviewed for coverage, but cannot calibrate CE.
        score = task.get("score")
        if score is None:
            continue
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid passage score")
        rows.append({**task, "split": label["split"], "hard_negative": label["hard_negative"], "positive": positive})
    if seen != set(tasks):
        reasons.append("passage_labels_missing")
    # Assign related queries/fixed/multi duplicates to one split, before threshold selection.
    family_splits = {}
    for row in rows:
        family_splits.setdefault(row["family"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in family_splits.values()):
        reasons.append("query_family_split_leakage")
    groups = {}
    for kind in sorted({row["query_kind"] for row in rows}):
        partitions = {split: [row for row in rows if row["query_kind"] == kind and row["split"] == split]
                      for split in ("development", "validation")}
        group_reasons = []
        if len({fingerprint(row.get("scoring_config")) for row in rows if row["query_kind"] == kind}) != 1:
            group_reasons.append("mixed_scoring_configuration")
        for split, partition in partitions.items():
            # Duplicate modes/repeats are one labeled query/passage sample, not extra evidence.
            unique = {}
            for row in partition:
                sample_key = (row["family"], row["doc_id"], row["window_sha256"])
                if sample_key in unique and any(unique[sample_key][field] != row[field] for field in ("positive", "hard_negative")):
                    raise ValueError("Conflicting duplicate passage labels")
                unique.setdefault(sample_key, row)
            partitions[split] = list(unique.values())
            for positive in (True, False):
                if sum(row["positive"] == positive for row in unique.values()) < min_per_class:
                    group_reasons.append(f"{split}:insufficient_{'positive' if positive else 'negative'}_samples")
            if not any(row["hard_negative"] for row in unique.values()):
                group_reasons.append(f"{split}:hard_negative_missing")
        grid = sorted({.7, *[row["score"] for row in partitions["development"]]})
        trials = [metrics(partitions["development"], threshold) for threshold in grid]
        eligible = [trial for trial in trials if trial["precision"] >= min_precision and trial["tp"]
                    and not trial["hard_negative_admissions"]]
        proposal = max(eligible, key=lambda trial: (trial["recall"], trial["precision"], trial["threshold"])) if eligible else None
        validation = metrics(partitions["validation"], proposal["threshold"]) if proposal else None
        if not proposal:
            group_reasons.append("no_development_threshold_meets_precision_requirement")
        if validation and (validation["precision"] < min_precision or validation["hard_negative_admissions"] or not validation["tp"]):
            group_reasons.append("validation_failed")
        groups[kind] = {"status": "blocked" if group_reasons else "proposal_ready_for_review",
                        "reasons": group_reasons, "development": proposal, "validation": validation,
                        "current_threshold_validation": metrics(partitions["validation"], .7)}
    if not groups:
        reasons.append("no_reviewed_scored_samples")
    return {"status": "blocked" if reasons or any(g["status"] == "blocked" for g in groups.values())
            else "proposal_ready_for_review", "reasons": sorted(set(reasons)), "groups": groups,
            "runtime_threshold_changed": False, "scope": "independent_passage_support_not_diagnostic_confidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-precision", type=float, default=.95)
    parser.add_argument("--min-per-class", type=int, default=10)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = calibrate(audit, labels, args.min_precision, args.min_per_class)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "reasons": report["reasons"], "output": args.output}, ensure_ascii=False))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
