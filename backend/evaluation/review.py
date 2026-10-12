"""Human review queues, declared release policy and comparable version reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def policy_fingerprint(path: Path | None) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    except OSError:
        return None


def review_queue(report: dict) -> list[dict]:
    rows = []
    for dimension in ("quality", "faithfulness", "safety"):
        result = report.get("results", {}).get(dimension, {})
        details = result.get("e2e", {}).get("details", []) if dimension == "safety" else result.get("details", [])
        # Stable random ordering selects ceil(20%) of ordinary answers, per dimension.
        ordinary = [r for r in details if dimension != "safety" and r.get("status") == "completed" and r.get("passed", True)]
        ranked = sorted(ordinary, key=lambda r: hashlib.sha256((r["case"]["id"] + r.get("mode", "fixed")).encode()).hexdigest())
        sample = {id(r) for r in ranked[:math.ceil(len(ranked) * .2)]}
        for row in details:
            reason = "semantic_review_required" if row.get("semantic_review") == "pending" else "high_risk" if dimension == "safety" or row.get("case", {}).get("high_risk") else "failure_or_disagreement" if row.get("status") != "completed" or not row.get("passed", True) or row.get("judge", {}).get("contradiction") else "random_sample" if id(row) in sample else None
            if reason:
                identity = dimension + ":" + row["case"]["id"] + ":" + row.get("mode", "fixed")
                rows.append({"review_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                             "dimension": dimension, "case_id": row["case"]["id"], "reason": reason,
                             "input": row["case"].get("input", row["case"].get("question")),
                             "answer": row.get("response", {}).get("data", {}).get("response", ""),
                             "references": row.get("references", []), "status": "pending",
                             "supported_claims": [], "missing_conditions": [], "contradicted_claims": []})
    return rows


def metric_at(report: dict, path: str):
    value = report["results"]
    for part in path.split("."):
        value = value[part]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Metric is unknown or nonnumeric")
    return value


def release_gate(report: dict, policy_path: Path, review_path: Path | None,
                 deployment_path: Path | None = None) -> dict:
    incomplete, failed = [], []
    if report["gate"]["status"] != "passed":
        return {**report["gate"], "scope": "release"}
    if report["mode"] != "live_full" or report.get("limit") is not None:
        incomplete.append("full_live_unsliced_run_required")
    if set(report.get("modes", [])) != {"fixed", "multi"}:
        incomplete.append("both_application_modes_required")
    if set(report["selected"]) != {"rag", "faithfulness", "quality", "safety", "router", "performance", "scenarios"}:
        incomplete.append("all_dimensions_required")
    if report["results"].get("rag", {}).get("label_scope") != "expert_qrels":
        incomplete.append("expert_reviewed_passage_labels_required")
    evidence = report["results"].get("rag", {}).get("evidence_contract", {})
    if not evidence.get("positive_count") or not evidence.get("negative_count") or evidence.get("unlabeled", 0):
        incomplete.append("positive_and_negative_evidence_acceptance_required")
    if evidence.get("failed", 0):
        failed.append("rag:evidence_contract_failure")
    if not report.get("dataset_audit", {}).get("frozen_ready"):
        incomplete.append("reviewed_frozen_dataset_required")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("Policy must be an object")
        if not report.get("predeclared_policy_sha256") or policy_fingerprint(policy_path) != report["predeclared_policy_sha256"]:
            incomplete.append("policy_not_declared_before_run_or_changed")
        if not policy.get("version") or policy.get("behavior_version") != report["manifest"]["behavior_version"]:
            raise ValueError("Policy version mismatch")
        thresholds = policy["thresholds"]
        if not isinstance(thresholds, list) or not all(isinstance(t, dict) and isinstance(t.get("metric"), str) for t in thresholds):
            raise ValueError("Invalid threshold list")
        dimensions = {t["metric"].split(".")[0] for t in thresholds}
        if not {"quality", "faithfulness", "performance"} <= dimensions:
            raise ValueError("Quality, faithfulness and performance targets required")
        if not {"rag.final_metrics.precision", "rag.final_metrics.recall"} <= {t["metric"] for t in thresholds}:
            raise ValueError("Final evidence relevance and recall targets required")
        for threshold in thresholds:
            bound, op = threshold["value"], threshold["operator"]
            if type(bound) not in (int, float) or not math.isfinite(bound) or op not in {"min", "max"}:
                raise ValueError("Invalid threshold")
            actual = metric_at(report, threshold["metric"])
            if (op == "min" and actual < bound) or (op == "max" and actual > bound):
                failed.append(threshold["metric"])
    except (OSError, ValueError, KeyError, TypeError):
        incomplete.append("invalid_or_incomplete_policy")
        policy = {}
    try:
        if deployment_path is None:
            raise ValueError("Deployment report missing")
        deployed = json.loads(deployment_path.read_text(encoding="utf-8"))
        if (deployed.get("driver") != "real_http_sse_frontend_proxy" or deployed.get("limit") is not None
                or deployed.get("service", {}).get("real_model_enabled") is not True
                or deployed.get("service", {}).get("hybrid_retrieval_enabled") is not True
                or deployed.get("manifest", {}).get("workspace_sha256") != report["manifest"]["workspace_sha256"]):
            raise ValueError("Deployment must use the same code and complete workload")
        if deployed.get("gate", {}).get("status") != "passed":
            failed.append("deployment:request_contract_failure")
        if deployed.get("evidence_gate", {}).get("status") != "passed":
            incomplete.append("deployment:expected_evidence_checks_required")
        if deployed.get("service", {}).get("retrieval_readiness", {}).get("status") != "ready":
            incomplete.append("deployment:retrieval_prerequisites_or_recent_execution_unready")
        targets = policy.get("deployment_targets", [])
        if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
            raise ValueError("Invalid deployment target list")
        if not targets or {t["mode"] for t in targets} != {"fixed", "multi"} or not any(t["scenario"] == "diagnose" for t in targets):
            raise ValueError("Predeclared deployment targets required")
        for target in targets:
            level, mode, scenario = target["concurrency"], target["mode"], target["scenario"]
            bound, minimum = target["max_final_p95_s"], target["min_success_rate"]
            if (type(level) is not int or level < 1 or mode not in {"fixed", "multi"} or scenario not in {"chat", "diagnose"}
                    or type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0
                    or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1):
                raise ValueError("Invalid deployment target")
            workload = next(w for w in deployed["workloads"] if w["concurrency"] == level)
            if workload.get("capacity_sample_ready") is not True:
                incomplete.append(f"deployment:{level}:insufficient_observed_load")
            required_hold = target.get("min_target_overlap_s", 1)
            if type(required_hold) not in (int, float) or not math.isfinite(required_hold) or required_hold < 1:
                raise ValueError("Invalid target overlap duration")
            if workload.get("observed_load", {}).get("time_at_target_s", 0) < required_hold:
                incomplete.append(f"deployment:{level}:target_overlap_too_short")
            rows = [r for r in workload["details"] if r["mode"] == mode and r["scenario"] == scenario]
            value = workload["latency_s"][mode + ":" + scenario]["final_visible_s"]["p95"]
            if not rows or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Missing deployment measurement")
            if value > bound or sum(r["passed"] for r in rows) / len(rows) < minimum:
                failed.append(f"deployment:{level}:{mode}:{scenario}")
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        incomplete.append("missing_invalid_or_incomplete_deployment_acceptance")
    try:
        if review_path is None:
            raise ValueError("Review missing")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("run_id") != report["manifest"]["run_id"] or review.get("workspace_sha256") != report["manifest"]["workspace_sha256"]:
            raise ValueError("Review belongs to another run")
        items = review["items"]
        if len({i["review_id"] for i in items}) != len(items):
            raise ValueError("Duplicate review IDs")
        indexed = {i["review_id"]: i for i in items}
        for row in report["review_queue"]:
            item = indexed.get(row["review_id"], {})
            if item.get("status") not in {"approved", "rejected"} or not item.get("reviewer") or not item.get("reason"):
                incomplete.append("review:" + row["review_id"])
            if item.get("status") == "rejected":
                failed.append("review:" + row["review_id"])
    except (OSError, ValueError, KeyError, TypeError):
        incomplete.append("missing_or_invalid_human_review")
    return {"status": "failed" if failed else "blocked" if incomplete else "passed", "scope": "release",
            "failures": failed, "incomplete": incomplete}


def compare(before: dict, after: dict) -> dict:
    reasons = []
    for key in ("profile", "behavior_version", "corpus_sha256", "public_settings", "roles", "provider_fingerprints"):
        if before["manifest"].get(key) != after["manifest"].get(key):
            reasons.append("manifest:" + key)
    for key in ("selected", "limit", "modes"):
        if before.get(key) != after.get(key):
            reasons.append(key)
    for path in before["manifest"]["files"].keys() | after["manifest"]["files"].keys():
        if ("/datasets/" in path or path.startswith("backend/resources/")) and before["manifest"]["files"].get(path) != after["manifest"]["files"].get(path):
            reasons.append(("dataset:" if "/datasets/" in path else "resource:") + path)
    metrics = {}
    for dimension, row in after["results"].items():
        for metric, value in row.get("metrics", {}).items():
            old = before["results"].get(dimension, {}).get("metrics", {}).get(metric)
            if type(value) in (int, float) and type(old) in (int, float):
                metrics[dimension + "." + metric] = {"before": old, "after": value, "delta": value - old}
    return {"status": "incomparable" if reasons else "comparable", "reasons": reasons, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--deployment-report", type=Path)
    parser.add_argument("--export-review", type=Path)
    args = parser.parse_args()
    if not args.export_review and not args.compare and not args.policy:
        parser.error("select --export-review, --compare or --policy")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.export_review:
        queue = {"run_id": report["manifest"]["run_id"], "workspace_sha256": report["manifest"]["workspace_sha256"],
                 "instructions": "Blind review; add reviewer, reason and approved/rejected status. No clinical efficacy claim.",
                 "items": report["review_queue"]}
        args.export_review.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    result = compare(json.loads(args.compare.read_text(encoding="utf-8")), report) if args.compare else release_gate(report, args.policy, args.review, args.deployment_report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"passed", "comparable"} else 1 if result["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
