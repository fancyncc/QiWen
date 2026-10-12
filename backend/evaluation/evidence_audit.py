"""Export source-bound passage review tasks from a synthetic deployment report.

No relevance labels are inferred from keyword hits or reranker scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit_report(report):
    tasks, cases = [], []
    for workload_index, workload in enumerate(report.get("workloads", [])):
        for job_index, job in enumerate(workload.get("details", [])):
            # Inspect every turn: the final follow-up is not the entire conversation.
            turns = job.get("turn_results") or [job]
            for turn_index, turn in enumerate(turns):
                retrieval = turn.get("retrieval") or job.get("retrieval") or {}
                for attempt_index, attempt in enumerate(retrieval.get("attempts", [])):
                    case_key = f'{workload_index}:{job_index}:{turn_index}:{attempt_index}'
                    recalled = {row["id"]: row for row in attempt.get("recall_audit", [])}
                    ranked = {row["id"]: row for row in attempt.get("rerank", {}).get("ranked", [])}
                    windows = {row["id"]: row for row in attempt.get("rerank", {}).get("input_windows", [])}
                    cases.append({"case_key": case_key, "case_id": job["id"], "mode": job.get("mode"),
                                  "outcome": attempt.get("outcome"), "query_plan": attempt.get("query_plan", {}),
                                  "source_manifest_sha256": attempt.get("source_manifest_sha256"),
                                  "recall_count": len(recalled), "scored_count": len(ranked),
                                  "label_status": "pending_independent_review"})
                    for doc_id in dict.fromkeys([*recalled, *ranked]):
                        score = ranked.get(doc_id, {})
                        window = windows.get(doc_id, {})
                        quote = score.get("evidence", window.get("quote"))
                        reason = ("excluded_from_rerank_pool" if not recalled.get(doc_id, {}).get("in_rerank_pool", True)
                                  else "rerank_not_completed" if not score else score.get("rejection_reason", ""))
                        family = {"kind": attempt.get("evidence_kind"),
                                  "query": attempt.get("query_plan", {}).get("semantic_query"),
                                  "primary": sorted(attempt.get("query_plan", {}).get("primary", [])),
                                  "source_hints": attempt.get("query_plan", {}).get("source_hints", []),
                                  "constraints": {k: attempt.get("query_plan", {}).get(k) for k in
                                      ("required_term_groups", "required_entities", "action_polarity")}}
                        task = {"task_id": f"{case_key}:{doc_id}", "case_key": case_key, "case_id": job["id"],
                                "family": job.get("family") or fingerprint(family), "query_kind": attempt.get("evidence_kind", "unknown"),
                                "query_plan": attempt.get("query_plan", {}), "doc_id": doc_id,
                                "scoring_config": {key: attempt.get("rerank", {}).get(key)
                                                   for key in ("model", "revision", "max_length")},
                                "title": window.get("title"), "chapter": window.get("chapter"),
                                "body_sha256": score.get("window", {}).get("body_sha256", window.get("body_sha256")),
                                "window_sha256": fingerprint(quote) if quote is not None else None,
                                "quote": quote, "score": score.get("score"), "rejection_reason": reason,
                                "recall": recalled.get(doc_id), "token_diagnostic": score.get("token_diagnostic"),
                                "selected": score.get("selected", False)}
                        task["binding_sha256"] = fingerprint(task)
                        tasks.append(task)
    return {"version": 1, "deployment_sha256": fingerprint(report), "cases": cases, "tasks": tasks,
            "calibration_status": "blocked_pending_independent_passage_labels",
            "scope": "stage_diagnostics_not_clinical_validation"}


def deduplicate_tasks(audit):
    """Merge identical review inputs, retaining every original run binding.

    Different windows, questions, sources, model settings or scores remain separate.
    Unscored recall tasks never merge with scored passage tasks.
    """
    groups = {}
    for task in audit["tasks"]:
        query = {key: value for key, value in task.get("query_plan", {}).items() if key != "attempt"}
        key = fingerprint({"family": task["family"], "query": query, "doc_id": task["doc_id"],
            "body_sha256": task.get("body_sha256"), "quote": task.get("quote"),
            "review_text": task.get("review_text"), "scoring_config": task.get("scoring_config"), "score": task.get("score")})
        occurrence = {"task_id": task["task_id"], "binding_sha256": task["binding_sha256"],
                      "case_key": task["case_key"], "selected": task.get("selected"),
                      "rejection_reason": task.get("rejection_reason")}
        if key not in groups:
            groups[key] = {**task, "task_id": "review:" + key, "occurrences": []}
        groups[key]["occurrences"].append(occurrence)
    tasks = []
    for task in groups.values():
        task["binding_sha256"] = fingerprint({k: v for k, v in task.items() if k != "binding_sha256"})
        tasks.append(task)
    return {**audit, "version": 2, "raw_task_count": len(audit["tasks"]), "tasks": tasks,
            "unique_task_count": len(tasks), "duplicate_task_count": len(audit["tasks"]) - len(tasks)}


def locate_gold(gold_id, trace, source_ids, source_scope_ids=None):
    """Distinguish corpus, scope, recall, pool, gate and final selection losses."""
    if gold_id not in source_ids:
        return "source_absent"
    if source_scope_ids is not None and gold_id not in source_scope_ids:
        return "outside_requested_source_scope"
    recalled = next((row for row in trace.get("recall_audit", []) if row["id"] == gold_id), None)
    if recalled is None:
        return "recall_miss"
    if not recalled["in_rerank_pool"]:
        return "pool_exclusion"
    ranked = next((row for row in trace.get("rerank", {}).get("ranked", []) if row["id"] == gold_id), None)
    if ranked is None:
        return "rerank_not_completed"
    if not ranked["accepted"]:
        return "relevance_gate_rejection"
    return "selected" if ranked.get("selected") else "post_gate_exclusion"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-current-sources", action="store_true", help="Run on the deployed backend to freeze its public sources")
    parser.add_argument("--gold-labels", help="Independently reviewed JSONL with case_id and positive source doc_id")
    parser.add_argument("--deduplicate", action="store_true", help="One review per identical question/source/window/score; retain run mappings")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    audit = audit_report(report)
    if args.with_current_sources:
        from app.db import all_rows
        from app.evidence_sources import source_lookup
        from app.retrieval import source_fingerprint
        from app.evidence_sources import matching_concepts, concept_rows
        sources = list(source_lookup(all_rows("SELECT id,title,chapter,body FROM corpus")).values())
        manifest_sha = source_fingerprint(sources)
        if any(case["source_manifest_sha256"] != manifest_sha for case in audit["cases"]):
            raise ValueError("Deployed source snapshot differs from report; rerun deployment evaluation")
        audit["source_snapshot"] = sources
        audit["source_manifest_sha256"] = manifest_sha
        lookup = {row["id"]: row for row in sources}
        for task in audit["tasks"]:
            row = lookup.get(task["doc_id"])
            if row and task.get("quote") is None:
                task["review_text"] = row["body"]
                task["title"], task["chapter"] = row["title"], row["chapter"]
                task["body_sha256"] = hashlib.sha256(row["body"].encode()).hexdigest()
                task["review_scope"] = "unscored_full_source_not_encoder_window"
                task["binding_sha256"] = fingerprint({k: v for k, v in task.items() if k != "binding_sha256"})
    if args.gold_labels:
        if not args.with_current_sources:
            raise ValueError("Gold-stage diagnostics require a matching source snapshot")
        from app.retrieval import source_fingerprint
        gold = [json.loads(line) for line in Path(args.gold_labels).read_text(encoding="utf-8").splitlines() if line.strip()]
        audit["gold_stage_diagnostics"] = []
        lookup = {row["id"]: row for row in sources}
        for label in gold:
            if (label.get("label_status") != "expert_reviewed" or not label.get("reviewer")
                    or not label.get("reviewed_at") or not label.get("rationale") or not label.get("supported_scope")):
                raise ValueError("Gold labels must be independently reviewed")
            for workload in report.get("workloads", []):
                for job in workload.get("details", []):
                    if job["id"] != label["case_id"]:
                        continue
                    for turn in job.get("turn_results") or [job]:
                        for attempt in (turn.get("retrieval") or job.get("retrieval") or {}).get("attempts", []):
                            plan = attempt.get("query_plan", {})
                            scope = matching_concepts(plan) or [row for row in sources if row["id"] not in
                                    {r["id"] for r in concept_rows()}]
                            if plan.get("source_hints"):
                                scope = [row for row in scope if any(hint in row["title"] or hint in row["chapter"]
                                                                    for hint in plan["source_hints"])]
                            if source_fingerprint(scope) != attempt.get("source_scope_sha256"):
                                raise ValueError("Source scope does not match retrieval report")
                            audit["gold_stage_diagnostics"].append({"case_id": job["id"], "mode": job.get("mode"),
                                "doc_id": label["doc_id"], "stage": locate_gold(label["doc_id"], attempt, lookup,
                                                                                {row["id"] for row in scope})})
    if args.deduplicate:
        audit = deduplicate_tasks(audit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = output.with_suffix(".labels.jsonl")
    if labels.exists() or output.exists():
        raise ValueError("Refusing to overwrite independent labels")
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    labels.write_text("".join(json.dumps({"task_id": task["task_id"], "binding_sha256": task["binding_sha256"],
        "split": None, "label_status": "pending", "reviewer": None, "reviewed_at": None,
        "passage_relevant": None, "supports_question": None, "hard_negative": None,
        "supported_scope": None, "supported_claims": [], "missing_conditions": [],
        "contradicted_claims": [], "rationale": None}, ensure_ascii=False) + "\n" for task in audit["tasks"]), encoding="utf-8")
    print(json.dumps({"audit": str(output), "labels": str(labels), "tasks": len(audit["tasks"]),
                      "calibration_status": audit["calibration_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
