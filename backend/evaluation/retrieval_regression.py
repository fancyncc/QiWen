"""Run the known RAG failures on the current corpus without a generation model.

Live mode requires actual hybrid retrieval, CrossEncoder and MMR when applicable.
Keyword checks are development proxies; independent semantic acceptance stays blocked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .benchmark import load_jsonl, relevant, normalize
from .runtime import manifest, write_report

CASE_IDS = {"RAG-003", "RAG-006", "RAG-022", "RAG-022-MAI", "RAG-035"}


async def run(cases, *, live):
    from app.classics import retrieve_classics
    from app.query_plan import knowledge_plan
    from app import retrieval
    warmup = {"status": "not_needed"}
    if live:
        from app import local_ranking
        warmup = await local_ranking.warm_models()
        if warmup["status"] != "available":
            return {"status": "blocked", "count": len(cases), "passed": 0, "details": [], "warmup": warmup,
                    "reason": "cached_local_models_unavailable", "retrieval_mode": retrieval.RAG_MODE}
    details = []
    for case in cases:
        docs, trace = await retrieve_classics(knowledge_plan(case["query"]))
        text = "\n".join(normalize(row["retrieval"]["evidence"]) for row in docs)
        missing = [group for group in case.get("final_required_term_groups", []) if not any(normalize(t) in text for t in group)]
        proxy_passed = (any(relevant(row, case) for row in docs) and not missing if case.get("expected_evidence", True)
                        else not docs and trace.get("outcome") == "no_match")
        components = (trace.get("vector_status") == "success" and trace["rerank"]["status"] == "success" and
            trace.get("dependency_status") == "ok" and trace["mmr"]["status"] in {"success", "not_needed"}) if live else True
        details.append({"id": case["id"], "case": case, "passed": proxy_passed and components,
            "proxy_passed": proxy_passed, "required_components_passed": components,
            "missing_final_term_groups": missing, "trace": trace,
            "references": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                            "quote": row["retrieval"]["evidence"]} for row in docs]})
    return {"status": "completed", "count": len(details), "passed": sum(r["passed"] for r in details),
            "details": details, "retrieval_mode": retrieval.RAG_MODE, "warmup": warmup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "bm25"), default="live")
    parser.add_argument("--all-cases", action="store_true", help="Run the full golden retrieval regression including negative cases")
    parser.add_argument("--reports", type=Path, default=Path(__file__).with_name("reports"))
    args = parser.parse_args()
    if args.mode == "bm25":
        os.environ.update(QIWEN_RAG_MODE="bm25", QIWEN_RAG_RERANK_MODE="none")
    from app.db import init_db
    init_db()
    cases = load_jsonl("rag_golden.jsonl")
    if not args.all_cases:
        cases = [row for row in cases if row["id"] in CASE_IDS]
        alias_case = next(row for row in cases if row["id"] == "RAG-035")
        cases.append({**alias_case, "id": "RAG-035-NORMALIZED", "query": alias_case["query"].replace("黄耆", "黄芪")})
    result = asyncio.run(run(cases, live=args.mode == "live"))
    report = {"manifest": manifest("retrieval_" + args.mode), "results": result,
              "driver": "current_corpus_retrieval_component",
              "workloads": [{"details": [{"id": r["id"], "family": r["case"].get("family"), "mode": "component",
                  "retrieval": {"attempts": [r["trace"]]}} for r in result["details"]]}],
              "scope": "real_retrieval_component; development_proxy_not_expert_semantics",
              "semantic_acceptance": "blocked_pending_independent_review",
              "gate": {"status": "blocked" if result["status"] == "blocked" else
                       "passed" if result["count"] == result["passed"] else "failed"}}
    path = write_report(args.reports, "retrieval-regression", report)
    print(json.dumps({"report": str(path), "count": result["count"], "passed": result["passed"], "gate": report["gate"],
        "cases": [{"id": r["id"], "passed": r["passed"], "outcome": r["trace"]["outcome"]} for r in result["details"]]}, ensure_ascii=False))
    return 0 if report["gate"]["status"] == "passed" else 2 if report["gate"]["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
