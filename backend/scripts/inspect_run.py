"""Read-only, allowlisted run diagnostics; never print prompts, profiles or config."""
from __future__ import annotations

import argparse
import json
import re

from app.db import connection


def label(value) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,80}", value) else "unknown"


def number(value):
    return value if isinstance(value, (int, float)) else None


def failure(value):
    value = value if isinstance(value, dict) else {}
    category = value.get("category")
    result = {"category": category if category in {"upstream_unavailable", "rate_limited", "authentication", "timeout", "connection", "invalid_output", "http_error", "unavailable"} else "unknown"}
    status = value.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599:
        result["http_status"] = status
    return result


def validation(value):
    value = value if isinstance(value, dict) else {}
    fields = {"decision", "risk_categories", "reason", "required_questions", "observations", "category", "entity", "subject", "assertion", "temporal_scope", "quote"}
    types = {"json_invalid", "missing", "extra_forbidden", "literal_error", "list_type", "string_type", "string_too_short", "string_too_long", "too_long", "model_type"}
    issues = [{"path": [part if isinstance(part, int) or part in fields else "unknown_field" for part in item.get("path", [])[:8]],
               "type": item.get("type") if item.get("type") in types else "unknown"}
              for item in value.get("issues", [])[:8] if isinstance(item, dict)]
    return {"kind": value.get("kind") if value.get("kind") in {"json", "schema", "response_structure"} else "unknown", "issues": issues}


def summarize(row: dict) -> dict:
    trace = json.loads(row.get("telemetry") or "{}")
    agents = trace.get("agents", {})
    artifacts = agents.get("artifacts", {})
    stages = [{"stage": label(e.get("stage")), "status": label(e.get("status")),
               "elapsed_ms": number(e.get("elapsed_ms"))} for e in trace.get("events", [])]
    tools = []
    for agent, entries in agents.get("traces", {}).items():
        for entry in entries:
            if not entry.get("tool"):
                continue
            result = entry.get("result") or {}
            tools.append({"agent": label(agent), "tool": label(entry["tool"]),
                          "status": label(entry.get("status")), "result_status": label(result.get("status")),
                          "formula_count": len(result.get("formulas", [])),
                          "document_count": len(result.get("documents", [])),
                          "herb_found": result.get("found") is True})
    findings = []
    if "preflight_unavailable" in trace.get("preflight", {}).get("risk_categories", []):
        findings.append("PREFLIGHT_SERVICE_UNAVAILABLE")
    if any(call.get("status") == "failed" for call in trace.get("model_calls", [])):
        findings.append("MODEL_CALL_FAILED")
    formula = artifacts.get("formula", {})
    if formula.get("status") == "completed" and not formula.get("plan"):
        findings.append("FORMULA_COMPLETED_WITHOUT_PERSONAL_PLAN")
    if (formula.get("comparison") and any(t["formula_count"] for t in tools)
            and "herb" not in artifacts):
        findings.append("LITERATURE_AVAILABLE_BUT_HERB_STAGE_NOT_RUN")
    if any(t["tool"] == "search_classics" and t["result_status"] == "no_match" for t in tools):
        findings.append("CLASSICS_SEARCH_NO_MATCH")
    if any(t["status"] != "success" for t in tools):
        findings.append("TOOL_CALL_FAILED")
    if (artifacts or any(event.get("stage") == "inquiry" for event in trace.get("events", []))) and not any(f.get("duration_text") for f in trace.get("inquiry", {}).get("facts", [])):
        findings.append("NO_DURATION_IN_COMMITTED_SYMPTOM_FACTS")
    if row["status"] in {"cancelled", "failed"}:
        findings.append("RUN_ENDED_WITHOUT_FINAL_ANSWER")
    return {"run_id": label(row["id"]), "thread_id": label(row["thread_id"]), "status": label(row["status"]),
            "elapsed_ms": number(trace.get("elapsedMs")), "findings": findings, "stages": stages,
            "cancel_reason": label(trace.get("cancel_reason")),
            "preflight": {"decision": label(trace.get("preflight", {}).get("decision")),
                          "failure": failure(trace.get("preflight", {}).get("failure")),
                          "analysis_allowed": trace.get("preflight", {}).get("analysis_allowed") is True,
                          "pending_safety_question_count": len(trace.get("preflight", {}).get("question_keys", []))},
            "artifacts": {label(k): {"status": label(v.get("status")), "has_plan": bool(v.get("plan")),
                                      "has_comparison": bool(v.get("comparison")),
                                      "has_question": bool(v.get("question")),
                                      "uncertainty_count": len(v.get("uncertainties", []))}
                          for k, v in artifacts.items()}, "tools": tools,
            "model_calls": [{"node": label(c.get("node")), "status": label(c.get("status")),
                             "elapsed_ms": number(c.get("elapsed_ms")), "retries": number(c.get("retries")),
                             "failure": failure(c.get("failure")),
                             "validation": validation(c.get("validation")),
                             "attempts": [{"status": label(a.get("status")), "failure": failure(a.get("failure"))} for a in c.get("attempts", [])]}
                            for c in trace.get("model_calls", [])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--run-id")
    scope.add_argument("--thread-id")
    args = parser.parse_args()
    where, params = ("WHERE id=?", (args.run_id,)) if args.run_id else (
        ("WHERE thread_id=?", (args.thread_id,)) if args.thread_id else ("", ()))
    with connection() as db:
        row = db.execute(f"SELECT id,thread_id,status,telemetry FROM chat_runs {where} ORDER BY created_at DESC LIMIT 1", params).fetchone()
    if not row:
        raise SystemExit("No matching run found.")
    print(json.dumps(summarize(dict(row)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
