"""Supplement the archived benchmark with retrieval and Agent process metrics.

Live runs use an isolated Mock database. Scripted fault tests measure runtime
controls, not a model's ability to plan recovery. No business database is used.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from .benchmark import ROOT, distribution, load_jsonl, register, relevant


def proportion(passed: int, total: int) -> dict:
    """Wilson 95% interval, descriptive for fixed synthetic samples."""
    if not total:
        return {"passed": passed, "total": total, "rate": None, "wilson_95": None}
    p, z = passed / total, 1.959963984540054
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return {"passed": passed, "total": total, "rate": p,
            "wilson_95": [max(0.0, center - half), min(1.0, center + half)]}


def ranked_metrics(flags: list[bool], total_relevant: int, k: int) -> dict:
    hits = sum(flags[:k])
    return {"hit": int(hits > 0), "precision": hits / k,
            "recall": hits / total_relevant if total_relevant else None}


def supplement_rag(source: dict) -> dict:
    from app.db import all_rows, init_db
    init_db()
    corpus = all_rows("SELECT id,title,chapter,body FROM corpus")
    details = []
    for row in source["results"]["rag"]["details"]:
        case = row["case"]
        count = sum(relevant(doc, case) for doc in corpus)
        flags = [bool(doc["relevant"]) for doc in row["results"]]
        details.append({"id": case["id"], "total_relevant": count,
                        "k3": ranked_metrics(flags, count, 3), "k5": ranked_metrics(flags, count, 5)})
    metrics = {}
    for k in (3, 5):
        vals = [row[f"k{k}"] for row in details]
        metrics[f"hit_at_{k}"] = proportion(sum(v["hit"] for v in vals), len(vals))
        metrics[f"precision_at_{k}"] = statistics.mean(v["precision"] for v in vals)
        recalls = [v["recall"] for v in vals if v["recall"] is not None]
        metrics[f"recall_at_{k}"] = statistics.mean(recalls) if recalls else None
    return {"status": "completed", "count": len(details), "corpus_count": len(corpus),
            "corpus_sha256": hashlib.sha256(json.dumps(sorted(corpus, key=lambda d: d["id"]),
                ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "relevance": "book_and_any_keyword_proxy_not_expert_qrels",
            "zero_relevant_cases": sum(row["total_relevant"] == 0 for row in details),
            "metrics": metrics, "details": details}


def dependency_errors(events: list[dict], mode: str) -> list[str]:
    """Check necessary edges only, allowing revisions and alternate paths."""
    done, errors, previous_synthesis = {}, [], -1
    for i, event in enumerate(events):
        stage = event.get("stage")
        if event.get("status") == "done":
            done[stage] = i
        if event.get("status") != "start":
            continue
        prerequisites = {"formula": ["bianzheng"], "herb": ["formula"],
                         "safety": ["reviewer"]}.get(stage, [])
        if stage == "reviewer":
            prerequisites = ["synthesis"] if mode == "multi" else []
            if mode == "fixed" and not ({"formula", "revise_formula"} & done.keys()):
                errors.append("reviewer_without_candidate")
        if stage == "synthesis":
            prerequisites = ["bianzheng", "formula", "herb"]
            if min(done.get("formula", -1), done.get("herb", -1)) <= previous_synthesis:
                errors.append("synthesis_before_current_revision_experts")
            previous_synthesis = i
        for parent in prerequisites:
            if parent not in done:
                errors.append(f"{stage}_before_{parent}")
    return errors


def tool_checks(actions: list[dict], required: list[str], semantics: dict | None = None) -> dict:
    from app.agents import tools as t
    registry = {spec.name: spec for spec in (t.PROFILE_TOOL, t.CLASSICS_TOOL,
                t.FORMULA_TOOL, t.HERB_TOOL, t.DOSAGE_TOOL, t.COMPATIBILITY_TOOL, t.USER_RISK_TOOL)}
    allowed = {"辨证 Agent": {"get_patient_profile", "search_classics"},
               "方剂 Agent": {"search_formulas", "lookup_herb", "get_dosage_evidence", "check_herb_compatibility"},
               "本草 Agent": {"get_patient_profile", "lookup_herb", "get_dosage_evidence",
                            "check_herb_compatibility", "check_user_risks"}}
    errors, valid = [], 0
    for action in actions:
        name = action.get("tool")
        if name not in allowed.get(action.get("agent"), set()):
            errors.append("tool_outside_role_allowlist")
        try:
            registry[name].args_schema.model_validate(action.get("arguments"))
            valid += 1
            constraints = (semantics or {}).get(name, {})
            for field, permitted in constraints.items():
                value = action["arguments"].get(field)
                if not isinstance(value, str) or not any(term in value for term in permitted):
                    errors.append("tool_argument_semantic_mismatch")
        except Exception:
            errors.append("invalid_tool_arguments")
    missing = sorted(set(required) - {a.get("tool") for a in actions})
    # Repeated calls across safety revisions may be necessary; don't call these waste.
    counts = Counter((a.get("agent"), a.get("tool"),
                      json.dumps(a.get("arguments"), sort_keys=True, ensure_ascii=False)) for a in actions)
    return {"count": len(actions), "schema_valid": valid, "errors": errors,
            "missing_required_tools": missing, "selection_passed": not errors and not missing,
            "repeated_signature_calls": sum(n - 1 for n in counts.values())}


def run_tasks(client, repeat: int, limit: int | None) -> dict:
    from app import workflow
    cases = load_jsonl("agent_tasks.jsonl", limit)
    details = []
    async def empty(_plan):
        return [], {"warning": ""}
    for case in cases:
        for repetition in range(1, repeat + 1):
            record = {"id": case["id"], "scenario": case["scenario"], "repeat": repetition,
                      "mode": case["mode"], "responses": [], "constraints": False,
                      "dependencies": False, "state_consistency": False, "task_success": False}
            try:
                headers = register(client)
                if case.get("profile"):
                    uid = client.get("/api/me", headers=headers).json()["id"]
                    client.patch(f"/api/profiles/{uid}", headers=headers, json=case["profile"]).raise_for_status()
                thread = client.post("/api/threads", headers=headers, json={"mode": case["mode"]}).json()["id"]
                responses = []
                for text in case["turns"]:
                    manager = patch.object(workflow, "retrieve_classics", empty) if case.get("fault") else patch.dict(os.environ, {})
                    with manager:
                        response = client.post("/api/chat", headers=headers,
                                               json={"thread_id": thread, "message": text, "mode": case["mode"]})
                    responses.append({"status_code": response.status_code, "data": response.json()})
                record["responses"] = responses
                data = responses[-1]["data"]
                trace = data.get("trace") or {}
                events = trace.get("events", [])
                starts = [e.get("stage") for e in events if e.get("status") == "start"]
                constraints = {
                    "required_stages": set(case.get("required_stages", [])) <= set(starts),
                    "forbidden_stages": not (set(case.get("forbidden_stages", [])) & set(starts)),
                    "required_text": all(s in data.get("response", "") for s in case.get("required_text", [])),
                    "forbidden_text": all(s not in data.get("response", "") for s in case.get("forbidden_text", [])),
                    "symptoms": set(case.get("required_symptoms", [])) <= set(data.get("symptoms", [])),
                    "denials": not (set(case.get("denied_symptoms", [])) & set(data.get("symptoms", []))),
                    "notes": "expert_notes" not in case or len(data.get("notes", [])) == case["expert_notes"],
                    "bounded_search": starts.count("search") <= 3,
                    "bounded_safety": starts.count("safety") <= 3,
                }
                for field, stage in (("search_count", "search"), ("rewrite_count", "rewrite_query")):
                    if field in case:
                        constraints[field] = starts.count(stage) == case[field]
                dependency_failures = dependency_errors(events, case["mode"])
                history = client.get(f"/api/threads/{thread}/messages", headers=headers).json()
                run_events = client.get(f'/api/runs/{data["run_id"]}/events', headers=headers).json()
                actions = [e["payload"] for e in run_events if e.get("type") == "agent.action"]
                tools = tool_checks(actions, case["required_tools"], case.get("tool_arguments"))
                selected_tools = {a.get("tool") for a in actions}
                tools["alternative_groups_passed"] = all(set(group) & selected_tools for group in case.get("required_tool_groups", []))
                tools["selection_passed"] = tools["selection_passed"] and tools["alternative_groups_passed"]
                if not case["required_tools"]:
                    tools["selection_passed"] = tools["selection_passed"] and not actions
                finals = [e["payload"] for e in run_events if e.get("type") == "answer.final"]
                state_ok = (len(history) == 2 * len(case["turns"]) and
                            history[-1]["content"] == data.get("response") and len(finals) == 1 and
                            finals[0].get("response") == data.get("response"))
                before = len(history)
                cancel1 = client.post(f'/api/runs/{data["run_id"]}/cancel', headers=headers).json()
                cancel2 = client.post(f'/api/runs/{data["run_id"]}/cancel', headers=headers).json()
                idempotent = (cancel1.get("status") == cancel2.get("status") == "completed" and
                              len(client.get(f"/api/threads/{thread}/messages", headers=headers).json()) == before)
                intents_ok = [r["data"].get("intent_type") for r in responses] == case["intents"]
                requests_ok = all(r["status_code"] == 200 for r in responses)
                record.update(constraint_checks=constraints, constraints=all(constraints.values()),
                              dependency_errors=dependency_failures, dependencies=not dependency_failures,
                              state_consistency=state_ok, cancel_idempotency=idempotent, tool_checks=tools,
                              actions=actions, request_success=requests_ok, intents_correct=intents_ok)
                record["task_success"] = all((requests_ok, intents_ok, record["constraints"],
                                              record["dependencies"], state_ok, idempotent, tools["selection_passed"]))
            except Exception as exc:
                record["error_type"] = type(exc).__name__
            details.append(record)
    metrics = {key: proportion(sum(bool(r.get(key)) for r in details), len(details))
               for key in ("task_success", "constraints", "dependencies", "state_consistency", "cancel_idempotency")}
    actions = [r["tool_checks"] for r in details if "tool_checks" in r]
    metrics["tool_selection"] = proportion(sum(r["selection_passed"] for r in actions), len(details))
    metrics["tool_argument_schema_validity"] = proportion(sum(r["schema_valid"] for r in actions), sum(r["count"] for r in actions))
    metrics["repeated_signature_calls"] = sum(r["repeated_signature_calls"] for r in actions)
    by_case = {case["id"]: proportion(sum(r["task_success"] for r in details if r["id"] == case["id"]), repeat) for case in cases}
    # Repeated deterministic tasks are correlated, not 24 independent samples.
    metrics["all_repeats_passed_unique_tasks"] = proportion(sum(v["passed"] == repeat for v in by_case.values()), len(cases))
    for key, value in metrics.items():
        if isinstance(value, dict) and key != "all_repeats_passed_unique_tasks":
            value["wilson_95"] = None
            value["interval_reason"] = "correlated_repeated_runs_not_independent_samples"
    for value in by_case.values():
        value["wilson_95"] = None
    for case in cases:
        records = [r for r in details if r["id"] == case["id"]]
        by_case[case["id"]]["distinct_final_responses"] = len({
            r["responses"][-1]["data"].get("response", "") for r in records if r["responses"]})
    return {"status": "completed", "mode": "mock", "unique_tasks": len(cases), "repeat": repeat,
            "count": len(details), "metrics": metrics, "by_case": by_case, "details": details,
            "scope": "workflow_success_not_clinical_correctness; argument_semantics_only_when_explicitly_labelled"}


def runtime_controls() -> dict:
    from pydantic import BaseModel, ConfigDict
    from app.agents.runtime import AgentLoopError, ToolSpec, run_agent_graph
    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        query: str
    class Result(BaseModel):
        status: str
        evidence: list[str]
    details = []
    for fault in ("exception", "timeout", "invalid_arguments", "forbidden_tool", "loop"):
        calls = 0
        async def handler(_state, _args):
            nonlocal calls
            calls += 1
            if calls == 1 and fault == "exception":
                raise RuntimeError("synthetic")
            if calls == 1 and fault == "timeout":
                await asyncio.sleep(.02)
            return {"evidence": ["synthetic-evidence"]}
        def planner(step, observations):
            if fault == "loop":
                return {"tool": "search", "arguments": {"query": "synthetic"}}
            if observations and observations[-1]["status"] == "success":
                return {"final": {"status": "completed", "evidence": observations[-1]["result"]["evidence"]}}
            return {"tool": "forbidden" if fault == "forbidden_tool" and step == 1 else "search",
                    "arguments": {"query": 123 if fault == "invalid_arguments" and step == 1 else "synthetic"}}
        try:
            run = asyncio.run(run_agent_graph(name="补充评测", role="bianzheng", task="合成运行时控制检查",
                             state={}, tools=[ToolSpec("search", "synthetic", Args, handler, .005)],
                             output_schema=Result, mock_planner=planner, max_steps=4))
            passed = run.output["evidence"] == ["synthetic-evidence"] and run.trace[0]["status"] == "failed"
            details.append({"fault": fault, "passed": passed, "calls": calls, "trace": run.trace})
        except AgentLoopError:
            details.append({"fault": fault, "passed": fault == "loop" and calls == 2,
                            "calls": calls, "termination": "bounded_loop_error"})
        except Exception as exc:
            details.append({"fault": fault, "passed": False, "error_type": type(exc).__name__})
    return {"status": "completed", "scope": "scripted_mock_planner_runtime_controls_only",
            "metrics": proportion(sum(r["passed"] for r in details), len(details)), "details": details}


def authorization_checks(client) -> dict:
    owner, other = register(client), register(client)
    result = client.post("/api/chat", headers=owner, json={"message": "你好"}).json()
    cases = [("read_history", "get", f'/api/threads/{result["thread_id"]}/messages', None),
             ("read_events", "get", f'/api/runs/{result["run_id"]}/events', None),
             ("cancel_other_run", "post", f'/api/runs/{result["run_id"]}/cancel', None),
             ("write_other_thread", "post", "/api/chat", {"message": "越权写入", "thread_id": result["thread_id"]})]
    before = client.get(f'/api/threads/{result["thread_id"]}/messages', headers=owner).json()
    details = []
    for name, method, path, payload in cases:
        kwargs = {"headers": other}
        if payload is not None:
            kwargs["json"] = payload
        response = getattr(client, method)(path, **kwargs)
        after = client.get(f'/api/threads/{result["thread_id"]}/messages', headers=owner).json()
        details.append({"case": name, "status_code": response.status_code,
                        "attack_succeeded": response.status_code < 400 or after != before})
    return {"status": "completed", "scope": "four_cross_account_requests",
            "attack_success": proportion(sum(r["attack_succeeded"] for r in details), len(details)), "details": details}


def archived_operations(source: dict) -> dict:
    rows = source["results"]["performance"]["details"]
    requests = proportion(sum(r["status_code"] == 200 for r in rows), len(rows))
    intents = proportion(sum(r["status_code"] == 200 and r["intent"] == r["case"]["scenario"] for r in rows), len(rows))
    scenarios = {s: distribution([r["elapsed_s"] for r in rows if r["case"]["scenario"] == s]) for s in ("chat", "diagnose")}
    for s in scenarios:
        from .benchmark import percentile
        scenarios[s]["p99"] = round(percentile([r["elapsed_s"] for r in rows if r["case"]["scenario"] == s], .99), 3)
    benign = [r for r in source["results"]["router"]["details"] if r["case"]["expected"] == "chat"]
    return {"request_success": requests, "scenario_completion": intents, "latency_s": scenarios,
            "http_error_rate": proportion(sum(r["status_code"] >= 400 for r in rows), len(rows)),
            "http_timeout_rate": proportion(sum(r["status_code"] in {408, 504} for r in rows), len(rows)),
            "false_safety_refusal": proportion(sum(r["actual"] in {"safety_rejected", "safety_advisory"} for r in benign), len(benign)),
            "safety_e2e_ci": proportion(sum(r["passed"] for r in source["results"]["safety"]["e2e"]["details"]), source["results"]["safety"]["e2e"]["count"]),
            "cost": {"status": "not_run", "reason": "mock_no_billable_model_calls; infrastructure_and_tool_prices_unavailable"},
            "p99_note": "15_samples_per_scenario_descriptive_only_not_load_test"}


def main() -> int:
    from .runtime import environment, manifest, write_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional archived report; never used as current gate evidence")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reports", type=Path, default=Path(__file__).with_name("reports"))
    args = parser.parse_args()
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeat and limit must be positive")
    source = json.loads(args.source.read_text(encoding="utf-8")) if args.source else None
    with environment("mock_contract"):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client:
            report = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "mock",
                      "archive": {"source": args.source.name, "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                                  "rag": supplement_rag(source), "operations": archived_operations(source)} if source else None,
                      "agent": run_tasks(client, args.repeat, args.limit), "runtime_controls": runtime_controls(),
                      "authorization": authorization_checks(client)}
        report["manifest"] = manifest("mock_contract")
    report["pending_metrics"] = {
        "context_recall_precision": "needs_answer_facts_and_admitted_context_labels",
        "answer_relevance_correctness_citations": "needs_real_answers_and_reference_claim_labels",
        "knowledge_abstention": "needs_answerable_unanswerable_dataset_and_real_model",
        "tool_argument_semantic_accuracy_redundancy": "needs_task_specific_semantic_and_necessary_call_labels",
        "prompt_injection_leakage": "needs_user_document_tool_payload_attack_set_and_real_model",
        "ttft_throughput": "needs_stream_and_concurrency_load_driver_and_target_load",
        "cost_per_request_success": "needs_complete_usage_and_model_tool_infrastructure_prices",
        "business_effect": "needs_user_feedback_or_measured_business_baseline",
    }
    passed = (report["agent"]["count"] > 0 and report["agent"]["metrics"]["task_success"]["rate"] == 1
              and report["runtime_controls"]["metrics"]["rate"] == 1
              and report["authorization"]["attack_success"]["rate"] == 0)
    report["gate"] = {"status": "passed" if passed else "failed", "scope": "current_agent_runtime_authorization_only"}
    path = write_report(args.reports, "supplement", report)
    print(json.dumps({"report": str(path), "gate": report["gate"],
                      "agent": report["agent"]["metrics"], "by_case": report["agent"]["by_case"],
                      "runtime": report["runtime_controls"]["metrics"],
                      "authorization": report["authorization"]["attack_success"]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
