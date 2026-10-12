"""HTTP/SSE workload through the deployed frontend proxy, synthetic accounts only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from .benchmark import distribution, load_jsonl
from .runtime import manifest, write_report
from .usage import usage_summary


class LoadActivity:
    """Measure overlapping chat streams, excluding account setup and cleanup."""
    def __init__(self, target):
        self.target = target
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.area = 0.0
        self.at_target = 0.0
        self.start = self.last = time.monotonic()

    def change(self, delta):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.area += self.active * elapsed
            if self.active >= self.target:
                self.at_target += elapsed
            self.last = now
            self.active += delta
            self.peak = max(self.peak, self.active)

    def summary(self):
        self.change(0)
        return {"max_in_flight": self.peak,
                "mean_in_flight": self.area / max(.001, self.last - self.start),
                "time_at_target_s": self.at_target, "scope": "chat_streams_only"}


def retrieval_summary(data, *, include_synthetic_diagnostics=False):
    trace = data.get("trace") or {}
    classics = trace.get("classics_retrieval") or {}
    final = classics.get("final") or {}
    observations = [item for rows in (trace.get("agents") or {}).get("traces", {}).values()
                    for item in rows if item.get("tool")]
    history = classics.get("attempts", []) or ([final] if final else [])
    attempts = [attempt for item in history for attempt in (item.get("attempts") or [item])]
    def attempt_summary(item):
        result = {k: item[k] for k in ("outcome", "timings", "elapsed_ms", "remaining_ms", "cached", "candidate_pool_limit",
                                       "candidate_count", "vector_status", "indexed_count", "dependency_status", "retryable",
                                       "retry_minimum_ms") if k in item}
        rerank = item.get("rerank") or {}
        scores = [row["score"] for row in rerank.get("ranked", [])
                  if type(row.get("score")) in (float, int) and math.isfinite(row["score"])]
        result["rerank"] = {k: rerank[k] for k in ("status", "input_count", "output_count", "threshold",
                                                 "model", "revision", "max_length") if k in rerank}
        result["rerank"]["highest_score"] = max(scores) if scores else None
        result["recall_audit"] = item.get("recall_audit", [])
        result["rerank"]["ranked"] = (rerank.get("ranked", []) if include_synthetic_diagnostics else
                                       [{key: row[key] for key in ("id", "score", "accepted", "rejection_reason") if key in row}
                                        for row in rerank.get("ranked", [])])
        result["evidence_kind"] = item.get("evidence_kind")
        result["recall_query_count"] = item.get("recall_query_count")
        if include_synthetic_diagnostics:
            result["query_plan"] = item.get("query_plan", {})
        result["source_manifest_sha256"] = item.get("source_manifest_sha256")
        result["source_scope_sha256"] = item.get("source_scope_sha256")
        if include_synthetic_diagnostics:
            result["rerank"]["input_windows"] = rerank.get("input_windows", [])
        result["admitted_count"] = item.get("admitted_count")
        result["mmr_status"] = (item.get("mmr") or {}).get("status")
        return result
    return {"outcome": final.get("outcome", "not_called" if not final else "unknown"),
            "evidence_count": len(final.get("selected", [])),
            "tool_calls": len(observations),
            "tool_failures": sum(item.get("status") != "success" for item in observations),
            "retrieval_failures": sum(item.get("outcome") not in {"success", "no_match", "source_changed"} for item in attempts),
            "attempts": [attempt_summary(item) for item in attempts]}


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Base URL must not contain credentials, query or fragment")
    return value.rstrip("/")


def is_loopback(base: str) -> bool:
    return urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}


def stream_request(client, headers, body, max_duration_s=180) -> dict:
    if type(max_duration_s) not in (int, float) or not math.isfinite(max_duration_s) or max_duration_s <= 0:
        raise ValueError("Invalid stream duration")
    start = time.monotonic()
    events, first_progress, first_candidate_delta, final_visible, final = [], None, None, None, None
    with client.stream("POST", "/api/chat?stream=true", headers=headers, json=body) as response:
        status = response.status_code
        if status != 200:
            return {"status_code": status, "passed": False, "elapsed_s": time.monotonic() - start}
        for line in response.iter_lines():
            if time.monotonic() - start > max_duration_s:
                raise httpx.TimeoutException("Stream exceeded evaluation duration")
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            events.append(event)
            if event.get("type") in {"stage", "phase"} and first_progress is None:
                first_progress = time.monotonic() - start
            if event.get("type") == "candidate.delta" and first_candidate_delta is None:
                first_candidate_delta = time.monotonic() - start
            if event.get("type") == "answer.final":
                if final is not None:
                    raise ValueError("Multiple authoritative finals")
                final = event.get("payload", event)
                final_visible = time.monotonic() - start
    sequences = [e["sequence"] for e in events if "sequence" in e]
    ordered = all(a < b for a, b in zip(sequences, sequences[1:]))
    return {"status_code": status, "passed": bool(final) and ordered, "data": final or {},
            "stages": sorted({e.get("stage") or e.get("payload", {}).get("stage") for e in events
                              if e.get("stage") or e.get("payload", {}).get("stage")}),
            "failure_reason": "missing_authoritative_final" if not final else "event_order" if not ordered else None,
            "terminal_event": events[-1].get("type") if events else None,
            "first_progress_s": first_progress, "final_visible_s": final_visible,
            "first_candidate_delta_s": first_candidate_delta,
            "candidate_delta_scope": "proxy_candidate_event_not_provider_ttft_or_visible_answer",
            "timeout_scope": "idle_io_timeout_configured_by_client; elapsed_guard_on_each_received_line",
            "elapsed_guard_s": max_duration_s,
            "elapsed_s": time.monotonic() - start, "event_count": len(events), "ordered": ordered}


def request_one(base: str, case: dict, mode: str, prices=None, activity=None) -> dict:
    record = {"id": case["id"], "family": case.get("family"), "scenario": case["scenario"], "mode": mode, "passed": False}
    thread = None
    request_started = None
    with httpx.Client(base_url=base, timeout=180, trust_env=not is_loopback(base)) as client:
        try:
            registered = client.post("/api/register", json={"username": "eval_http_" + uuid4().hex[:12], "password": "synthetic-evaluation-password"})
            registered.raise_for_status()
            headers = {"Authorization": "Bearer " + registered.json()["access_token"]}
            created = client.post("/api/threads", headers=headers, json={"mode": mode})
            created.raise_for_status()
            thread = created.json()["id"]
            request_started = time.monotonic()
            turn_results = []
            for message in case.get("turns") or [case["input"]]:
                if activity:
                    activity.change(1)
                try:
                    result = stream_request(client, headers, {"thread_id": thread, "message": message, "mode": mode})
                finally:
                    if activity:
                        activity.change(-1)
                turn_record = {k: result[k] for k in ("passed", "elapsed_s", "final_visible_s") if k in result}
                turn_record["retrieval"] = retrieval_summary(result.get("data", {}), include_synthetic_diagnostics=True)
                turn_results.append(turn_record)
                if not result["passed"]:
                    break
            data = result.pop("data", {})
            if not result["passed"]:
                history = client.get(f"/api/threads/{thread}/messages", headers=headers)
                if history.status_code == 200 and history.json():
                    failed_trace = history.json()[-1].get("trace") or {}
                    if isinstance(failed_trace, str):
                        failed_trace = json.loads(failed_trace)
                    record["failed_model_calls"] = failed_trace.get("model_calls", [])
                    record["failed_stages"] = [e.get("stage") for e in failed_trace.get("events", []) if e.get("status") == "failed"]
            usage = usage_summary(data.get("trace", {}).get("model_calls", []), prices)
            if result["passed"]:
                history = client.get(f"/api/threads/{thread}/messages", headers=headers)
                history.raise_for_status()
                result["passed"] = history.json()[-1]["content"] == data.get("response") and data.get("intent_type") == case["scenario"]
            evidence = retrieval_summary(data, include_synthetic_diagnostics=True)
            record.update(result, usage=usage, intent=data.get("intent_type"), retrieval=evidence)
            record.update(input=case["input"], answer=data.get("response", ""),
                          model_calls=data.get("trace", {}).get("model_calls", []), semantic_review="pending")
            cited = data.get("trace", {}).get("answer_citations")
            if cited is None:
                cited = (data.get("trace", {}).get("classics_retrieval") or {}).get("final", {}).get("selected", [])
            record["references"] = [{key: s.get(key) for key in ("id", "title", "chapter", "excerpt", "evidence", "quote_displayed")}
                                    for s in cited]
            record.update(turn_results=turn_results, conversation_elapsed_s=time.monotonic() - request_started,
                          case_kind=case.get("kind", case["scenario"]),
                          route_kind="knowledge" if "knowledge_search" in result.get("stages", []) else data.get("intent_type"))
            record["passed"] = record["passed"] and len(turn_results) == len(case.get("turns") or [case["input"]]) and all(t["passed"] for t in turn_results)
            if "expected_evidence" in case:
                record["evidence_passed"] = ((evidence["evidence_count"] > 0 and evidence["retrieval_failures"] == 0)
                                             if case["expected_evidence"] else evidence["evidence_count"] == 0 and evidence["outcome"] in {"no_match", "not_called"})
            if case.get("rag_expectation"):
                from .benchmark import relevant, normalize
                final_trace = (data.get("trace", {}).get("classics_retrieval") or {}).get("final") or {}
                references = [{"id": s["id"], "title": s["title"], "chapter": s.get("chapter", ""),
                    "body": s.get("evidence") or s.get("excerpt", ""),
                    "retrieval": {"evidence": s.get("evidence") or s.get("excerpt", "")}}
                    for s in final_trace.get("selected", [])]
                expected = case["rag_expectation"]
                combined = "\n".join(normalize(r["body"]) for r in references)
                missing = [g for g in expected.get("final_required_term_groups", []) if not any(normalize(t) in combined for t in g)]
                record["full_retrieval_chain_passed"] = (final_trace.get("vector_status") == "success" and
                    final_trace.get("dependency_status") == "ok" and
                    final_trace.get("rerank", {}).get("status") == "success" and
                    final_trace.get("mmr", {}).get("status") in {"success", "not_needed"})
                record["evidence_proxy_passed"] = (any(relevant(r, expected) for r in references) and not missing and
                    record["full_retrieval_chain_passed"] and record.get("evidence_passed", False))
                record["proxy_references"] = references
                record["missing_final_term_groups"] = missing
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            if request_started is not None:
                record["elapsed_s"] = time.monotonic() - request_started
            record["timed_out"] = isinstance(exc, httpx.TimeoutException)
        finally:
            if thread:
                try:
                    record["cleanup_status"] = client.delete(f"/api/threads/{thread}", headers=headers).status_code
                    if record["cleanup_status"] not in {204, 404}:
                        record["passed"] = False
                        record["cleanup_failed"] = True
                except Exception as exc:
                    record["cleanup_error_type"] = type(exc).__name__
                    record["cleanup_failed"] = True
                    record["passed"] = False
    return record


def workload(base: str, cases: list[dict], concurrency: int, modes: tuple, repeat: int, prices=None) -> dict:
    tasks = [(case, mode) for _ in range(repeat) for case in cases for mode in modes]
    started = time.monotonic()
    activity = LoadActivity(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_one, base, case, mode, prices, activity) for case, mode in tasks]
        rows = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    successful = [r for r in rows if r["passed"]]
    groups = {}
    for mode in modes:
        for scenario in {c["scenario"] for c in cases}:
            all_selected = [r for r in rows if r["mode"] == mode and r["scenario"] == scenario]
            selected = [r for r in all_selected if r["passed"]]
            groups[mode + ":" + scenario] = {key: distribution([r[key] for r in selected if r.get(key) is not None])
                for key in ("first_progress_s", "final_visible_s", "elapsed_s")}
            groups[mode + ":" + scenario].update(count=len(all_selected), passed=len(selected),
                success_rate=len(selected) / len(all_selected) if all_selected else None,
                error_latencies_s=distribution([r["elapsed_s"] for r in all_selected if not r["passed"] and "elapsed_s" in r]))
    known_costs = [r.get("usage", {}).get("provider_cost") for r in rows]
    cost_complete = bool(rows) and all(c is not None for c in known_costs)
    observed = activity.summary()
    by_kind = {}
    for mode in modes:
        for kind in {r.get("case_kind", r["scenario"]) for r in rows}:
            group = [r for r in rows if r["mode"] == mode and r.get("case_kind", r["scenario"]) == kind]
            by_kind[mode + ":" + kind] = {"count": len(group), "passed": sum(r["passed"] for r in group),
                "conversation_elapsed_s": distribution([r["conversation_elapsed_s"] for r in group if r["passed"] and "conversation_elapsed_s" in r]),
                "retrieval_failures": sum(r.get("retrieval", {}).get("retrieval_failures", 0) for r in group)}
    return {"concurrency": concurrency, "observed_load": observed,
            "by_case_kind": by_kind,
            "capacity_sample_ready": len(rows) >= concurrency * 3 and observed["max_in_flight"] >= concurrency and observed["time_at_target_s"] >= 1,
            "capacity_sample_rule": "at_least_3_jobs_per_slot; measured_target_overlap_at_least_1s; declared_SLA_still_required",
            "count": len(rows), "passed": len(successful),
            "tool_failure_count": sum(r.get("retrieval", {}).get("tool_failures", 0) for r in rows),
            "retrieval_failure_count": sum(r.get("retrieval", {}).get("retrieval_failures", 0) for r in rows),
            "evidence_check_count": sum("evidence_passed" in r for r in rows),
            "evidence_failure_count": sum(r.get("evidence_passed") is False for r in rows),
            "evidence_proxy_check_count": sum("evidence_proxy_passed" in r for r in rows),
            "evidence_proxy_failure_count": sum(r.get("evidence_proxy_passed") is False for r in rows),
            "timeout_count": sum(r.get("timed_out", False) for r in rows),
            "cleanup_failure_count": sum(r.get("cleanup_failed", False) for r in rows),
            "success_rate": len(successful) / len(rows) if rows else None, "latency_s": groups,
            "error_latencies_s": distribution([r["elapsed_s"] for r in rows if not r["passed"] and "elapsed_s" in r]),
            "end_to_end_jobs_per_s": len(rows) / elapsed if elapsed else None,
            "throughput_scope": "includes_synthetic_account_setup_and_cleanup",
            "provider_cost_per_success": sum(known_costs) / len(successful) if cost_complete and successful else None,
            "usage_coverage": sum(r.get("usage", {}).get("total_tokens") is not None for r in rows) / len(rows) if rows else None,
            "details": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:23000")
    parser.add_argument("--concurrency", default="1,3,5,10")
    parser.add_argument("--modes", default="fixed,multi")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", choices=("deployment_cases.jsonl", "performance.jsonl", "rag_regression.jsonl"), default="deployment_cases.jsonl")
    parser.add_argument("--prices", type=Path, help="Model -> input/output_per_million, one declared currency")
    parser.add_argument("--reports", type=Path, default=Path(__file__).with_name("reports"))
    args = parser.parse_args()
    levels, modes = [int(n) for n in args.concurrency.split(",")], tuple(args.modes.split(","))
    if not levels or min(levels) < 1 or max(levels) > 100 or args.repeat < 1 or args.limit is not None and args.limit < 1 or set(modes) - {"fixed", "multi"}:
        parser.error("invalid workload")
    base = endpoint(args.base_url)
    prices = json.loads(args.prices.read_text(encoding="utf-8")) if args.prices else None
    cases = load_jsonl(args.dataset, args.limit)
    with httpx.Client(base_url=base, timeout=30, trust_env=not is_loopback(base)) as client:
        health_response = client.get("/api/health")
        health_response.raise_for_status()
        health = health_response.json()
        ready_response = client.get("/api/readiness")
        ready = ready_response.json() if ready_response.status_code == 200 else {"status": "unavailable"}
    report = {"manifest": manifest("deployed_http"), "driver": "real_http_sse_frontend_proxy",
              "service": {"healthy": health.get("status") == "ok",
                          "retrieval_readiness": ready,
                          "real_model_enabled": health.get("model_mode") == "openai_compatible",
                          "hybrid_retrieval_enabled": health.get("rag_mode") == "hybrid"},
              "scope": "synthetic_load_no_clinical_quality_claim", "limit": args.limit, "dataset": args.dataset,
              "cleanup": "threads_deleted; synthetic_accounts_retained_by_service_without_account_delete_api",
              "workloads": [workload(base, cases, level, modes, args.repeat, prices) for level in levels]}
    report["gate"] = {"status": "passed" if all(r["count"] and r["count"] == r["passed"] for r in report["workloads"]) else "failed",
                      "scope": "request_route_stream_persistence_not_sla"}
    records = [r for w in report["workloads"] for r in w["details"]]
    report["model_judge"] = {"status": "not_enabled"}
    report["review_queue"] = [{"review_id": hashlib.sha256((report["manifest"]["run_id"] + ":" + str(i)).encode()).hexdigest()[:16],
        "dimension": "deployed_answer", "case_id": r["id"], "mode": r["mode"],
        "input": r.get("input"), "answer": r.get("answer", ""), "references": r.get("references", []),
        "reason": "semantic_review_required", "status": "pending", "supported_claims": [],
        "missing_conditions": [], "contradicted_claims": []} for i, r in enumerate(records)]
    executed = sum(any(c.get("status") == "success" and c.get("model") and
                       c.get("role") not in {"cross_encoder", "mmr", "embedding"}
                       for c in r.get("model_calls", [])) for r in records)
    report["generation_gate"] = {"status": "passed" if report["service"]["real_model_enabled"] and executed == len(records) and records else
                                "failed" if report["service"]["real_model_enabled"] else "not_applicable",
                                "executed": executed, "count": len(records), "scope": "actual_successful_model_calls_not_semantic_quality"}
    report["capacity_gate"] = {"status": "ready_for_sla_check" if all(w["capacity_sample_ready"] for w in report["workloads"]) else "blocked",
                               "scope": "observed_load_and_sample_size_not_sla"}
    checked = sum(w["evidence_check_count"] for w in report["workloads"])
    report["evidence_gate"] = {"status": "failed" if any(w["evidence_failure_count"] for w in report["workloads"]) else
                                         "passed" if checked == sum(w["count"] for w in report["workloads"]) and checked else "blocked",
                               "scope": "expected_evidence_presence_not_semantic_quality"}
    proxy_count = sum(w["evidence_proxy_check_count"] for w in report["workloads"])
    report["evidence_proxy_gate"] = {"status": "failed" if any(w["evidence_proxy_failure_count"] for w in report["workloads"]) else
        "passed" if proxy_count else "not_applicable", "scope": "actual_final_quotes_and_complete_retrieval_components; pending_expert_review"}
    with httpx.Client(base_url=base, timeout=15, trust_env=not is_loopback(base)) as client:
        response = client.get("/api/readiness")
        report["service"]["retrieval_readiness_before"] = ready
        report["service"]["retrieval_readiness"] = response.json() if response.status_code == 200 else {"status": "unavailable"}
    path = write_report(args.reports, "deployment", report)
    print(json.dumps({"report": str(path), "gate": report["gate"], "counts": [
        {"concurrency": r["concurrency"], "count": r["count"], "passed": r["passed"]} for r in report["workloads"]],
        "capacity_gate": report["capacity_gate"], "evidence_gate": report["evidence_gate"],
        "evidence_proxy_gate": report["evidence_proxy_gate"]}, ensure_ascii=False))
    return 0 if report["gate"]["status"] == "passed" and all(report[k]["status"] != "failed" for k in
        ("evidence_gate", "evidence_proxy_gate", "generation_gate")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
