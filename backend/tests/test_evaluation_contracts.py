import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation import benchmark, runtime
from evaluation.datasets import audit
from evaluation.review import compare, release_gate
from evaluation.usage import usage_summary


@pytest.mark.parametrize("value", ["false", 0, None])
def test_judge_rejects_non_boolean_contradiction(value):
    with pytest.raises(ValidationError):
        benchmark.validate_verdict({"case_id": "test", "score": 4, "contradiction": value, "reason": "test"}, benchmark.FaithfulnessVerdict, "test")


@pytest.mark.parametrize("value", [0, 6, "5", True, 4.0])
def test_judge_rejects_invalid_score_without_clamping(value):
    with pytest.raises(ValidationError):
        benchmark.validate_verdict({"case_id": "test", "score": value, "contradiction": False, "reason": "test"}, benchmark.FaithfulnessVerdict, "test")


def test_judge_rejects_other_case_and_missing_fields():
    with pytest.raises(ValueError):
        benchmark.validate_verdict({"case_id": "wrong", "score": 4, "contradiction": False, "reason": "test"}, benchmark.FaithfulnessVerdict, "test")
    with pytest.raises(ValidationError):
        benchmark.validate_verdict({"case_id": "test", "score": 4}, benchmark.FaithfulnessVerdict, "test")


def test_safety_failure_cannot_be_averaged_into_quality_pass(monkeypatch):
    monkeypatch.setattr(benchmark, "load_jsonl", lambda *_: [{"id": "case", "input": "合成", "expected": "追问"}])
    monkeypatch.setattr(benchmark, "register", lambda _: {})
    monkeypatch.setattr(benchmark, "chat", lambda *_: {"status_code": 200, "data": {"response": "合成"}})
    class Judge:
        def json(self, *_):
            return {"case_id": "case", "reason": "synthetic", "accuracy": 5, "completeness": 5,
                    "clarity": 5, "relevance": 5, "safety": 1}, {}
    result = benchmark.run_quality(None, Judge(), None)
    assert result["metrics"]["overall_mean"] == 4
    assert result["metrics"]["pass_rate_ge_3"] == 0
    assert runtime.gate({"quality": result}, ["quality"])["status"] == "failed"


def test_missing_usage_is_unknown_and_failed_attempts_are_included():
    calls = [{"model": "test", "attempts": [
        {"status": "failed", "input_tokens": 10, "output_tokens": 2},
        {"status": "success", "input_tokens": 10, "output_tokens": 8}]}]
    result = usage_summary(calls, {"test": {"input_per_million": 1, "output_per_million": 2}})
    assert result["total_tokens"] == 30 and result["provider_cost"] == pytest.approx(.00004)
    calls[0]["attempts"][0]["output_tokens"] = None
    result = usage_summary(calls)
    assert result["total_tokens"] is None and result["known_tokens"] == 18
    assert result["usage_coverage"] == .5 and result["provider_cost"] is None
    assert usage_summary([])["total_tokens"] is None


def test_manifest_never_serializes_credentials(monkeypatch):
    monkeypatch.setenv("QIWEN_MODEL_API_KEY", "synthetic-private-marker")
    monkeypatch.setenv("QIWEN_JUDGE_API_KEY", "synthetic-judge-private-marker")
    rendered = json.dumps(runtime.manifest("mock_contract"))
    assert "synthetic-private-marker" not in rendered and "synthetic-judge-private-marker" not in rendered
    assert "backend/resources/symptom_terms.json" in runtime.manifest("mock_contract")["files"]


def test_mock_dependency_check_never_probes_network(monkeypatch):
    monkeypatch.setattr(benchmark, "probe_model", lambda *_: pytest.fail("Mock must not probe configured APIs"))
    assert runtime.dependencies("mock_contract", judge_required=True)["status"] == "ready"


def test_live_judge_must_be_independent(monkeypatch):
    for name in ("QIWEN_MODEL_API_KEY", "QIWEN_JUDGE_API_KEY"):
        monkeypatch.setenv(name, "synthetic")
    for name in ("QIWEN_MODEL_BASE_URL", "QIWEN_JUDGE_BASE_URL"):
        monkeypatch.setenv(name, "https://synthetic.invalid/v1")
    for name in ("QIWEN_MODEL_NAME", "QIWEN_JUDGE_MODEL"):
        monkeypatch.setenv(name, "same-model")
    monkeypatch.setattr(benchmark, "probe_model", lambda *_: {"available": True})
    result = runtime.dependencies("live_core", judge_required=True)
    assert result["status"] == "blocked"
    assert result["checks"]["judge"]["status"] == "independent_judge_required"


def test_failed_empty_and_skipped_dimensions_do_not_pass():
    for row, status in (({"status": "not_run"}, "blocked"), ({"status": "completed", "count": 0}, "blocked"),
                        ({"status": "failed"}, "failed")):
        result = runtime.gate({"router": row}, ["router"])
        assert result["status"] == status
        assert runtime.exit_code(status) != 0


def test_dataset_family_and_duplicate_prompts_cannot_cross_splits(tmp_path):
    rows = [{"id": str(i), "family": "same", "input": "相同问题", "split": split, "version": 1,
             "label_status": "engineering_reviewed", "reviewer": "test", "label_reason": "test"}
            for i, split in enumerate(("development", "frozen"))]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert audit(tmp_path)["status"] == "failed"


def test_archived_and_current_reports_are_not_comparable_when_data_changed():
    original = {"manifest": {"profile": "mock_contract", "files": {"x/datasets/c.jsonl": "old"}}, "results": {}}
    changed = {"manifest": {"profile": "mock_contract", "files": {"x/datasets/c.jsonl": "new"}}, "results": {}}
    assert compare(original, changed)["status"] == "incomparable"


def test_report_comparison_detects_new_dataset_and_changed_modes():
    before = {"manifest": {"files": {}}, "results": {}, "modes": ["fixed"]}
    after = {"manifest": {"files": {"x/datasets/new.jsonl": "new"}}, "results": {}, "modes": ["fixed", "multi"]}
    result = compare(before, after)
    assert "modes" in result["reasons"] and "dataset:x/datasets/new.jsonl" in result["reasons"]


def test_release_requires_unchanged_predeclared_policy_and_deployment(tmp_path):
    from evaluation.review import policy_fingerprint
    policy = {"version": 1, "behavior_version": runtime.BEHAVIOR_VERSION, "thresholds": [
        {"metric": name + ".metrics.score", "operator": "min", "value": 1}
        for name in ("quality", "faithfulness", "performance")] + [
            {"metric": "rag.final_metrics." + name, "operator": "min", "value": .5} for name in ("precision", "recall")], "deployment_targets": [
            {"concurrency": 1, "mode": mode, "scenario": "diagnose", "max_final_p95_s": 10, "min_success_rate": 1}
            for mode in ("fixed", "multi")]}
    path, review_path, deployed_path = (tmp_path / name for name in ("policy.json", "review.json", "deployed.json"))
    path.write_text(json.dumps(policy), encoding="utf-8")
    report = {"gate": {"status": "passed"}, "mode": "live_full", "modes": ["fixed", "multi"],
              "selected": list(benchmark.DIMENSIONS), "dataset_audit": {"frozen_ready": True},
              "manifest": {"run_id": "synthetic", "workspace_sha256": "synthetic", "behavior_version": runtime.BEHAVIOR_VERSION},
              "review_queue": [{"review_id": "synthetic"}], "predeclared_policy_sha256": policy_fingerprint(path),
              "results": {name: {"metrics": {"score": 2}} for name in ("quality", "faithfulness", "performance")}}
    report["results"]["rag"] = {"label_scope": "expert_qrels", "final_metrics": {"precision": 1, "recall": 1},
                                "evidence_contract": {"positive_count": 1, "negative_count": 1, "failed": 0}}
    review_path.write_text(json.dumps({"run_id": "synthetic", "workspace_sha256": "synthetic", "items": [
        {"review_id": "synthetic", "status": "approved", "reviewer": "test", "reason": "synthetic"}]}), encoding="utf-8")
    deployed = {"driver": "real_http_sse_frontend_proxy", "manifest": {"workspace_sha256": "synthetic"},
                "service": {"real_model_enabled": True, "hybrid_retrieval_enabled": True, "retrieval_readiness": {"status": "ready"}},
                "limit": None, "gate": {"status": "passed"}, "evidence_gate": {"status": "passed"},
                "workloads": [{"concurrency": 1, "capacity_sample_ready": True, "observed_load": {"time_at_target_s": 5},
                    "details": [{"mode": mode, "scenario": "diagnose", "passed": True} for mode in ("fixed", "multi")],
                    "latency_s": {mode + ":diagnose": {"final_visible_s": {"p95": 2}} for mode in ("fixed", "multi")}}]}
    deployed_path.write_text(json.dumps(deployed), encoding="utf-8")
    assert release_gate(report, path, review_path, deployed_path)["status"] == "passed"
    assert release_gate(report, path, review_path)["status"] == "blocked"
    deployed["service"]["real_model_enabled"] = False
    deployed_path.write_text(json.dumps(deployed), encoding="utf-8")
    assert release_gate(report, path, review_path, deployed_path)["status"] == "blocked"
    deployed["service"]["real_model_enabled"] = True
    policy["thresholds"][0]["value"] = 0
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert "policy_not_declared_before_run_or_changed" in release_gate(report, path, review_path, deployed_path)["incomplete"]
    report["predeclared_policy_sha256"] = policy_fingerprint(path)
    deployed["workloads"][0]["latency_s"]["multi:diagnose"]["final_visible_s"]["p95"] = 20
    deployed_path.write_text(json.dumps(deployed), encoding="utf-8")
    assert release_gate(report, path, review_path, deployed_path)["status"] == "failed"


@pytest.mark.parametrize("events,passed", [
    ([{"type": "stage", "sequence": 1}, {"type": "answer.final", "sequence": 2, "payload": {"response": "synthetic"}}], True),
    ([{"type": "stage", "sequence": 1}], False),
    ([{"type": "stage", "sequence": 2}, {"type": "answer.final", "sequence": 1, "payload": {"response": "synthetic"}}], False),
])
def test_deployed_stream_gate_checks_final_and_order(events, passed):
    import httpx
    from evaluation.deployment import stream_request
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="\n\n".join("data: " + json.dumps(e) for e in events)))
    with httpx.Client(base_url="http://synthetic.invalid", transport=transport) as client:
        result = stream_request(client, {}, {})
    assert result["passed"] is passed


def test_deployed_stream_rejects_multiple_authoritative_finals():
    import httpx
    from evaluation.deployment import stream_request
    event = {"type": "answer.final", "payload": {"response": "synthetic"}}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=("data: " + json.dumps(event) + "\n\n") * 2))
    with httpx.Client(base_url="http://synthetic.invalid", transport=transport) as client, pytest.raises(ValueError):
        stream_request(client, {}, {})


def test_deployed_stream_progress_cannot_extend_duration_forever(monkeypatch):
    import httpx
    from evaluation import deployment
    clock = iter([0, 181])
    monkeypatch.setattr(deployment.time, "monotonic", lambda: next(clock))
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text='data: {"type":"stage"}\n\n'))
    with httpx.Client(base_url="http://synthetic.invalid", transport=transport) as client, pytest.raises(httpx.TimeoutException):
        deployment.stream_request(client, {}, {})


def test_deployed_request_cannot_pass_when_synthetic_cleanup_fails(monkeypatch):
    import httpx
    from evaluation import deployment
    original = httpx.Client
    def handler(request):
        path = request.url.path
        if path == "/api/register":
            return httpx.Response(200, json={"access_token": "synthetic"})
        if path == "/api/threads":
            return httpx.Response(201, json={"id": "synthetic"})
        if path == "/api/chat":
            return httpx.Response(200, text='data: {"type":"answer.final","sequence":1,"payload":{"response":"synthetic","intent_type":"chat"}}\n\n')
        if path.endswith("/messages"):
            return httpx.Response(200, json=[{"content": "synthetic"}])
        return httpx.Response(500)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    result = deployment.request_one("http://synthetic.invalid", {"id": "synthetic", "input": "synthetic", "scenario": "chat"}, "fixed")
    assert not result["passed"] and result["cleanup_failed"] and result["cleanup_status"] == 500


def test_evaluation_chroma_cleanup_does_not_close_other_indexes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    stopped = []
    owned = SimpleNamespace(settings=SimpleNamespace(persist_directory=str(tmp_path / "chroma")), stop=lambda: stopped.append("owned"))
    other = SimpleNamespace(settings=SimpleNamespace(persist_directory=str(tmp_path / "other")), stop=lambda: stopped.append("other"))
    cls = SimpleNamespace(_identifier_to_system={"owned": owned, "other": other}, _identifier_to_refcount={"owned": 1, "other": 1})
    monkeypatch.setitem(sys.modules, "chromadb.api.shared_system_client", SimpleNamespace(SharedSystemClient=cls))
    runtime.close_evaluation_indexes(tmp_path)
    assert stopped == ["owned"] and set(cls._identifier_to_system) == {"other"}


def test_deployment_measures_chat_overlap_separately_from_job_count(monkeypatch):
    from evaluation import deployment
    clock = iter([0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(deployment.time, "monotonic", lambda: next(clock))
    activity = deployment.LoadActivity(2)
    for delta in (1, 1, -1, -1):
        activity.change(delta)
    result = activity.summary()
    assert result["max_in_flight"] == 2 and result["time_at_target_s"] == 1
    assert result["mean_in_flight"] == .8


def test_completed_answer_still_exposes_failed_retrieval():
    from evaluation.deployment import retrieval_summary
    result = retrieval_summary({"trace": {
        "classics_retrieval": {"final": {"outcome": "timeout", "selected": [], "timings": {"candidate_retrieval": {"elapsed_ms": 30}}}},
        "agents": {"traces": {"bianzheng": [{"tool": "search_classics", "status": "failed"}]}}}})
    assert result["tool_failures"] == 1 and result["retrieval_failures"] == 1 and result["evidence_count"] == 0


def test_small_workload_cannot_claim_requested_capacity(monkeypatch):
    from evaluation import deployment
    def request(base, case, mode, prices, activity):
        activity.change(1)
        activity.change(-1)
        return {"mode": mode, "scenario": case["scenario"], "passed": True, "elapsed_s": .001}
    monkeypatch.setattr(deployment, "request_one", request)
    result = deployment.workload("http://synthetic.invalid", [{"id": "one", "scenario": "chat"}], 10, ("fixed", "multi"), 1)
    assert result["count"] == 2 and not result["capacity_sample_ready"]
    assert result["observed_load"]["max_in_flight"] < 10


def test_tool_semantic_labels_reject_structurally_valid_wrong_query():
    from evaluation.supplement import tool_checks
    actions = [{"agent": "辨证 Agent", "tool": "search_classics", "arguments": {"query": "桂枝汤"}}]
    result = tool_checks(actions, ["search_classics"], {"search_classics": {"query": ["失眠", "安神"]}})
    assert result["schema_valid"] == 1 and "tool_argument_semantic_mismatch" in result["errors"]


def test_evaluation_cli_smoke_runs_current_tools_and_business_contracts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "backend"), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-m", "evaluation.benchmark", "--profile", "mock_contract",
                             "--only", "router,safety,scenarios", "--limit", "1", "--reports", str(tmp_path)],
                            cwd=root, env=env, capture_output=True, timeout=90)
    # Do not expose stdout/stderr; diagnostics may originate in a configured provider.
    assert result.returncode == 0
    report = json.loads(next(tmp_path.glob("benchmark-*.json")).read_text(encoding="utf-8"))
    assert report["gate"]["status"] == "passed"
    assert report["release_gate"]["status"] == "blocked"


def test_retired_tool_is_rejected_without_crashing_registry():
    from evaluation.supplement import tool_checks
    assert tool_checks([], [])["selection_passed"]
    assert not tool_checks([{"agent": "辨证 Agent", "tool": "search_memories", "arguments": {}}], [])["selection_passed"]


def test_checkpoint_pruning_closes_connection_without_waiting_for_gc(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    from app import checkpoint_retention
    path = tmp_path / "checkpoints.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE checkpoints(thread_id TEXT)")
    conn.close()
    tracked = []
    class Tracked:
        def __init__(self, *args, **kwargs):
            self.conn = sqlite3.connect(*args, **kwargs)
            self.closed = False
            tracked.append(self)
        def __enter__(self):
            self.conn.__enter__()
            return self
        def __exit__(self, *args):
            return self.conn.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.conn, name)
        def close(self):
            self.conn.close()
            self.closed = True
    monkeypatch.setattr(checkpoint_retention, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(checkpoint_retention, "sqlite3", SimpleNamespace(connect=Tracked))
    monkeypatch.setattr(checkpoint_retention, "all_rows", lambda *_: [
        {"id": str(i), "thread_id": "thread", "status": "completed"} for i in range(5)])
    checkpoint_retention.prune()
    assert tracked and all(c.closed for c in tracked)
    path.unlink()


def test_thread_deletion_waits_for_checkpoint_writer_without_blocking_loop(tmp_path, monkeypatch):
    import asyncio
    import sqlite3
    from app import main, checkpoint_retention
    main.init_db()
    path = tmp_path / "busy-checkpoints.sqlite3"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE checkpoints(thread_id TEXT)")
    writer.execute("INSERT INTO checkpoints VALUES('synthetic-thread')")
    writer.commit()
    writer.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(checkpoint_retention, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(main, "owned_thread", lambda *_: {})
    monkeypatch.setattr(main, "all_rows", lambda *_: [])
    async def check():
        task = asyncio.create_task(main.delete_thread("synthetic-thread", {"id": "synthetic"}))
        # This commit must run while deletion is waiting on SQLite's write lock.
        await asyncio.sleep(.05)
        writer.commit()
        await asyncio.wait_for(task, timeout=2)
    try:
        asyncio.run(check())
        assert writer.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
    finally:
        writer.close()


def test_input_observer_is_process_only_and_not_persisted():
    from fastapi.testclient import TestClient
    from app import model
    from app.main import app
    captured = []
    token = model.input_observer.set(captured.append)
    try:
        with TestClient(app) as client:
            headers = benchmark.register(client)
            uid = client.get("/api/me", headers=headers).json()["id"]
            client.patch(f"/api/profiles/{uid}", headers=headers, json={"history": "PRIVATE_INPUT_MARKER"}).raise_for_status()
            result = benchmark.chat(client, headers, "我咳嗽三天，伴有白痰，夜间明显")
            assert result["status_code"] == 200
            assert "PRIVATE_INPUT_MARKER" in json.dumps(captured, ensure_ascii=False)
            assert "PRIVATE_INPUT_MARKER" not in json.dumps(result["data"]["trace"], ensure_ascii=False)
            assert "observer" not in result["data"]["trace"]
    finally:
        model.input_observer.reset(token)
    assert model.input_observer.get() is None
