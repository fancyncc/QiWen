import json

from scripts.inspect_run import summarize


def test_diagnostics_distinguish_completed_run_from_incomplete_agent_flow_and_omit_sensitive_content():
    secret = "sensitive-value-that-must-not-be-printed"
    row = {"id": "run-id", "thread_id": "thread-id", "status": "completed", "telemetry": json.dumps({
        "elapsedMs": 112668, "profile": {"allergies": secret}, "headers": {"Authorization": secret},
        "agents": {"artifacts": {"formula": {"status": "completed", "plan": None, "comparison": secret}},
                   "traces": {"formula": [{"tool": "search_formulas", "status": "success", "arguments": {"query": secret},
                                           "result": {"formulas": [{"name": secret}]}}]}},
        "model_calls": [{"node": "formula", "status": "success", "prompt": secret, "api_key": secret}],
    })}
    result = summarize(row)
    assert result["status"] == "completed"
    assert "FORMULA_COMPLETED_WITHOUT_PERSONAL_PLAN" in result["findings"]
    assert "LITERATURE_AVAILABLE_BUT_HERB_STAGE_NOT_RUN" in result["findings"]
    assert result["tools"][0]["formula_count"] == 1
    assert secret not in json.dumps(result)


def test_diagnostics_report_tool_failure_and_cancelled_run_without_raw_errors():
    row = {"id": "run-id", "thread_id": "thread-id", "status": "cancelled", "telemetry": json.dumps({
        "agents": {"traces": {"formula": [{"tool": "search_formulas", "status": "error", "error": "private error"}]}}})}
    result = summarize(row)
    assert "TOOL_CALL_FAILED" in result["findings"]
    assert "RUN_ENDED_WITHOUT_FINAL_ANSWER" in result["findings"]
    assert "private error" not in json.dumps(result)


def test_outage_diagnostics_preserve_status_but_never_provider_details():
    trace = {"preflight": {"risk_categories": ["preflight_unavailable"], "failure": {"category": "upstream_unavailable", "http_status": 500}},
             "model_calls": [{"status": "failed", "failure": {"category": "upstream_unavailable", "http_status": 500, "body": "secret"},
                              "attempts": [{"status": "failed", "failure": {"category": "secret", "url": "secret"}}]}]}
    result = summarize({"id": "r", "thread_id": "t", "status": "completed", "telemetry": json.dumps(trace)})
    assert "PREFLIGHT_SERVICE_UNAVAILABLE" in result["findings"]
    assert "NO_DURATION_IN_COMMITTED_SYMPTOM_FACTS" not in result["findings"]
    assert result["model_calls"][0]["failure"]["http_status"] == 500
    assert "secret" not in json.dumps(result)
