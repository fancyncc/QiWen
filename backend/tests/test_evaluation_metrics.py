import math

from evaluation.supplement import dependency_errors, proportion, ranked_metrics


def test_ranked_recall_is_not_query_hit_rate():
    metrics = ranked_metrics([True, False, True], total_relevant=4, k=3)
    assert metrics == {"hit": 1, "precision": 2 / 3, "recall": .5}
    assert ranked_metrics([True], 1, 5)["precision"] == .2
    assert ranked_metrics([], 0, 5)["recall"] is None


def test_wilson_interval_does_not_claim_certainty_for_small_perfect_sample():
    metrics = proportion(30, 30)
    assert metrics["rate"] == 1
    assert math.isclose(metrics["wilson_95"][0], .8864866, abs_tol=1e-6)
    assert proportion(0, 0)["rate"] is None


def test_dependencies_allow_safety_revision_without_cross_round_comparison():
    def event(stage, status):
        return {"stage": stage, "status": status}
    events = [event("bianzheng", "done")]
    for _ in range(2):
        for stage in ("formula", "herb", "synthesis", "reviewer", "safety"):
            events.extend([event(stage, "start"), event(stage, "done")])
    assert dependency_errors(events, "multi") == []
    assert "synthesis_before_current_revision_experts" in dependency_errors(
        events + [event("synthesis", "start")], "multi")


def test_quality_failures_remain_in_success_denominator(monkeypatch):
    from evaluation import benchmark
    class Judge:
        def json(self, *_args):
            return {axis: 5 for axis in ("accuracy", "completeness", "clarity", "safety", "relevance")}, {}
    monkeypatch.setattr(benchmark, "load_jsonl", lambda *_args: [
        {"input": "ok", "expected": "ok"}, {"input": "failed", "expected": "ok"}])
    monkeypatch.setattr(benchmark, "register", lambda _client: {})
    monkeypatch.setattr(benchmark, "chat", lambda _client, _headers, message: {
        "status_code": 200 if message == "ok" else 503, "data": {"response": "ok"}})
    result = benchmark.run_quality(None, Judge(), None)
    assert result["metrics"]["pass_rate_ge_3"] == .5
    assert result["scoring_coverage"] == .5
    assert result["failed"] == 1
