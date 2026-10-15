"""Distinct bounded search retries must not invent patient findings or select a prescription."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.agents import tools
from app.main import app
from app.terms import search_plan


def test_clinical_retry_plans_are_distinct_and_stay_with_reported_findings():
    state = {"symptoms": ["腹胀", "怕冷", "大便稀"], "safety_state": {"subject": "self"}}
    plans = [search_plan(state, attempt=i) for i in range(3)]
    assert len({(p["keyword_query"], p["semantic_query"]) for p in plans}) == 3
    assert plans[1]["lexical_terms"] == state["symptoms"]
    assert set(plans[2]["lexical_terms"]) <= set(state["symptoms"])
    assert all("脾肾阳虚" not in p["keyword_query"] for p in plans)


def test_multi_classics_no_match_retries_distinct_queries_once_inside_tool(monkeypatch):
    seen = []
    async def retrieve(plan):
        seen.append(plan)
        return [], {"outcome": "no_match", "warning": "", "retryable": True, "selected": []}
    monkeypatch.setattr(tools, "retrieve_classics", retrieve)
    result = asyncio.run(tools.search_classics({"symptoms": ["腹胀", "怕冷", "大便稀"]}, tools.SearchArgs(query="腹胀")))
    assert len(seen) == 3
    assert len({(p["keyword_query"], p["semantic_query"]) for p in seen}) == 3
    assert result["status"] == "no_match" and not result["documents"]
    assert len(result["retrieval_trace"]["query_attempts"]) == 3


@pytest.mark.parametrize("outcome", ["timeout", "dependency_error", "model_unavailable"])
def test_dependency_failures_are_not_retried_as_missing_patient_information(monkeypatch, outcome):
    seen = []
    async def retrieve(plan):
        seen.append(plan)
        return [], {"outcome": outcome, "warning": "系统检索未完成", "retryable": False, "selected": []}
    monkeypatch.setattr(tools, "retrieve_classics", retrieve)
    result = asyncio.run(tools.search_classics({"symptoms": ["腹胀", "怕冷"]}, tools.SearchArgs(query="腹胀")))
    assert len(seen) == 1 and result["status"] == outcome


def test_formula_sources_prioritize_explicit_names_without_claiming_personal_selection():
    with TestClient(app):
        result = tools.search_formulas({"symptoms": ["腹胀", "怕冷", "大便稀"]}, tools.SearchArgs(query="理中丸"))
        assert result["formulas"][0]["name"] == "理中丸"
        assert all(row["selection_status"] == "requires_clinical_comparison" for row in result["formulas"])
        assert not tools.search_formulas({}, tools.SearchArgs(query="不存在的合成方剂ZXQ"))["formulas"]
