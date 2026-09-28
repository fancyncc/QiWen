import asyncio

from pydantic import BaseModel, ConfigDict

from app import model
from app.agents.runtime import ToolSpec, run_react_agent


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: str
    total: int


def test_react_agent_observes_tool_result_before_finishing(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    calls = []

    def add(_state, args):
        calls.append(args.value)
        return {"total": args.value + 1}

    def planner(step, observations):
        if not observations:
            return {"tool": "add", "arguments": {"value": 2}}
        assert observations[-1]["result"]["total"] == 3
        return {"final": {"status": "completed", "total": 3}}

    run = asyncio.run(run_react_agent(
        name="测试 Agent", role="bianzheng", task="计算", state={},
        tools=[ToolSpec("add", "加一", AddArgs, add)], output_schema=Result,
        mock_planner=planner, max_steps=3,
    ))
    assert calls == [2]
    assert run.output == {"status": "completed", "total": 3}
    assert [item["status"] for item in run.trace] == ["success", "completed"]


def test_react_agent_denies_unregistered_tool_and_can_recover(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")

    def planner(step, observations):
        if not observations:
            return {"tool": "not_allowed", "arguments": {}}
        assert observations[-1]["status"] == "failed"
        return {"final": {"status": "completed", "total": 0}}

    run = asyncio.run(run_react_agent(
        name="测试 Agent", role="bianzheng", task="拒绝越权工具", state={}, tools=[],
        output_schema=Result, mock_planner=planner, max_steps=3,
    ))
    assert run.trace[0]["error"] == "ValueError"
    assert run.output["total"] == 0
