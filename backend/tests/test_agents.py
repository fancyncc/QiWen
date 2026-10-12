import asyncio
import json

import pytest
from pydantic import BaseModel, ConfigDict

from app import model
from app.agents.runtime import AgentLoopError, ToolSpec, run_agent_graph


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: str
    total: int


def test_agent_graph_observes_tool_result_before_finishing(monkeypatch):
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

    run = asyncio.run(run_agent_graph(
        name="测试 Agent", role="bianzheng", task="计算", state={},
        tools=[ToolSpec("add", "加一", AddArgs, add)], output_schema=Result,
        mock_planner=planner, max_steps=3,
    ))
    assert calls == [2]
    assert run.output == {"status": "completed", "total": 3}
    assert [item["status"] for item in run.trace] == ["success", "completed"]


def test_agent_graph_denies_unregistered_tool_and_can_recover(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")

    def planner(step, observations):
        if not observations:
            return {"tool": "not_allowed", "arguments": {}}
        assert observations[-1]["status"] == "failed"
        return {"final": {"status": "completed", "total": 0}}

    run = asyncio.run(run_agent_graph(
        name="测试 Agent", role="bianzheng", task="拒绝越权工具", state={}, tools=[],
        output_schema=Result, mock_planner=planner, max_steps=3,
    ))
    assert run.trace[0]["error"] == "ValueError"
    assert run.output["total"] == 0


def test_agent_graph_enforces_repeat_and_round_limits(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    calls, steps, observations_seen = [], [], []

    def planner(step, observations):
        steps.append(step)
        observations_seen[:] = observations
        return {"tool": "add", "arguments": {"value": 2}}

    with pytest.raises(AgentLoopError, match="最大行动步数"):
        asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="循环", state={},
            tools=[ToolSpec("add", "加一", AddArgs, lambda _, args: calls.append(args.value))],
            output_schema=Result, mock_planner=planner, max_steps=4,
        ))
    assert steps == [1, 2, 3, 4]
    assert calls == [2, 2]
    assert observations_seen[-1]["error"] == "ValueError"


@pytest.mark.parametrize("fault", ["invalid_arguments", "timeout"])
def test_agent_graph_recovers_from_tool_failure_on_final_round(monkeypatch, fault):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    calls = []
    cleaned_up = []

    async def handler(_, args):
        calls.append(args.value)
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.append(True)

    def planner(step, observations):
        if step == 1:
            return {"tool": "add", "arguments": {"value": "2" if fault == "invalid_arguments" else 2}}
        assert observations[-1]["error"] == ("ValidationError" if fault == "invalid_arguments" else "TimeoutError")
        return {"final": {"status": "need_user_input", "total": 0}}

    run = asyncio.run(run_agent_graph(
        name="测试", role="bianzheng", task="故障恢复", state={},
        tools=[ToolSpec("add", "加一", AddArgs, handler, timeout=0.01)],
        output_schema=Result, mock_planner=planner, max_steps=2,
    ))
    assert run.trace[-1]["step"] == 2
    assert run.trace[-1]["stop_reason"] == "need_user_input"
    assert calls == ([] if fault == "invalid_arguments" else [2])
    assert cleaned_up == ([] if fault == "invalid_arguments" else [True])


def test_agent_graph_rejects_invalid_final_without_completed_event(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    events = []
    token = model.invocation.set(model.InvocationContext(emit=lambda kind, payload: events.append(kind)))
    try:
        with pytest.raises(AgentLoopError, match="结构化结果"):
            asyncio.run(run_agent_graph(
                name="测试", role="bianzheng", task="校验", state={}, tools=[],
                output_schema=Result, mock_planner=lambda *_: {"final": {"status": "completed"}}, max_steps=1,
            ))
    finally:
        model.invocation.reset(token)
    assert "agent.completed" not in events


def test_agent_graph_model_receives_all_tool_results_and_emits_events(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    decisions, events = [], []

    async def chat(messages, schemas, *, role):
        decisions.append(messages)
        assert role == "bianzheng"
        assert schemas[0]["function"]["name"] == "add"
        if len(decisions) == 1:
            return {"tool_calls": [{"id": f"call-{value}", "name": "add", "arguments": {"value": value}}
                                   for value in (2, 3)]}
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "tool"]
        assert [message["tool_call_id"] for message in messages[-2:]] == ["call-2", "call-3"]
        assert [json.loads(message["content"])["result"]["total"] for message in messages[-2:]] == [3, 4]
        return {"content": '{"status":"completed","total":7}'}

    monkeypatch.setattr(model, "achat_tools", chat)
    token = model.invocation.set(model.InvocationContext(emit=lambda kind, payload: events.append(kind)))
    try:
        run = asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="多工具", state={},
            tools=[ToolSpec("add", "加一", AddArgs, lambda _, args: {"total": args.value + 1})],
            output_schema=Result, mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=2,
        ))
    finally:
        model.invocation.reset(token)
    assert run.output["total"] == 7
    assert len(decisions) == 2
    assert events == ["agent.action", "agent.observation", "agent.action", "agent.observation", "agent.completed"]


def test_agent_graph_cancellation_stops_remaining_tools(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    context = model.InvocationContext()
    calls, events = [], []
    context.emit = lambda kind, payload: events.append(kind)

    def handler(_, args):
        calls.append(args.value)
        context.cancelled = True
        return {"total": args.value + 1}

    token = model.invocation.set(context)
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_agent_graph(
                name="测试", role="bianzheng", task="取消", state={},
                tools=[ToolSpec("add", "加一", AddArgs, handler)], output_schema=Result,
                mock_planner=lambda *_: {"tool_calls": [
                    {"id": f"call-{value}", "name": "add", "arguments": {"value": value}} for value in (2, 3)]},
                max_steps=2,
            ))
    finally:
        model.invocation.reset(token)
    assert calls == [2]
    assert "agent.completed" not in events
