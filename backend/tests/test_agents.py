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


def test_agent_graph_repairs_invalid_json_once_within_round_limit(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    calls = []

    async def chat(messages, schemas, *, role):
        calls.append(messages)
        return {"content": '{"status":"completed",\n"total":3,}'}

    async def structured(system, user, **kwargs):
        calls.append(json.loads(user)["conversation"])
        assert "字符串内换行必须转义" in calls[-1][-1]["content"]
        assert kwargs["stream_output"] is False
        assert kwargs["schema"] is Result
        return {"status": "completed", "total": 3}

    monkeypatch.setattr(model, "achat_tools", chat)
    monkeypatch.setattr(model, "acomplete_json", structured)
    run = asyncio.run(run_agent_graph(
        name="测试", role="bianzheng", task="格式恢复", state={}, tools=[], output_schema=Result,
        mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=2,
    ))
    assert run.output["total"] == 3
    assert len(calls) == 2
    assert len(run.trace) == 1 and run.trace[0]["step"] == 2


def test_structured_tool_protocol_preserves_tools_and_observations(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    calls, tools = [], []
    async def unavailable(*args, **kwargs):
        raise RuntimeError("native tools unavailable")
    async def structured(system, user, **kwargs):
        messages = json.loads(user)["conversation"]
        calls.append(messages)
        if len(calls) == 1:
            return {"tool": "add", "arguments": {"value": 2}, "final": None}
        assert json.loads(messages[-1]["content"])["result"]["total"] == 3
        assert kwargs["schema"] is Result
        assert "最后一次决策" in system
        return {"status": "completed", "total": 3}
    monkeypatch.setattr(model, "achat_tools", unavailable)
    monkeypatch.setattr(model, "acomplete_json", structured)
    context = model.InvocationContext()
    token = model.invocation.set(context)
    try:
        run = asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="JSON 工具兼容", state={},
            tools=[ToolSpec("add", "加一", AddArgs, lambda _, args: (tools.append(args.value) or {"total": args.value + 1}))],
            output_schema=Result, mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=2,
        ))
        assert context.tool_protocol == "json"
    finally:
        model.invocation.reset(token)
    assert tools == [2] and run.output["total"] == 3
    assert run.trace[-1]["protocol"] == "json"


def test_json_conversation_budget_drops_old_pairs_but_preserves_patient_and_latest(monkeypatch):
    from app.context_budget import budget_conversation
    from app.context import ContextBudgetError
    monkeypatch.setenv("QIWEN_MODEL_CONTEXT_LIMIT", "5000")
    monkeypatch.setenv("QIWEN_MODEL_OUTPUT_RESERVE", "128")
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "critical patient facts"}]
    for index in range(3):
        messages.extend([{"role": "assistant", "tool_calls": [{"id": str(index), "function": {"name": "lookup"}}]},
                         {"role": "tool", "tool_call_id": str(index), "content": '"evidence"' * 150}])
    payload, dropped = budget_conversation("instructions", messages, "bianzheng", remaining=2)
    kept = json.loads(payload)["conversation"]
    assert kept[0]["content"] == "critical patient facts"
    assert kept[-1]["tool_call_id"] == "2" and dropped
    assert all(m.get("tool_call_id") not in dropped for m in kept)
    with pytest.raises(ContextBudgetError):
        budget_conversation("instructions", [{"role": "user", "content": "critical" * 1000}], "bianzheng", remaining=1)


def test_structured_protocol_executes_batched_tools_within_one_round(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    context = model.InvocationContext(tool_protocol="json")
    seen = []
    async def structured(system, user, **kwargs):
        if kwargs["schema"] is Result:
            messages = json.loads(user)["conversation"]
            assert [json.loads(m["content"])["result"]["total"] for m in messages if m["role"] == "tool"] == [3, 4]
            return {"status": "completed", "total": 7}
        return {"tool": None, "final": None, "tool_calls": [
            {"name": "add", "arguments": {"value": 2}}, {"name": "add", "arguments": {"value": 3}}]}
    monkeypatch.setattr(model, "acomplete_json", structured)
    token = model.invocation.set(context)
    try:
        run = asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="批量工具", state={},
            tools=[ToolSpec("add", "加一", AddArgs, lambda _, args: (seen.append(args.value) or {"total": args.value + 1}))],
            output_schema=Result, mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=2,
        ))
    finally:
        model.invocation.reset(token)
    assert seen == [2, 3] and run.output["total"] == 7


def test_malformed_json_tool_decision_gets_one_bounded_retry(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    calls = []
    async def structured(system, user, **kwargs):
        calls.append(kwargs["schema"])
        if len(calls) == 1:
            raise model.InvalidModelOutput("invalid decision")
        assert "上次工具决策未通过" in json.loads(user)["conversation"][-1]["content"]
        return {"status": "completed", "total": 3}
    monkeypatch.setattr(model, "acomplete_json", structured)
    token = model.invocation.set(model.InvocationContext(tool_protocol="json"))
    try:
        run = asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="格式修正", state={}, tools=[], output_schema=Result,
            mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=3,
        ))
    finally:
        model.invocation.reset(token)
    assert run.output["total"] == 3 and len(calls) == 2
    assert run.trace[-1]["step"] == 2


def test_agent_graph_does_not_loop_on_repeated_invalid_results(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    steps = []
    def invalid(step, observations):
        steps.append(step)
        return {"content": "invalid json"}
    with pytest.raises(AgentLoopError, match="结构化结果"):
        asyncio.run(run_agent_graph(
            name="测试", role="bianzheng", task="限制格式修正", state={}, tools=[], output_schema=Result,
            mock_planner=invalid, max_steps=6,
        ))
    assert steps == [1, 2]


def test_ambiguous_json_decision_uses_fresh_final_schema_after_real_observations(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    calls, executed = [], []
    async def structured(system, user, **kwargs):
        calls.append(kwargs["schema"])
        if len(calls) == 1:
            return {"tool": "add", "arguments": {"value": 2}, "final": None}
        if len(calls) == 2:
            # The premature final must never be accepted as the result.
            return {"tool": "add", "arguments": {"value": 9}, "final": {"status": "completed", "total": 999}}
        assert kwargs["schema"] is Result and "最后一次决策" in system
        payload = json.loads(user)["conversation"]
        assert any(m.get("role") == "tool" and json.loads(m["content"])["result"]["total"] == 3 for m in payload)
        return {"status": "completed", "total": 3}
    monkeypatch.setattr(model, "acomplete_json", structured)
    token = model.invocation.set(model.InvocationContext(tool_protocol="json"))
    try:
        run = asyncio.run(run_agent_graph(name="格式恢复", role="herb", task="保留实际证据", state={},
            tools=[ToolSpec("add", "加一", AddArgs, lambda _, args: (executed.append(args.value) or {"total": args.value + 1}))],
            output_schema=Result, mock_planner=lambda *_: pytest.fail("unexpected mock"), max_steps=6))
    finally:
        model.invocation.reset(token)
    assert len(calls) == 3 and executed == [2] and run.output["total"] == 3


def test_null_optional_tool_wrappers_are_empty_without_coercing_clinical_values():
    from app.agents.runtime import StructuredDecision
    from pydantic import ValidationError
    value = StructuredDecision.model_validate({"tool": None, "arguments": None, "tool_calls": None,
                                               "final": {"status": "completed", "total": 3}})
    assert value.arguments == {} and value.tool_calls == [] and value.final["total"] == 3
    assert StructuredDecision.model_validate({"tool_calls": [{"name": "lookup", "arguments": None}]}).tool_calls[0].arguments == {}
    with pytest.raises(ValidationError):
        StructuredDecision.model_validate({"tool_calls": "null"})
    with pytest.raises(ValidationError):
        Result.model_validate({"status": "completed", "total": "3"})


def test_agent_graph_accepts_fenced_json_without_weakening_schema(monkeypatch):
    monkeypatch.setattr(model, "MODEL_MODE", "mock")
    run = asyncio.run(run_agent_graph(
        name="测试", role="bianzheng", task="兼容 JSON 围栏", state={}, tools=[], output_schema=Result,
        mock_planner=lambda *_: {"content": '```json\n{"status":"completed","total":3}\n```'}, max_steps=1,
    ))
    assert run.output["total"] == 3


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
