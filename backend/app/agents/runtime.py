from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from .. import model
from ..blocking import deadline, run_blocking


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


ToolHandler = Callable[[dict, BaseModel], Any | Awaitable[Any]]
MockPlanner = Callable[[int, list[dict]], dict]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    timeout: float = 10.0

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }}


@dataclass
class AgentRun:
    output: dict
    trace: list[dict]
    observations: list[dict]


class AgentLoopError(RuntimeError):
    pass


def _brief(value: Any, limit: int = 1800) -> Any:
    from ..context import project_observation
    return project_observation(value, limit)


async def _execute_tool(spec: ToolSpec, state: dict, arguments: dict) -> Any:
    parsed = spec.args_schema.model_validate(arguments)
    context = model.invocation.get()
    if context:
        context.check()
    token = deadline.set(time.monotonic() + spec.timeout)
    try:
        async with asyncio.timeout(spec.timeout):
            result = (spec.handler(state, parsed) if inspect.iscoroutinefunction(spec.handler)
                      else await run_blocking(spec.handler, state, parsed))
            if inspect.isawaitable(result):
                result = await result
    finally:
        deadline.reset(token)
    if context:
        context.check()
    return result


class AgentState(TypedDict):
    messages: list[dict]
    observations: list[dict]
    trace: list[dict]
    repeats: dict[str, int]
    step: int
    decision: dict
    calls: list[dict]
    output: dict


async def run_agent_graph(
    *,
    name: str,
    role: str,
    task: str,
    state: dict,
    tools: list[ToolSpec],
    output_schema: type[BaseModel],
    mock_planner: MockPlanner,
    max_steps: int,
) -> AgentRun:
    """Run model -> tools -> model routing in a bounded LangGraph subgraph.

    Raw chain-of-thought is neither requested nor stored.  The durable trace
    contains only selected actions, tool observations and the stop reason.
    """
    registry = {tool.name: tool for tool in tools}
    schemas = [tool.openai_schema() for tool in tools]
    system = (
        f"你是{name}。围绕任务自主选择允许的工具。每次收到工具结果后判断证据是否足够；"
        "不足时继续调用工具，需要用户补充时在最终 JSON 中返回 need_user_input。"
        "不得编造工具结果、出处、剂量或安全结论。不要输出思维链。"
        "结束时只输出符合以下 JSON Schema 的对象："
        + json.dumps(output_schema.model_json_schema(), ensure_ascii=False)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    context = model.invocation.get()

    async def decide(current: AgentState) -> dict:
        if context:
            context.check()
        step = current["step"] + 1
        if step > max_steps:
            raise AgentLoopError(f"{name}达到最大行动步数仍未完成任务")
        if model.MODEL_MODE == "mock":
            decision = mock_planner(step, current["observations"])
        else:
            decision = await model.achat_tools(current["messages"], schemas, role=role)

        calls = decision.get("tool_calls") or []
        if decision.get("tool"):
            calls = [{"id": f"mock-{name}-{step}", "name": decision["tool"],
                      "arguments": decision.get("arguments", {})}]

        return {"step": step, "decision": decision, "calls": calls}

    def route(current: AgentState) -> str:
        return "tools" if current["calls"] else "validate"

    async def execute_tools(current: AgentState) -> dict:
        messages = list(current["messages"])
        observations = list(current["observations"])
        trace = list(current["trace"])
        repeats = Counter(current["repeats"])
        assistant_calls = [{"id": call["id"], "type": "function", "function": {
            "name": call["name"], "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}}
            for call in current["calls"]]
        messages.append({"role": "assistant", "content": current["decision"].get("content") or None,
                         "tool_calls": assistant_calls})
        for call in current["calls"]:
            if context:
                context.check()
            tool_name = call["name"]
            arguments = call.get("arguments") or {}
            action = {"step": current["step"], "tool": tool_name, "arguments": arguments}
            if context and context.emit:
                context.emit("agent.action", {"agent": name, **action})
            key = tool_name + ":" + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            repeats[key] += 1
            try:
                if tool_name not in registry:
                    raise ValueError("工具不在当前 Agent 的允许列表中")
                if repeats[key] > 2:
                    raise ValueError("同一工具和参数重复次数超过限制")
                result = await _execute_tool(registry[tool_name], state, arguments)
                if context:
                    context.check()
                observation = {**action, "status": "success", "result": result}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                observation = {**action, "status": "failed", "error": type(exc).__name__}
            observations.append(observation)
            public_observation = _brief(observation)
            trace.append(public_observation)
            if context and context.emit:
                context.emit("agent.observation", {"agent": name, "observation": public_observation})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content":
                             json.dumps(public_observation, ensure_ascii=False, default=str)})
        return {"messages": messages, "observations": observations, "trace": trace, "repeats": dict(repeats)}

    async def validate(current: AgentState) -> dict:
        if context:
            context.check()
        decision = current["decision"]
        final_value = decision.get("final")
        if final_value is None:
            final_value = decision.get("content", "")
        try:
            artifact = (output_schema.model_validate(final_value) if isinstance(final_value, dict)
                        else output_schema.model_validate_json(final_value)).model_dump()
        except Exception as exc:
            raise AgentLoopError(f"{name}未返回有效的结构化结果") from exc
        completed = {"step": current["step"], "status": "completed", "stop_reason": artifact.get("status", "completed")}
        if context and context.emit:
            context.emit("agent.completed", {"agent": name, **completed})
        return {"output": artifact, "trace": [*current["trace"], completed]}

    graph = StateGraph(AgentState)
    graph.add_node("model", decide)
    graph.add_node("tools", execute_tools)
    graph.add_node("validate", validate)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", "validate": "validate"})
    graph.add_edge("tools", "model")
    graph.add_edge("validate", END)
    # The parent workflow owns persistence; do not checkpoint raw model messages.
    try:
        result = await graph.compile(checkpointer=False).ainvoke(
            {"messages": messages, "observations": [], "trace": [], "repeats": {}, "step": 0,
             "decision": {}, "calls": [], "output": {}},
            config={"recursion_limit": max(2, 2 * max_steps + 2)},
        )
    except Exception as exc:
        # Some LangGraph versions wrap a node's CancelledError; keep the
        # application's cancellation contract at the subgraph boundary.
        if (context and context.cancelled) or isinstance(exc.__cause__, asyncio.CancelledError):
            raise asyncio.CancelledError from None
        raise
    return AgentRun(result["output"], result["trace"], result["observations"])
