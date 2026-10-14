from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

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
    format_retries: int
    retry_validation: bool
    decision_protocol: str
    decision_failed: bool


class StructuredToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StructuredDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final: dict[str, Any] | None = None
    tool_calls: list[StructuredToolCall] = Field(default_factory=list, max_length=20)


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

    async def structured_decision(current: AgentState) -> dict:
        remaining = max_steps - current["step"]
        final_round = remaining == 1 or (remaining == 2 and current["step"] > 0)
        instruction = (
            system + "\n当前使用 JSON 工具决策协议。每轮只返回一个 JSON 对象。"
            "需要工具时返回 tool（下列工具名）、arguments（参数对象）、final=null；"
            '同轮需要多次查询药材或剂量时，请返回 tool=null、tool_calls=[{"name":"lookup_herb","arguments":{"name":"药材名"}},...]，由程序逐项执行；'
            "证据足够时返回 tool=null、arguments={}、final（符合最终产物 Schema 的对象）。"
            "工具结果只会由程序执行后提供，不得编造。工具定义："
            + json.dumps(schemas, ensure_ascii=False)
        )
        instruction += "\n已有实际工具结果足够形成带不确定性的判断时，应立即返回 final；不要仅因查询措辞不同重复检索同一证据。"
        if final_round:
            instruction = (system + "\n这是最后一次决策，只输出最终产物 JSON，不再请求工具。"
                           "根据已经取得的实际资料完成任务；资料不足时返回资料待补充或受限状态，不得编造证据。")
        schema = output_schema if final_round else StructuredDecision
        from ..context_budget import budget_conversation
        # Account for escaping in the JSON envelope, including the schema that
        # acomplete_json appends. Keep patient input and the latest tool group.
        full_system = instruction + "\n只输出 JSON 对象，不要 Markdown。JSON Schema：" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        payload, dropped = budget_conversation(full_system, current["messages"], role, remaining=remaining)
        if dropped:
            instruction += "\n较早工具轮次因预算被整组省略。只能依据当前可见工具证据，不得引用已省略结果。"
        call_count = len(context.calls) if context else 0
        try:
            decision = await model.acomplete_json(
                instruction, payload, role=role,
                schema=schema, stream_output=False)
        finally:
            if context and len(context.calls) > call_count:
                context.calls[-1]["kind"] = "structured_tool_decision"
                context.calls[-1].setdefault("budget", {})["dropped_tool_call_ids"] = dropped
        if final_round:
            return {"final": decision}
        has_tools = bool(decision.get("tool") or decision.get("tool_calls"))
        if has_tools == (decision.get("final") is not None) or (decision.get("tool") and decision.get("tool_calls")):
            raise model.InvalidModelOutput(f"{name}必须选择工具或最终产物之一")
        if decision.get("tool_calls"):
            return {"tool_calls": [{"id": f"json-{name}-{current['step'] + 1}-{index}",
                                    "name": call["name"], "arguments": call["arguments"]}
                                   for index, call in enumerate(decision["tool_calls"])]}
        return decision

    async def decide(current: AgentState) -> dict:
        if context:
            context.check()
        step = current["step"] + 1
        if step > max_steps:
            raise AgentLoopError(f"{name}达到最大行动步数仍未完成任务")
        if model.MODEL_MODE == "mock":
            decision = mock_planner(step, current["observations"])
            protocol = current["decision_protocol"]
        else:
            protocol = context.tool_protocol if context else current["decision_protocol"]
            if protocol == "json":
                try:
                    decision = await structured_decision(current)
                except model.InvalidModelOutput:
                    if current["format_retries"] >= 1 or step >= max_steps:
                        raise
                    return {"step": step, "decision_failed": True,
                            "format_retries": current["format_retries"] + 1,
                            "messages": [*current["messages"], {"role": "user", "content":
                                "上次工具决策未通过 JSON/Schema 校验。请只返回符合当前 Schema 的 JSON 对象，不要 Markdown 或额外文字。"}]}
            else:
                try:
                    decision = await model.achat_tools(current["messages"], schemas, role=role)
                except RuntimeError:
                    if model.TOOL_PROTOCOL == "native":
                        raise
                    # Some compatible gateways fail on the native tools payload
                    # while ordinary strict JSON requests work. Both paths use
                    # the same model, tools, argument validation and step budget.
                    protocol = "json"
                    if context:
                        context.tool_protocol = protocol
                    decision = await structured_decision(current)

        calls = decision.get("tool_calls") or []
        if decision.get("tool"):
            calls = [{"id": f"action-{name}-{step}", "name": decision["tool"],
                      "arguments": decision.get("arguments", {})}]

        return {"step": step, "decision": decision, "calls": calls, "decision_protocol": protocol, "decision_failed": False}

    def route(current: AgentState) -> str:
        if current.get("decision_failed"):
            return "retry"
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
        if isinstance(final_value, str):
            # Accept a single JSON fence while preserving strict schema checks.
            stripped = final_value.strip()
            if stripped.startswith("```json\n") and stripped.endswith("\n```"):
                final_value = stripped[len("```json\n"):-len("\n```")].strip()
        try:
            artifact = (output_schema.model_validate(final_value) if isinstance(final_value, dict)
                        else output_schema.model_validate_json(final_value)).model_dump()
        except Exception as exc:
            if current["step"] < max_steps and current["format_retries"] < 1:
                next_protocol = "native" if model.TOOL_PROTOCOL == "native" else "json"
                if model.MODEL_MODE != "mock" and context:
                    context.tool_protocol = next_protocol
                # Give the model one bounded chance to fix serialization. This
                # changes no clinical or schema requirements and emits no result.
                return {"messages": [*current["messages"],
                                     {"role": "assistant", "content": json.dumps(final_value, ensure_ascii=False) if isinstance(final_value, dict) else str(final_value)},
                                     {"role": "user", "content": "上次最终结果未通过 JSON/Schema 校验。请保留已有证据，只修正输出格式；返回严格符合系统 Schema 的单个 JSON 对象，字符串内换行必须转义。不要使用 Markdown、说明文字或思维链。"}],
                        "format_retries": current["format_retries"] + 1, "retry_validation": True,
                        "decision_protocol": next_protocol}
            raise AgentLoopError(f"{name}未返回有效的结构化结果") from exc
        completed = {"step": current["step"], "status": "completed", "stop_reason": artifact.get("status", "completed"),
                     "protocol": current["decision_protocol"]}
        if context and context.emit:
            context.emit("agent.completed", {"agent": name, **completed})
        return {"output": artifact, "trace": [*current["trace"], completed], "retry_validation": False}

    graph = StateGraph(AgentState)
    graph.add_node("model", decide)
    graph.add_node("tools", execute_tools)
    graph.add_node("validate", validate)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", "validate": "validate", "retry": "model"})
    graph.add_edge("tools", "model")
    graph.add_conditional_edges("validate", lambda current: "retry" if current.get("retry_validation") else "done",
                                {"retry": "model", "done": END})
    # The parent workflow owns persistence; do not checkpoint raw model messages.
    try:
        result = await graph.compile(checkpointer=False).ainvoke(
            {"messages": messages, "observations": [], "trace": [], "repeats": {}, "step": 0,
             "decision": {}, "calls": [], "output": {}, "format_retries": 0, "retry_validation": False,
             "decision_protocol": context.tool_protocol if context else ("json" if model.TOOL_PROTOCOL == "json" else "native"),
             "decision_failed": False},
            config={"recursion_limit": max(2, 2 * max_steps + 2)},
        )
    except Exception as exc:
        # Some LangGraph versions wrap a node's CancelledError; keep the
        # application's cancellation contract at the subgraph boundary.
        if (context and context.cancelled) or isinstance(exc.__cause__, asyncio.CancelledError):
            raise asyncio.CancelledError from None
        raise
    return AgentRun(result["output"], result["trace"], result["observations"])
