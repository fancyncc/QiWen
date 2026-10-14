from __future__ import annotations

import asyncio
import json
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .config import API_BASE, API_KEY, MODEL_MODE, MODEL_NAME, TOOL_PROTOCOL
from .prescription import CandidatePlan

ROLES = ("inquiry", "knowledge", "safety_preflight", "summary", "profile_extraction", "retrieval_evaluation",
         "query_rewrite", "bianzheng", "formula", "herb", "synthesis", "reviewer")

# In-process test hook only. No HTTP/config switch and no prompt persistence.
input_observer: ContextVar[Callable | None] = ContextVar("evaluation_input_observer", default=None)


def observe_input(kind: str, value: dict):
    context = invocation.get()
    observer = context.observer if context is not None else input_observer.get()
    if observer is not None:
        observer({"kind": kind, "node": current_node.get(), **value})


@dataclass
class InvocationContext:
    calls: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)
    cancelled: bool = False
    emit: Callable | None = None
    observer: Callable | None = field(default_factory=lambda: input_observer.get(), repr=False)
    retrieval_deadline: float | None = None
    retrieval_terminal: str | None = None
    retrieval_cache: dict = field(default_factory=dict, repr=False)
    tool_protocol: str = "json" if TOOL_PROTOCOL == "json" else "native"

    def check(self):
        if self.cancelled:
            raise asyncio.CancelledError


invocation: ContextVar[InvocationContext | None] = ContextVar("model_invocation", default=None)
current_node: ContextVar[str | None] = ContextVar("model_node", default=None)


class InvalidModelOutput(RuntimeError):
    pass


@dataclass(frozen=True)
class RoleConfig:
    model: str
    timeout: float
    retries: int


def role_config(role: str) -> RoleConfig:
    if role not in ROLES:
        raise ValueError("未知模型角色")
    prefix = f"QIWEN_MODEL_{role.upper()}"
    timeout = os.getenv(prefix + "_TIMEOUT", "").strip() or os.getenv("QIWEN_MODEL_TIMEOUT", "").strip() or "45"
    retries = os.getenv(prefix + "_RETRIES", "").strip() or os.getenv("QIWEN_MODEL_RETRIES", "").strip() or "2"
    return RoleConfig(
        os.getenv(prefix, "").strip() or MODEL_NAME,
        min(300, max(0.05, float(timeout))),
        min(5, max(0, int(retries))),
    )


class SafetyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    category: Literal["urgent", "pregnancy_lactation", "age", "medication", "allergy", "constitution", "history"]
    entity: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=40)
    assertion: Literal["present", "absent", "uncertain"]
    temporal_scope: Literal["current", "past", "ended", "unknown"]
    quote: str = Field(min_length=1, max_length=500)


class SafetyPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["allow", "clarify", "block"]
    risk_categories: list[str] = Field(max_length=12)
    reason: str = Field(min_length=1, max_length=1000)
    required_questions: list[str] = Field(max_length=8)
    observations: list[SafetyObservation] = Field(default_factory=list, max_length=20)


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    approved: bool
    reason: str = Field(max_length=1000)


class JSONObject(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


async def acomplete(system: str, user: str, *, role: str = "inquiry", schema: type[BaseModel] | None = None, stream_output: bool = True):
    context = invocation.get()
    config = role_config(role)
    record = {"role": role, "node": current_node.get(), "model": config.model, "elapsed_ms": 0, "input_tokens": None,
              "output_tokens": None, "retries": 0, "status": "running", "attempts": []}
    if context:
        context.check()
        context.calls.append(record)
    started = time.monotonic()
    try:
        if MODEL_MODE != "openai_compatible" or not API_BASE or not API_KEY or not config.model:
            raise RuntimeError("真实模型配置不完整或未启用")
        url = API_BASE.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        if schema:
            system += "\n只输出 JSON 对象，不要 Markdown。JSON Schema：" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        payload = {"model": config.model, "temperature": 0.2,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        from .context_budget import budget_request
        payload["messages"], payload["max_tokens"], record["budget"] = budget_request(payload["messages"], [], role)
        observe_input("model_request", {"role": role, "messages": payload["messages"], "budget": record["budget"]})
        streaming = bool(stream_output and context and context.emit and role in {"bianzheng", "formula"})
        if streaming:
            payload.update(stream=True, stream_options={"include_usage": True})
        emitted = False
        # Prompt + strict local validation also supports providers without response_format.
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            for attempt in range(config.retries + 1):
                if context:
                    context.check()
                record["retries"] = attempt
                entry = {"attempt": attempt + 1, "status": "running", "input_tokens": None, "output_tokens": None}
                record["attempts"].append(entry)
                try:
                    async with asyncio.timeout(config.timeout):
                        async with client.stream("POST", url, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload) as response:
                            response.raise_for_status()
                            if streaming and "text/event-stream" in response.headers.get("content-type", ""):
                                chunks, usage, content_size = [], {}, 0
                                finished = False
                                async for line in response.aiter_lines():
                                    if context:
                                        context.check()
                                    if not line.startswith("data:"):
                                        continue
                                    raw = line[5:].strip()
                                    if raw == "[DONE]":
                                        finished = True
                                        break
                                    part = json.loads(raw)
                                    usage = part.get("usage") or usage
                                    for choice in part.get("choices", []):
                                        delta = choice.get("delta", {}).get("content") or ""
                                        if delta:
                                            chunks.append(delta)
                                            content_size += len(delta)
                                            if content_size > 200000:
                                                raise ValueError("模型正文超出长度限制")
                                            context.emit("candidate.delta", {"channel": "plan_json" if schema else "body", "text": delta})
                                            emitted = True
                                        if choice.get("finish_reason"):
                                            finished = True
                                if not finished:
                                    raise httpx.ReadError("模型流提前结束")
                                data = {"choices": [{"message": {"content": "".join(chunks)}}], "usage": usage}
                            else:
                                await response.aread()
                                data = response.json()
                    usage = data.get("usage") or {}
                    for target, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                        value = usage.get(source)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            entry[target] = value
                            record[target] = (record[target] or 0) + value
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("模型返回空正文")
                    result = schema.model_validate_json(content).model_dump() if schema else content
                    if streaming and not emitted:
                        context.emit("candidate.delta", {"channel": "plan_json" if schema else "body", "text": content})
                    entry["status"] = "success"
                    record["status"] = "success"
                    return result
                except asyncio.CancelledError:
                    entry["status"] = "cancelled"
                    raise
                except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError) as exc:
                    entry["status"] = "failed"
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                    if emitted or not retryable or attempt >= config.retries:
                        raise RuntimeError("模型调用失败，已达到重试上限") from None
                    if context:
                        context.check()
                    await asyncio.sleep(min(0.25 * 2 ** attempt, 2))
                except (ValueError, KeyError, IndexError, TypeError):
                    entry["status"] = "invalid_output"
                    raise InvalidModelOutput("模型未返回有效的结构化结果或正文") from None
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        record["elapsed_ms"] = round((time.monotonic() - started) * 1000)


async def acomplete_json(system: str, user: str, *, role: str, schema: type[BaseModel], stream_output: bool = True):
    return await acomplete(system, user, role=role, schema=schema, stream_output=stream_output)


async def achat_tools(messages: list[dict], tools: list[dict], *, role: str) -> dict:
    """Call an OpenAI-compatible model with an allow-listed tool schema.

    The LangGraph caller owns tool routing.  This adapter performs one bounded model
    decision and returns either tool calls or final text without executing any
    model-provided name directly.
    """
    context = invocation.get()
    config = role_config(role)
    record = {"role": role, "node": current_node.get(), "model": config.model, "elapsed_ms": 0,
              "input_tokens": None, "output_tokens": None, "retries": 0, "status": "running",
              "attempts": [], "kind": "tool_decision"}
    if context:
        context.check()
        context.calls.append(record)
    started = time.monotonic()
    try:
        if MODEL_MODE != "openai_compatible" or not API_BASE or not API_KEY or not config.model:
            raise RuntimeError("真实模型配置不完整或未启用")
        url = API_BASE.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {"model": config.model, "temperature": 0.1, "messages": messages,
                   "tools": tools, "tool_choice": "auto"}
        from .context_budget import budget_request
        payload["messages"], payload["max_tokens"], record["budget"] = budget_request(messages, tools, role)
        observe_input("model_request", {"role": role, "messages": payload["messages"], "budget": record["budget"]})
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            for attempt in range(config.retries + 1):
                if context:
                    context.check()
                record["retries"] = attempt
                entry = {"attempt": attempt + 1, "status": "running", "input_tokens": None,
                         "output_tokens": None}
                record["attempts"].append(entry)
                try:
                    async with asyncio.timeout(config.timeout):
                        response = await client.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
                        response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") or {}
                    for target, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                        value = usage.get(source)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            entry[target] = value
                            record[target] = (record[target] or 0) + value
                    message = data["choices"][0]["message"]
                    content = message.get("content") or ""
                    if not isinstance(content, str):
                        raise ValueError("Invalid assistant content")
                    parsed_calls = []
                    for raw in message.get("tool_calls") or []:
                        function = raw.get("function") or {}
                        arguments = function.get("arguments", "{}")
                        if not isinstance(arguments, str):
                            raise ValueError("Invalid tool arguments")
                        decoded = json.loads(arguments)
                        if not isinstance(decoded, dict):
                            raise ValueError("Tool arguments must be an object")
                        call_id, name = raw.get("id"), function.get("name")
                        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                            raise ValueError("Invalid tool call")
                        parsed_calls.append({"id": call_id, "name": name, "arguments": decoded})
                    if not parsed_calls and not content.strip():
                        raise ValueError("Model returned neither tool calls nor final content")
                    entry["status"] = record["status"] = "success"
                    return {"content": content, "tool_calls": parsed_calls}
                except asyncio.CancelledError:
                    entry["status"] = "cancelled"
                    raise
                except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError) as exc:
                    entry["status"] = "failed"
                    retryable = (not isinstance(exc, httpx.HTTPStatusError)
                                 or exc.response.status_code in {408, 429}
                                 or exc.response.status_code >= 500)
                    if not retryable or attempt >= config.retries:
                        raise RuntimeError("模型工具决策失败，已达到重试上限") from None
                    if context:
                        context.check()
                    await asyncio.sleep(min(0.25 * 2 ** attempt, 2))
                except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                    entry["status"] = "invalid_output"
                    raise RuntimeError("模型未返回有效的工具调用或正文") from None
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        record["elapsed_ms"] = round((time.monotonic() - started) * 1000)


def complete(system: str, user: str) -> str:
    """CLI compatibility; API and graph use the asynchronous interface."""
    return asyncio.run(acomplete(system, user))


def complete_json(system: str, user: str) -> dict:
    return asyncio.run(acomplete(system, user, schema=JSONObject))
