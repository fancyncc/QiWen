"""Conservative, provider-independent budgeting immediately before dispatch."""
from __future__ import annotations

import copy
import json
import os

from .context import ContextBudgetError


def estimate(value) -> int:
    # UTF-8 byte bound is deliberately conservative; no unknown tokenizer guess.
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def budget_request(messages: list[dict], tools: list[dict], role: str):
    prefix = f"QIWEN_MODEL_{role.upper()}"
    capacity = max(1024, int(os.getenv(prefix + "_CONTEXT_LIMIT", os.getenv("QIWEN_MODEL_CONTEXT_LIMIT", "32768"))))
    reserve = max(128, int(os.getenv(prefix + "_OUTPUT_RESERVE", os.getenv("QIWEN_MODEL_OUTPUT_RESERVE", "4096"))))
    available = capacity - reserve - 512
    packed = copy.deepcopy(messages)
    dropped = []
    # Tool protocols are indivisible: assistant calls plus all following tool replies.
    while estimate({"messages": packed, "tools": tools}) > available:
        start = next((i for i, message in enumerate(packed) if message.get("tool_calls")), None)
        if start is None:
            raise ContextBudgetError()
        end = start + 1
        while end < len(packed) and packed[end].get("role") == "tool":
            end += 1
        # Never discard the most recent evidence group; required inputs must fit.
        if not any(m.get("tool_calls") for m in packed[end:]):
            raise ContextBudgetError()
        dropped.extend(c["id"] for c in packed[start].get("tool_calls", []))
        del packed[start:end]
    if dropped:
        packed[0]["content"] += "\n较早工具轮次因预算被整组省略。只能依据当前可见工具证据，不得引用已省略结果。"
    total = estimate({"messages": packed, "tools": tools})
    if total > available:
        raise ContextBudgetError()
    return packed, reserve, {"estimated_input_tokens": total, "estimation": "utf8_byte_upper_bound",
                            "context_limit": capacity, "output_reserve": reserve, "dropped_tool_call_ids": dropped}
