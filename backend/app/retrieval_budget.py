"""One monotonic retrieval budget per invocation, with credential-free timings."""
from __future__ import annotations

import os
import time
import threading
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar

from .blocking import deadline

timings: ContextVar[dict | None] = ContextVar("retrieval_timings", default=None)
progress_trace: ContextVar[dict | None] = ContextVar("retrieval_progress_trace", default=None)
_timing_lock = threading.Lock()


def timeout_seconds():
    value = float(os.getenv("QIWEN_RAG_TOTAL_TIMEOUT", "30"))
    if not 0 < value <= 300:
        raise ValueError("Invalid QIWEN_RAG_TOTAL_TIMEOUT")
    return value


def invocation_deadline():
    from .model import invocation
    context = invocation.get()
    now = time.monotonic()
    if context:
        context.check()
        if context.retrieval_deadline is None:
            context.retrieval_deadline = now + timeout_seconds()
        end = context.retrieval_deadline
    else:
        end = now + timeout_seconds()
    parent = deadline.get()
    return min(end, parent) if parent is not None else end


@contextmanager
def timed_stage(name: str):
    data = timings.get()
    start = time.monotonic()
    item = {"status": "running", "elapsed_ms": 0, "_start": start}
    if data is not None:
        with _timing_lock:
            data[name] = item
    try:
        yield
        with _timing_lock:
            item["status"] = "success"
    except BaseException as exc:
        with _timing_lock:
            item.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        with _timing_lock:
            item["elapsed_ms"] = round((time.monotonic() - start) * 1000)


def snapshot(data):
    with _timing_lock:
        result = deepcopy(data)
    for item in result.values():
        start = item.pop("_start")
        if item["status"] == "running":
            item["elapsed_ms"] = round((time.monotonic() - start) * 1000)
    return result
