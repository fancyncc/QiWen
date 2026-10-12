"""Cancellation fences for blocking reads moved off the event loop."""
from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar

deadline: ContextVar[float | None] = ContextVar("tool_deadline", default=None)
_stopped: ContextVar[threading.Event | None] = ContextVar("read_stopped", default=None)


def check_cancelled():
    stopped = _stopped.get()
    if stopped and stopped.is_set():
        raise asyncio.CancelledError
    end = deadline.get()
    if end is not None and time.monotonic() >= end:
        raise TimeoutError("读取超时")


def remaining_timeout(default: float) -> float:
    check_cancelled()
    end = deadline.get()
    return min(default, max(.001, end - time.monotonic())) if end is not None else default


async def run_blocking(function, *args):
    stopped = threading.Event()
    token = _stopped.set(stopped)
    def execute():
        check_cancelled()
        result = function(*args)
        check_cancelled()
        return result
    try:
        return await asyncio.to_thread(execute)
    finally:
        stopped.set()
        _stopped.reset(token)
