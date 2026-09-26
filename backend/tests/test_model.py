import asyncio
import json

import httpx
import pytest

from app import model


def configure(monkeypatch, handler):
    monkeypatch.setattr(model, "MODEL_MODE", "openai_compatible")
    monkeypatch.setattr(model, "API_BASE", "https://stub.invalid/v1")
    monkeypatch.setattr(model, "API_KEY", "stub-only")
    monkeypatch.setattr(model, "MODEL_NAME", "default-model")
    client_type = httpx.AsyncClient
    monkeypatch.setattr(model.httpx, "AsyncClient", lambda **kwargs:
                        client_type(transport=httpx.MockTransport(handler), **kwargs))


def output(content="ok", usage=None):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": usage})


@pytest.mark.parametrize("role", model.ROLES)
def test_role_override_fallback_and_usage(monkeypatch, role):
    received = []
    def handler(request):
        received.append(json.loads(request.content)["model"])
        return output(usage={"prompt_tokens": 12, "completion_tokens": 3})
    configure(monkeypatch, handler)
    monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), "role-model")
    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            assert await model.acomplete("system", "user", role=role) == "ok"
            monkeypatch.setenv("QIWEN_MODEL_" + role.upper(), "")
            assert await model.acomplete("system", "user", role=role) == "ok"
            assert received == ["role-model", "default-model"]
            assert context.calls[0]["input_tokens"] == 12
            assert context.calls[0]["output_tokens"] == 3
            assert all(call["status"] == "success" and call["elapsed_ms"] >= 0 for call in context.calls)
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "503", "401", "invalid_json", "invalid_schema"])
def test_failures_are_bounded_and_structured(monkeypatch, failure):
    received = []
    async def handler(request):
        received.append(request)
        if failure == "timeout":
            await asyncio.sleep(1)
        if failure in {"503", "401"}:
            return httpx.Response(int(failure))
        return output("not-json" if failure == "invalid_json" else '{"approved":"yes","reason":""}')
    configure(monkeypatch, handler)
    monkeypatch.setenv("QIWEN_MODEL_REVIEWER_TIMEOUT", "0.05")
    monkeypatch.setenv("QIWEN_MODEL_REVIEWER_RETRIES", "2")
    async def run():
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            with pytest.raises(RuntimeError):
                await model.acomplete_json("system", "user", role="reviewer", schema=model.ReviewResult)
            expected = 3 if failure in {"timeout", "503"} else 1
            assert len(received) == expected
            call = context.calls[0]
            assert call["status"] == "failed"
            assert call["retries"] == expected - 1
            assert call["input_tokens"] is None
            assert call["elapsed_ms"] < 2500
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize("during_backoff", [False, True])
def test_cancel_closes_response_and_stops_retries(monkeypatch, during_backoff):
    closed = []
    received = []
    async def run():
        started = asyncio.Event()
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.sleep(60)
                yield b"{}"
            async def aclose(self):
                closed.append(True)
        async def handler(request):
            received.append(request)
            if during_backoff:
                started.set()
                return httpx.Response(503)
            return httpx.Response(200, stream=Body())
        configure(monkeypatch, handler)
        context = model.InvocationContext()
        token = model.invocation.set(context)
        try:
            task = asyncio.create_task(model.acomplete("s", "u", role="formula"))
            await started.wait()
            context.cancelled = True
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.3)
            assert len(received) == 1
            assert context.calls[0]["status"] == "cancelled"
            if not during_backoff:
                assert closed
        finally:
            model.invocation.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize("partial_failure", [False, True])
def test_streaming_usage_and_partial_stream_is_not_retried(monkeypatch, partial_failure):
    requests, events, closed = [], [], []
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
            if partial_failure:
                raise httpx.ReadError("stub disconnect")
            yield b'data: {"choices":[{"delta":{"content":" second"},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'
        async def aclose(self):
            closed.append(True)
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["stream"] and payload["stream_options"]["include_usage"]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())
    configure(monkeypatch, handler)
    async def run():
        context = model.InvocationContext(emit=lambda kind, payload: events.append((kind, payload)))
        token = model.invocation.set(context)
        try:
            if partial_failure:
                with pytest.raises(RuntimeError):
                    await model.acomplete("s", "u", role="bianzheng")
                assert context.calls[0]["status"] == "failed"
                assert len(events) == 1
            else:
                assert await model.acomplete("s", "u", role="bianzheng") == "first second"
                assert context.calls[0]["input_tokens"] == 17 and context.calls[0]["output_tokens"] == 2
                assert "".join(event[1]["text"] for event in events) == "first second"
            assert len(requests) == 1 and closed
        finally:
            model.invocation.reset(token)
    asyncio.run(run())
