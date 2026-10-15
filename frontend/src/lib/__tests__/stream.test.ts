import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { streamChat } from "@/lib/stream";
import { useAuthStore } from "@/stores/auth";
import type { ChatStreamEvent } from "@/lib/types";

/** 构造 SSE 响应流（data: {json}\n\n）。 */
function makeSSEStream(events: object[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(enc.encode(body));
      controller.close();
    },
  });
}

describe("streamChat (SSE 解析)", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("切换对话仅断开流，不发送取消请求", async () => {
    const controller = new AbortController();
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, body: new ReadableStream() });
    vi.stubGlobal("fetch", fetcher);
    const running = streamChat({ message: "腹胀" }, { signal: controller.signal, onEvent: vi.fn() });
    await Promise.resolve();
    controller.abort("view_changed");
    await running;
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(JSON.parse(fetcher.mock.calls[0][1].body).background).toBe(true);
  });

  it("服务重启后旧 SSE 未关闭，也会核对已取消状态并结束等待", async () => {
    vi.useFakeTimers();
    const events: ChatStreamEvent[] = [];
    const fetcher = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, body: new ReadableStream() })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "cancelled", thread_id: "existing" }) });
    vi.stubGlobal("fetch", fetcher);
    const running = streamChat({ message: "腹胀", thread_id: "existing" }, { onEvent: (event) => events.push(event) });
    await vi.advanceTimersByTimeAsync(15_000);
    await running;
    expect(events).toEqual([
      { type: "token", content: expect.stringContaining("已中断") },
      { type: "done", thread_id: "existing" },
    ]);
    expect(fetcher.mock.calls[1][0]).toMatch(/\/api\/runs\/[^/]+$/);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("最终事件丢失时从已提交事件恢复答案，不重复显示候选内容", async () => {
    vi.useFakeTimers();
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, body: new ReadableStream() })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "completed", thread_id: "existing" }) })
      .mockResolvedValueOnce({ ok: true, json: async () => [
        { type: "candidate.delta", payload: { text: "隐藏候选" } },
        { type: "answer.final", payload: { thread_id: "existing", response: "已审查答案", elapsed_ms: 100 } },
      ] }));
    const running = streamChat({ message: "腹胀" }, { onEvent: (event) => events.push(event) });
    await vi.advanceTimersByTimeAsync(15_000);
    await running;
    expect(events).toEqual([
      { type: "token", content: "已审查答案" },
      { type: "done", thread_id: "existing", notes: undefined, elapsed_ms: 100 },
    ]);
  });

  it("持续无结果时五分钟结束请求、通知后端取消并报告超时", async () => {
    vi.useFakeTimers();
    const error = vi.fn();
    const fetcher = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, body: new ReadableStream() })
      .mockResolvedValue({ ok: true, json: async () => ({ status: "running" }) });
    vi.stubGlobal("fetch", fetcher);
    const running = streamChat({ message: "腹胀" }, { onEvent: vi.fn(), onError: error });
    await vi.advanceTimersByTimeAsync(300_000);
    await running;
    expect(error).toHaveBeenCalledTimes(1);
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringContaining("超过 5 分钟") }));
    expect(fetcher.mock.calls.at(-1)?.[0]).toMatch(/\/cancel$/);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("network errors before response headers use onError", async () => {
    const error = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await streamChat({ message: "hi" }, { onEvent: vi.fn(), onError: error });
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: "offline" }));
  });
  it("abort before response headers requests backend cancellation and ignores late data", async () => {
    const controller = new AbortController();
    const event = vi.fn(), error = vi.fn();
    let respond!: (response: object) => void;
    const fetcher = vi.fn().mockImplementationOnce(() => new Promise((resolve) => { respond = resolve; }))
      .mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetcher);
    const running = streamChat({ message: "hi" }, { signal: controller.signal, onEvent: event, onError: error });
    controller.abort();
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[1][0]).toMatch(/\/api\/runs\/.+\/cancel$/);
    respond({ ok: true, status: 200, body: makeSSEStream([{ type: "answer.final", payload: { response: "late" } }]) });
    await running;
    expect(event).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });
  it("premature EOF reports that no final answer was received", async () => {
    const error = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, body: makeSSEStream([]) }));
    await streamChat({ message: "hi" }, { onEvent: vi.fn(), onError: error });
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringContaining("尚未收到最终答复") }));
  });
  it("自有后端候选被拒绝时不显示正文，只接收最终答案", async () => {
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, body: makeSSEStream([
      { type: "candidate.delta", payload: { channel: "body", text: "应当隐藏" } },
      { type: "review.rejected", payload: {} },
      { type: "answer.final", payload: { thread_id: "original-thread", response: "最终答案",
        notes: [{ expert: "辨证", content: "意见" }], elapsed_ms: 10 } },
    ]) }));
    await streamChat({ message: "腹痛" }, { onEvent: (e) => events.push(e) });
    expect(events).toEqual([
      { type: "token", content: "最终答案" },
      { type: "done", thread_id: "original-thread", notes: [{ specialist: "辨证", reasoning: "意见" }], elapsed_ms: 10 },
    ]);
  });
  beforeEach(() => {
    useAuthStore.setState({ token: "tok" });
    vi.unstubAllGlobals();
  });

  it("按序解析 token 与 done 事件", async () => {
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: makeSSEStream([
          { token: "你好" },
          { token: "世界" },
          { done: true, thread_id: "t-123" },
        ]),
      })
    );
    await streamChat({ message: "hi" }, { onEvent: (e) => events.push(e) });
    expect(events.map((e) => e.type)).toEqual(["token", "token", "done"]);
    expect((events[0] as { content: string }).content).toBe("你好");
    expect((events[2] as { thread_id: string }).thread_id).toBe("t-123");
  });

  it("解析 error 事件", async () => {
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: makeSSEStream([{ error: "处理失败" }]),
      })
    );
    await streamChat({ message: "hi" }, { onEvent: (e) => events.push(e) });
    expect(events[0]).toEqual({ type: "error", message: "处理失败" });
  });

  it("解析 stage 阶段进度事件", async () => {
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: makeSSEStream([
          { type: "stage", stage: "bianzheng", label: "辨证", status: "start" },
          { type: "stage", stage: "bianzheng", label: "辨证", status: "done" },
        ]),
      })
    );
    await streamChat({ message: "hi" }, { onEvent: (e) => events.push(e) });
    expect(events.map((e) => e.type)).toEqual(["stage", "stage"]);
    expect(events[0]).toEqual({
      type: "stage",
      stage: "bianzheng",
      label: "辨证",
      status: "start",
    });
    expect((events[1] as { status: string }).status).toBe("done");
  });

  it("done 事件携带 notes 会诊笔记", async () => {
    const events: ChatStreamEvent[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: makeSSEStream([
          { done: true, thread_id: "t-9", notes: [{ specialist: "辨证", syndrome: "脾胃虚寒" }] },
        ]),
      })
    );
    await streamChat({ message: "hi" }, { onEvent: (e) => events.push(e) });
    const done = events[0] as { type: string; thread_id: string; notes?: unknown[] };
    expect(done.type).toBe("done");
    expect(done.thread_id).toBe("t-9");
    expect(Array.isArray(done.notes)).toBe(true);
    expect((done.notes as [{ specialist: string }])[0].specialist).toBe("辨证");
  });

  it("请求失败时触发 onError", async () => {
    let errMsg = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, body: null })
    );
    await streamChat(
      { message: "hi" },
      { onEvent: () => {}, onError: (e) => (errMsg = e.message) }
    );
    expect(errMsg).toContain("500");
  });
});
