import { afterEach, describe, it, expect, vi } from "vitest";
import { renderHook, act, waitFor, cleanup } from "@testing-library/react";
import { useChat } from "@/hooks/useChat";
import type { ChatStreamEvent, MessageItem } from "@/lib/types";

// 捕获 onEvent 注入器，供用例逐个回放事件
let emit: ((ev: ChatStreamEvent) => void) | null = null;

vi.mock("@/lib/stream", () => ({
  streamChat: vi.fn(
    (_req: unknown, opts: { onEvent: (ev: ChatStreamEvent) => void }) => {
      emit = opts.onEvent;
      return Promise.resolve();
    }
  ),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    getMessages: vi.fn(() => Promise.resolve([])),
  },
  ApiError: class ApiError extends Error {},
}));

import { api } from "@/lib/api";
import { streamChat, type StreamOptions } from "@/lib/stream";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

function pendingStream() {
  let finish!: () => void;
  let options!: StreamOptions;
  vi.mocked(streamChat).mockImplementationOnce((_request, opts) => {
    options = opts;
    return new Promise<void>((resolve) => { finish = resolve; });
  });
  return { get options() { return options; }, finish: () => finish() };
}

describe("useChat request isolation", () => {
  it("late events and finally from a stopped turn cannot mutate or cancel the next turn", async () => {
    const first = pendingStream();
    const second = pendingStream();
    const { result } = renderHook(() => useChat(""));
    let firstSend!: Promise<void>, secondSend!: Promise<void>;
    await act(async () => { firstSend = result.current.send("first"); });
    act(() => result.current.stop());
    expect(first.options.signal?.aborted).toBe(true);
    await act(async () => { secondSend = result.current.send("second"); });
    act(() => {
      first.options.onEvent({ type: "token", content: "late result" });
      first.options.onEvent({ type: "done", thread_id: "wrong-thread" });
    });
    await act(async () => { first.finish(); await firstSend; });
    expect(result.current.streaming).toBe(true);
    expect(result.current.messages.at(-1)?.content).toBe("");
    expect(result.current.currentThreadId).toBe("");
    act(() => result.current.stop());
    expect(second.options.signal?.aborted).toBe(true);
    await act(async () => { second.finish(); await secondSend; });
  });

  it("switching threads aborts the old stream and ignores late events", async () => {
    vi.mocked(api.getMessages).mockImplementation(async (thread) => [{ role: "assistant", content: thread }]);
    const stream = pendingStream();
    const { result, rerender, unmount } = renderHook(({ thread }) => useChat(thread), { initialProps: { thread: "one" } });
    await waitFor(() => expect(result.current.messages[0]?.content).toBe("one"));
    let sending!: Promise<void>;
    await act(async () => { sending = result.current.send("question"); });
    rerender({ thread: "two" });
    await waitFor(() => expect(result.current.messages[0]?.content).toBe("two"));
    expect(stream.options.signal?.aborted).toBe(true);
    act(() => stream.options.onEvent({ type: "token", content: "stale" }));
    expect(result.current.messages[0].content).toBe("two");
    await act(async () => { stream.finish(); await sending; });
    const next = pendingStream();
    await act(async () => { sending = result.current.send("next"); });
    unmount();
    expect(next.options.signal?.aborted).toBe(true);
    next.finish(); await sending;
    vi.mocked(api.getMessages).mockImplementation(async () => []);
  });

  it("late initial history cannot replace a newly sent message", async () => {
    let history!: (messages: MessageItem[]) => void;
    vi.mocked(api.getMessages).mockImplementationOnce(() => new Promise((resolve) => { history = resolve; }));
    const stream = pendingStream();
    const { result } = renderHook(() => useChat("one"));
    let sending!: Promise<void>;
    await act(async () => { sending = result.current.send("new message"); });
    await act(async () => { history([{ role: "assistant", content: "old history" }]); });
    expect(result.current.messages[0].content).toBe("new message");
    await act(async () => { stream.finish(); await sending; });
  });
});

describe("useChat stages 持久化", () => {
  it("token 不抹 stages；并行专家不提前补勾；重跑重置笔记", async () => {
    const { result } = renderHook(() => useChat(""));

    await act(async () => {
      result.current.send("腹痛");
    });

    // 回放：阶段 -> token -> 并行专家（start 先于 done 到达）
    act(() => {
      emit?.({ type: "stage", stage: "inquiry", label: "问诊", status: "start" });
      emit?.({ type: "stage", stage: "inquiry", label: "问诊", status: "done" });
      emit?.({ type: "token", content: "诊" });
      emit?.({ type: "token", content: "断中" });
      emit?.({ type: "stage", stage: "bianzheng", label: "辨证", status: "start" });
      emit?.({ type: "stage", stage: "fangji", label: "方剂", status: "start" });
      emit?.({
        type: "stage",
        stage: "bianzheng",
        label: "辨证",
        status: "done",
        note: { specialist: "辨证", syndrome: "脾胃虚寒" },
      });
    });

    const last = result.current.messages[result.current.messages.length - 1];
    // token 之后 stages 仍在（回归：spread 丢失）
    expect(last.content).toBe("诊断中");
    expect(last.stages?.map((s) => s.stage)).toEqual(["inquiry", "bianzheng", "fangji"]);
    // 方剂 start 不得把仍在运行的辨证提前打成 done（假完成）
    expect(last.stages?.find((s) => s.stage === "bianzheng")?.status).toBe("done");
    expect(last.stages?.find((s) => s.stage === "fangji")?.status).toBe("start");
    expect(last.stages?.find((s) => s.stage === "bianzheng")?.note?.syndrome).toBe("脾胃虚寒");

    // 前序阶段重跑（重试）：回到 start 且笔记清空
    act(() => {
      emit?.({ type: "stage", stage: "bianzheng", label: "辨证", status: "start" });
    });
    const rerun = result.current.messages[result.current.messages.length - 1];
    expect(rerun.stages?.find((s) => s.stage === "bianzheng")?.status).toBe("start");
    expect(rerun.stages?.find((s) => s.stage === "bianzheng")?.note).toBeUndefined();

    act(() => {
      emit?.({
        type: "done",
        thread_id: "t1",
        notes: [{ specialist: "辨证" }],
        elapsed_ms: 1200,
      });
    });

    await waitFor(() => {
      const done = result.current.messages[result.current.messages.length - 1];
      expect(done.notes?.length).toBe(1);
      expect(done.elapsedMs).toBe(1200);
      // done 之后 stages 依旧完整
      expect(done.stages?.length).toBe(3);
    });
  });

  it("新一轮 send 后，迟到的上一轮 resync 被丢弃（防止覆盖新一轮消息）", async () => {
    let resolveGet!: (msgs: MessageItem[]) => void;
    vi.mocked(api.getMessages).mockImplementationOnce(
      () => new Promise<MessageItem[]>((r) => (resolveGet = r))
    );

    const { result } = renderHook(() => useChat(""));
    const streams: { onEvent: (ev: ChatStreamEvent) => void }[] = [];
    const { streamChat } = await import("@/lib/stream");
    vi.mocked(streamChat).mockImplementationOnce(
      ((_req: unknown, opts: { onEvent: (ev: ChatStreamEvent) => void }) => {
        streams.push(opts);
        return Promise.resolve();
      }) as typeof streamChat
    );

    // 第 1 轮：发送 → done（触发带守卫的 resync，getMessages 挂起）
    await act(async () => {
      await result.current.send("m1");
    });
    act(() => {
      streams[0].onEvent({ type: "done", thread_id: "t1" });
    });

    // 第 2 轮开启（turnSeq 递增）
    await act(async () => {
      await result.current.send("m2");
    });

    // 迟到的 resync 此刻才 resolve，必须被丢弃
    await act(async () => {
      resolveGet([
        { role: "user", content: "m1" },
        { role: "assistant", content: "a1" },
      ]);
    });

    // 新一轮的乐观消息必须完好：4 条 = u1,a1(流式空气泡),u2,a2
    const msgs = result.current.messages;
    expect(msgs).toHaveLength(4);
    expect(msgs[2].content).toBe("m2");
    expect(msgs[3].role).toBe("assistant");
    expect(msgs[3].content).toBe("");
  });
});
