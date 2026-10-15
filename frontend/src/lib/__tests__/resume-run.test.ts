import { afterEach, describe, expect, it, vi } from "vitest";
import { resumeChatRun } from "../resume-run";
import type { ChatStreamEvent } from "../types";

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("恢复后台问答", () => {
  it("恢复阶段进度和已审查结果，不重新发起问答、不展示候选正文", async () => {
    vi.useFakeTimers();
    const events: ChatStreamEvent[] = [];
    const fetcher = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "running", telemetry: {
        events: [{ stage: "formula", label: "方剂", status: "start" }],
      } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "completed" }) })
      .mockResolvedValueOnce({ ok: true, json: async () => [
        { type: "candidate.delta", payload: { text: "未审查候选" } },
        { type: "answer.final", payload: { response: "最终结果", elapsed_ms: 100 } },
      ] });
    vi.stubGlobal("fetch", fetcher);
    const running = resumeChatRun("run", "thread", { onEvent: (event) => events.push(event) });
    await vi.advanceTimersByTimeAsync(3_000);
    await running;
    expect(events).toEqual([
      { type: "stage", stage: "fangji", label: "方剂", status: "start" },
      { type: "token", content: "最终结果" },
      { type: "done", thread_id: "thread", notes: undefined, elapsed_ms: 100 },
    ]);
    expect(fetcher.mock.calls.every(([url]) => !url.includes("/api/chat"))).toBe(true);
  });

  it.each(["view_changed", "user_stop"])("%s 只在明确停止时取消后台任务", async (reason) => {
    const controller = new AbortController();
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: "running", telemetry: {
      events: [{ stage: "bianzheng", label: "辨证", status: "start" }],
    } }) });
    vi.stubGlobal("fetch", fetcher);
    await resumeChatRun("run", "thread", { signal: controller.signal, onEvent: () => controller.abort(reason) });
    const cancels = fetcher.mock.calls.filter(([url]) => url.endsWith("/cancel"));
    expect(cancels).toHaveLength(reason === "user_stop" ? 1 : 0);
  });
});
