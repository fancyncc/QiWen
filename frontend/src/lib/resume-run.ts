import { useAuthStore } from "@/stores/auth";
import { mapNote, mapStage, type BackendNote, type BackendStage } from "./backend-contract";
import type { StreamOptions } from "./stream";

/** Reattach to an existing run after navigation/reload; never starts a new turn. */
export async function resumeChatRun(runId: string, threadId: string, opts: StreamOptions): Promise<void> {
  const token = useAuthStore.getState().token;
  const base = (process.env.NEXT_PUBLIC_API_URL || "") + `/api/runs/${runId}`;
  const controller = new AbortController();
  let seenStages = 0;
  const emit: StreamOptions["onEvent"] = (event) => { if (!opts.signal?.aborted) opts.onEvent(event); };
  const abort = () => {
    if (opts.signal?.reason !== "view_changed") {
      void fetch(`${base}/cancel`, { method: "POST", keepalive: true,
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ reason: "user_stop" }) }).catch(() => {});
    }
    controller.abort();
  };
  if (opts.signal?.aborted) return;
  opts.signal?.addEventListener("abort", abort, { once: true });
  const pause = () => new Promise<void>((resolve) => {
    const finish = () => { clearTimeout(timer); controller.signal.removeEventListener("abort", finish); resolve(); };
    const timer = setTimeout(finish, 3_000);
    controller.signal.addEventListener("abort", finish, { once: true });
    if (controller.signal.aborted) finish();
  });
  try {
    while (!opts.signal?.aborted) {
      const options: RequestInit = { headers: token ? { Authorization: `Bearer ${token}` } : {},
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8_000)]) };
      const response = await fetch(base, options);
      if (!response.ok) throw new Error("无法恢复问答进度，请刷新后重试。");
      const run: { status: string; telemetry?: { events?: BackendStage[]; cancel_reason?: string } } = await response.json();
      if (opts.signal?.aborted) return;
      const stages = run.telemetry?.events ?? [];
      for (const stage of stages.slice(seenStages)) {
        emit({ type: "stage", ...mapStage(stage) });
      }
      seenStages = stages.length;
      if (run.status === "completed") {
        const saved = await fetch(`${base}/events`, options);
        if (!saved.ok) throw new Error("结果已生成，读取失败，请刷新对话查看。");
        const events: { type: string; payload: { response?: string; message?: { content?: string };
          notes?: BackendNote[]; elapsed_ms?: number; flow?: import("./types").WorkflowPresentation } }[] = await saved.json();
        if (opts.signal?.aborted) return;
        const final = events.find((item) => item.type === "answer.final")?.payload;
        if (!final) throw new Error("结果已结束，请刷新对话查看。");
        emit({ type: "token", content: final.message?.content ?? final.response ?? "" });
        emit({ type: "done", thread_id: threadId, notes: final.notes?.map(mapNote), elapsed_ms: final.elapsed_ms,
          ...(final.flow ? { flow: final.flow } : {}) });
        return;
      }
      if (run.status === "cancelled" || run.status === "failed") {
        emit({ type: "token", content: run.telemetry?.cancel_reason === "timeout"
          ? "本轮运行超过 5 分钟，已停止，请稍后重试。" : "本轮已中断，请重新发送。" });
        emit({ type: "done", thread_id: threadId });
        return;
      }
      await pause();
    }
  } catch (error) {
    if (!opts.signal?.aborted) opts.onError?.(error instanceof Error ? error : new Error("恢复问答失败，请刷新重试。"));
  } finally {
    opts.signal?.removeEventListener("abort", abort);
    controller.abort();
  }
}
