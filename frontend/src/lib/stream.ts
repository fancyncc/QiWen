import { useAuthStore } from "@/stores/auth";
import type { ChatRequest, ChatStreamEvent } from "./types";
import { errorMessage, mapNote, mapStage } from "./backend-contract";

/**
 * SSE 流式对话：POST /api/chat?stream=true
 *
 * 适配自有后端候选、审查、阶段及最终答案协议。
 * 页面只展示 answer.final 确认的正文，未通过审查的候选不进入答复气泡。
 *
 * 用 fetch + ReadableStream 解析 SSE，逐事件回调，支持 AbortController 中止。
 */

export interface StreamOptions {
  onEvent: (event: ChatStreamEvent) => void;
  onError?: (err: Error) => void;
  signal?: AbortSignal;
}

export async function streamChat(req: ChatRequest, opts: StreamOptions): Promise<void> {
  const token = useAuthStore.getState().token;
  const url = (process.env.NEXT_PUBLIC_API_URL || "") + "/api/chat?stream=true";
  const runId = crypto.randomUUID();
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ ...req, run_id: runId }),
    signal: opts.signal,
  });

  if (res.status === 401) {
    useAuthStore.getState().logout();
    if (typeof window !== "undefined") window.location.href = "/login";
    opts.onError?.(new Error("未登录或登录已过期"));
    return;
  }
  if (!res.ok || !res.body) {
    let msg = `请求失败 (${res.status})`;
    try {
      const d = await res.json();
      msg = errorMessage(d.detail, msg);
    } catch {
      /* ignore */
    }
    opts.onError?.(new Error(msg));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const cancelRun = () => {
    void fetch((process.env.NEXT_PUBLIC_API_URL || "") + `/api/runs/${runId}/cancel`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      keepalive: true,
    }).catch(() => {});
  };
  opts.signal?.addEventListener("abort", cancelRun, { once: true });

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 以双换行分隔事件
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLine = raw
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim())
          .join("");
        if (!dataLine) continue;
        try {
          const parsed = JSON.parse(dataLine);
          const payload = parsed.payload ?? parsed;
          if (parsed.type === "stage") {
            opts.onEvent({
              type: "stage",
              ...mapStage({
                stage: payload.stage ?? "", label: payload.label ?? "",
                status: payload.status, note: payload.note,
              }),
            });
          } else if (parsed.type === "answer.final") {
            opts.onEvent({ type: "token", content: payload.message?.content ?? payload.response ?? "" });
            opts.onEvent({
              type: "done", thread_id: payload.thread_id ?? "",
              notes: Array.isArray(payload.notes) ? payload.notes.map(mapNote) : undefined,
              elapsed_ms: payload.elapsed_ms,
            });
          } else if (parsed.type === "run.cancelled") {
            opts.onEvent({ type: "token", content: "已停止，本轮没有生成最终答复。" });
            opts.onEvent({ type: "done", thread_id: payload.thread_id ?? "" });
          } else if (parsed.type === "run.failed") {
            opts.onEvent({ type: "error", message: payload.error ?? "问答暂不可用，请稍后重试" });
          } else if (parsed.token !== undefined) {
            opts.onEvent({ type: "token", content: parsed.token });
          } else if (parsed.done) {
            opts.onEvent({
              type: "done",
              thread_id: parsed.thread_id ?? "",
              notes: Array.isArray(parsed.notes) ? parsed.notes : undefined,
              diagnosis: typeof parsed.diagnosis === "string" ? parsed.diagnosis : undefined,
              elapsed_ms: typeof parsed.elapsed_ms === "number" ? parsed.elapsed_ms : undefined,
            });
          } else if (parsed.error) {
            opts.onEvent({ type: "error", message: parsed.error });
          }
        } catch {
          /* 跳过无法解析的块 */
        }
      }
    }
  } catch (err) {
    // AbortError 属正常中止，不报错
    if ((err as Error).name !== "AbortError") {
      opts.onError?.(err as Error);
    }
  } finally {
    opts.signal?.removeEventListener("abort", cancelRun);
    reader.releaseLock();
  }
}
