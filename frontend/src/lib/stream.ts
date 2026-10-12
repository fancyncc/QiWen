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
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let terminalSeen = false;
  const emit = (event: ChatStreamEvent) => { if (!opts.signal?.aborted) opts.onEvent(event); };
  if (opts.signal?.aborted) return;
  const cancelRun = () => {
    void fetch((process.env.NEXT_PUBLIC_API_URL || "") + `/api/runs/${runId}/cancel`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      keepalive: true,
    }).catch(() => {});
  };
  opts.signal?.addEventListener("abort", cancelRun, { once: true });

  try {
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
      if (opts.signal?.aborted || useAuthStore.getState().token !== token) return;
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

    reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";


    while (true) {
      const { done, value } = await reader.read();
      if (done || opts.signal?.aborted) break;
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
            emit({
              type: "stage",
              ...mapStage({
                stage: payload.stage ?? "", label: payload.label ?? "",
                status: payload.status, note: payload.note,
              }),
            });
          } else if (parsed.type === "answer.final") {
            terminalSeen = true;
            emit({ type: "token", content: payload.message?.content ?? payload.response ?? "" });
            emit({
              type: "done", thread_id: payload.thread_id ?? "",
              notes: Array.isArray(payload.notes) ? payload.notes.map(mapNote) : undefined,
              elapsed_ms: payload.elapsed_ms,
            });
          } else if (parsed.type === "run.cancelled") {
            terminalSeen = true;
            emit({ type: "token", content: "已停止，本轮没有生成最终答复。" });
            emit({ type: "done", thread_id: payload.thread_id ?? "" });
          } else if (parsed.type === "run.failed") {
            terminalSeen = true;
            emit({ type: "error", message: payload.error ?? "问答暂不可用，请稍后重试" });
          } else if (parsed.token !== undefined) {
            emit({ type: "token", content: parsed.token });
          } else if (parsed.done) {
            terminalSeen = true;
            emit({
              type: "done",
              thread_id: parsed.thread_id ?? "",
              notes: Array.isArray(parsed.notes) ? parsed.notes : undefined,
              diagnosis: typeof parsed.diagnosis === "string" ? parsed.diagnosis : undefined,
              elapsed_ms: typeof parsed.elapsed_ms === "number" ? parsed.elapsed_ms : undefined,
            });
          } else if (parsed.error) {
            terminalSeen = true;
            emit({ type: "error", message: parsed.error });
          }
        } catch {
          /* 跳过无法解析的块 */
        }
      }
    }
    if (!terminalSeen && !opts.signal?.aborted) opts.onError?.(new Error("连接已中断，尚未收到最终答复，请刷新会话查看运行状态。"));
  } catch (err) {
    // AbortError 属正常中止，不报错
    if (!opts.signal?.aborted && (err as Error).name !== "AbortError") {
      opts.onError?.(err as Error);
    }
  } finally {
    opts.signal?.removeEventListener("abort", cancelRun);
    reader?.releaseLock();
  }
}
