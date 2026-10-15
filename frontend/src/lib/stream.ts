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
  let closed = false;
  let checking = false;
  let statusTimer: ReturnType<typeof setInterval> | undefined;
  const transport = new AbortController();
  const emit = (event: ChatStreamEvent) => { if (!opts.signal?.aborted) opts.onEvent(event); };
  if (opts.signal?.aborted) return;
  const cancelRun = (reason = "user_stop") => {
    void fetch((process.env.NEXT_PUBLIC_API_URL || "") + `/api/runs/${runId}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ reason }),
      keepalive: true,
    }).catch(() => {});
  };
  const stopTransport = () => {
    transport.abort();
    void reader?.cancel().catch(() => {});
  };
  const abort = () => {
    if (opts.signal?.reason !== "view_changed") cancelRun();
    stopTransport();
  };
  opts.signal?.addEventListener("abort", abort, { once: true });
  // Bound even a connection that never returns headers or closes its body.
  const deadline = setTimeout(() => {
    if (closed || terminalSeen || opts.signal?.aborted) return;
    terminalSeen = true;
    opts.onError?.(new Error("本轮等待已超过 5 分钟，已停止请求。请稍后重试。"));
    cancelRun("timeout");
    stopTransport();
  }, 300_000);

  const checkStatus = async () => {
    if (checking || closed || terminalSeen || opts.signal?.aborted) return;
    checking = true;
    try {
      const options: RequestInit = {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        signal: AbortSignal.any([transport.signal, AbortSignal.timeout(8_000)]),
      };
      const base = (process.env.NEXT_PUBLIC_API_URL || "") + `/api/runs/${runId}`;
      const response = await fetch(base, options);
      if (!response.ok) return;
      const run = await response.json();
      if (closed || terminalSeen || opts.signal?.aborted) return;
      if (run.status === "completed") {
        const eventsResponse = await fetch(`${base}/events`, options);
        if (!eventsResponse.ok) return;
        const events: { type: string; payload: {
          response?: string; message?: { content?: string }; thread_id?: string;
          notes?: Parameters<typeof mapNote>[0][]; elapsed_ms?: number; flow?: import("./types").WorkflowPresentation;
        } }[] = await eventsResponse.json();
        if (closed || terminalSeen || opts.signal?.aborted) return;
        const final = events.find((event) => event.type === "answer.final")?.payload;
        if (!final) return;
        terminalSeen = true;
        emit({ type: "token", content: final.message?.content ?? final.response ?? "" });
        emit({ type: "done", thread_id: final.thread_id ?? run.thread_id ?? "",
          notes: final.notes?.map(mapNote), elapsed_ms: final.elapsed_ms, ...(final.flow ? { flow: final.flow } : {}) });
        stopTransport();
      } else if (run.status === "cancelled" || run.status === "failed") {
        terminalSeen = true;
        emit({ type: "token", content: run.status === "cancelled"
          ? "本轮已中断，未生成最终答复。请重新发送。"
          : "本轮处理失败，未生成最终答复。请稍后重试。" });
        emit({ type: "done", thread_id: run.thread_id ?? req.thread_id ?? "" });
        stopTransport();
      }
    } catch {
      // A transient status failure must not discard a still-working SSE stream.
    } finally {
      checking = false;
    }
  };

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ ...req, run_id: runId, background: true }),
      signal: transport.signal,
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
    // Reconcile authoritative state if a proxy leaves an old stream open after restart.
    statusTimer = setInterval(() => { void checkStatus(); }, 15_000);
    const decoder = new TextDecoder();
    let buffer = "";


    while (!terminalSeen && !opts.signal?.aborted) {
      const { done, value } = await reader.read();
      if (done || terminalSeen || opts.signal?.aborted) break;
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
                status: payload.status, note: payload.note, outcome: payload.outcome,
              }),
            });
          } else if (parsed.type === "answer.final") {
            terminalSeen = true;
            emit({ type: "token", content: payload.message?.content ?? payload.response ?? "" });
            emit({
              type: "done", thread_id: payload.thread_id ?? "",
              notes: Array.isArray(payload.notes) ? payload.notes.map(mapNote) : undefined,
              elapsed_ms: payload.elapsed_ms,
              ...(payload.flow ? { flow: payload.flow } : {}),
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
    if (!terminalSeen && !opts.signal?.aborted && (err as Error).name !== "AbortError") {
      opts.onError?.(err as Error);
    }
  } finally {
    closed = true;
    clearTimeout(deadline);
    clearInterval(statusTimer);
    opts.signal?.removeEventListener("abort", abort);
    stopTransport();
    reader?.releaseLock();
  }
}
