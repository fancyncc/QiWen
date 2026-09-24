"use client";

import ReactMarkdown from "react-markdown";
import { toast } from "sonner";
import type { Message } from "@/lib/types";
import ConsultationTrace from "./ConsultationTrace";
import PrescriptionCard, { extractPrescription } from "./PrescriptionCard";

function normalizeHeadings(content: string) {
  return content.replace(/^【([^】\n]+)】\s*$/gm, "## $1");
}

export default function MessageBubble({
  message, username, mode, streaming,
}: {
  message: Message;
  username: string;
  mode: "fixed" | "multi";
  streaming: boolean;
}) {
  if (message.role === "user") return <article className="message-row user-row">
    <div className="user-bubble">{message.content}</div>
    <span className="bubble-avatar user-avatar">{username.charAt(0).toUpperCase()}</span>
  </article>;

  const warning = /药材清单已隐藏|人工复核|安全警告/.test(message.content);
  const prescription = extractPrescription(message.content);
  const normalized = normalizeHeadings(message.content);
  const footerIndex = normalized.search(/\n>\s*仅供学习/);
  const body = prescription
    ? normalized.slice(0, footerIndex < 0 ? undefined : footerIndex)
        .replace(/##\s*方剂知识示例[\s\S]*$/, "").trim()
    : normalized;
  const footer = prescription && footerIndex >= 0 ? normalized.slice(footerIndex).trim() : "";
  const inferredSteps = (message.intent_type === "diagnose" || message.notes?.length)
    ? (mode === "multi"
      ? ["inquiry", "search", "bianzheng", "formula", "herb", "synthesis", "reviewer", "safety"]
      : ["inquiry", "search", "bianzheng", "formula", "safety"])
    : [];
  const traceEvents = message.trace?.events?.length
    ? message.trace.events
    : inferredSteps.map(stage => ({ type: "stage" as const, stage, label: stage, status: "done" as const }));
  async function copy() {
    try { await navigator.clipboard.writeText(message.content); toast.success("已复制回答"); }
    catch { toast.error("复制失败，请手动选择文本"); }
  }
  return <article className="message-row assistant-row">
    <span className="bubble-avatar assistant-avatar" aria-hidden="true">岐</span>
    <div className={`assistant-bubble ${warning ? "has-warning" : ""}`}>
      <div className="assistant-bubble-head"><strong>岐问</strong><small>中医知识助手</small></div>
      <ConsultationTrace
        events={traceEvents}
        notes={message.notes || []}
        elapsedMs={message.trace?.elapsedMs}
        mode={message.trace?.mode || mode}
        streaming={streaming}
      />
      {warning && <div className="safety-banner"><span>!</span>候选方案需要人工安全复核</div>}
      <div className="markdown-body">
        {message.content ? <ReactMarkdown>{body}</ReactMarkdown> : <p className="typing-state">正在整理答复<span>···</span></p>}
      </div>
      <PrescriptionCard content={message.content} />
      {footer && <div className="markdown-body prescription-footer"><ReactMarkdown>{footer}</ReactMarkdown></div>}
      {!streaming && message.content && <button className="copy-button" type="button" onClick={copy} aria-label="复制回答">复制</button>}
    </div>
  </article>;
}
