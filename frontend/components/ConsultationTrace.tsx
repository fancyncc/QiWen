"use client";

import { useEffect, useMemo, useState } from "react";
import type { Note, StageEvent } from "@/lib/types";

const FIXED = [
  ["inquiry", "问诊"], ["search", "古籍检索"], ["bianzheng", "辨证"],
  ["formula", "方剂"], ["safety", "安全校验"],
] as const;
const MULTI = [
  ["inquiry", "问诊"], ["search", "古籍检索"], ["bianzheng", "辨证"],
  ["formula", "方剂"], ["herb", "本草"], ["synthesis", "综合"],
  ["reviewer", "安全审查"], ["safety", "安全校验"],
] as const;

export default function ConsultationTrace({
  events, notes, elapsedMs, mode, streaming,
}: {
  events: StageEvent[];
  notes: Note[];
  elapsedMs?: number;
  mode: "fixed" | "multi";
  streaming: boolean;
}) {
  const [expanded, setExpanded] = useState(streaming);
  const completed = useMemo(() => new Set(events.filter(event => event.status === "done").map(event => event.stage)), [events]);
  const started = useMemo(() => new Set(events.filter(event => event.status === "start").map(event => event.stage)), [events]);
  const hasDiagnosis = completed.has("bianzheng") || completed.has("formula") || notes.length > 0;
  useEffect(() => { setExpanded(streaming); }, [streaming]);
  if (!streaming && !hasDiagnosis) return null;
  const steps = mode === "multi" ? MULTI : FIXED;
  const completedCount = steps.filter(([key]) => completed.has(key)).length;
  return <section className="consultation-trace" aria-label="会诊轨迹">
    <button className="trace-summary" type="button" onClick={() => setExpanded(value => !value)} aria-expanded={expanded}>
      <span className="trace-emblem">✦</span>
      <strong>{streaming ? "会诊进行中" : "会诊轨迹"}</strong>
      <span className="trace-meta">{streaming ? `已完成 ${completedCount}/${steps.length} 步` : `${completedCount || steps.length} 步完成`}</span>
      {!streaming && elapsedMs !== undefined && <span className="trace-meta">{(elapsedMs / 1000).toFixed(1)} 秒</span>}
      {notes.length > 0 && <span className="trace-meta">{notes.length} 位专家</span>}
      <span className={`trace-chevron ${expanded ? "up" : ""}`}>⌄</span>
    </button>
    {expanded && <div className="trace-body">
      <div className="trace-steps">
        {steps.map(([key, label]) => {
          const status = completed.has(key) ? "done" : streaming && started.has(key) ? "running" : "pending";
          return <div className={`trace-step ${status}`} key={key}>
            <span className="trace-step-dot">{status === "done" ? "✓" : status === "running" ? "·" : ""}</span>
            <span>{label}</span>
            <small>{status === "done" ? "已完成" : status === "running" ? "进行中" : "待开始"}</small>
          </div>;
        })}
      </div>
      {notes.length > 0 && <div className="trace-notes">
        {notes.map((note, index) => <article key={index}>
          <strong>{note.expert}专家</strong><p>{note.content}</p>
        </article>)}
      </div>}
    </div>}
  </section>;
}
