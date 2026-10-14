"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, Loader2, Stethoscope } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import type { ConsultationNote, Stage } from "@/lib/types";

/** 固定阶段顺序（与后端 graph_multiagent STAGE_LABELS 一致）。 */
const STAGE_ORDER: { stage: string; label: string }[] = [
  { stage: "inquiry", label: "问诊" },
  { stage: "bianzheng", label: "辨证" },
  { stage: "fangji", label: "方剂" },
  { stage: "bencao", label: "本草" },
  { stage: "synthesis", label: "综合" },
  { stage: "reviewer", label: "安全审查" },
  { stage: "safety_check", label: "安全校验" },
];

/** 专家按辨证、方剂、本草的证据依赖顺序执行。 */
const SPECIALIST_STAGES = new Set(["bianzheng", "fangji", "bencao"]);

type CellStatus = "pending" | "active" | "done";

/** 只整理实际收到的阶段，不把未执行的专家阶段补成已完成。 */
function reconcileStages(stages: Stage[]): Stage[] {
  const seen = new Map(stages.map((s) => [s.stage, s]));
  return STAGE_ORDER.flatMap((stage) => {
    const actual = seen.get(stage.stage);
    return actual ? [actual] : [];
  });
}

/** 由 per-message stages 数组推导每个固定阶段的显示状态。 */
function deriveStatus(stageId: string, stages: Stage[]): CellStatus {
  const found = stages.find((s) => s.stage === stageId);
  if (!found) return "pending";
  return found.status === "done" ? "done" : "active";
}

/** 步骤状态图标：done=勾、active=旋转、pending=暗点。 */
function StepIcon({ status }: { status: CellStatus }) {
  if (status === "done") return <Check className="h-3 w-3" aria-hidden />;
  if (status === "active")
    return <Loader2 className="h-3 w-3 animate-spin" aria-hidden />;
  return <span className="h-1.5 w-1.5 rounded-full bg-current opacity-40" aria-hidden />;
}

/** 从阶段条目中挑选一行式结论摘要（渐进揭示用），无内容返回 null。 */
function noteSummaryText(note?: ConsultationNote): string | null {
  if (!note) return null;
  if (note.syndrome) {
    const conf =
      typeof note.confidence === "number" ? ` · 置信度 ${Math.round(note.confidence * 100)}%` : "";
    return `证候：${note.syndrome}${conf}`;
  }
  if (note.recommended_formulas?.length) {
    return `推荐方：${note.recommended_formulas.join("、")}`;
  }
  if (note.safety_warnings?.length) {
    return `安全警告：${note.safety_warnings.join("；")}`;
  }
  if (note.issues?.length) {
    return `问题：${note.issues.join("；")}`;
  }
  if (note.suggestions) {
    return `建议：${note.suggestions}`;
  }
  return null;
}

/** 专家阶段完成后的单行结论卡。 */
function NoteSummaryLine({ note }: { note?: ConsultationNote }) {
  const text = noteSummaryText(note);
  if (!text) return null;
  const isWarning = Boolean(note?.safety_warnings?.length);
  return (
    <div
      className={`ml-5 truncate border-l border-primary/20 py-0.5 pl-2 text-[11px] ${
        isWarning ? "text-destructive" : "text-muted-foreground"
      }`}
      title={text}
    >
      {text}
    </div>
  );
}

/** 单个步骤行。 */
function StepRow({
  stage,
  label,
  stages,
}: {
  stage: string;
  label: string;
  stages: Stage[];
}) {
  const status = deriveStatus(stage, stages);
  const note = stages.find((s) => s.stage === stage)?.note;
  return (
    <div>
      <div
        className={`flex items-center gap-1.5 py-0.5 text-xs ${
          status === "pending" ? "text-muted-foreground/60" : "text-foreground"
        }`}
      >
        <StepIcon status={status} />
        {label}
      </div>
      {status === "done" && <NoteSummaryLine note={note} />}
    </div>
  );
}

/** 按流程顺序展示实际启动过的阶段。 */
function StepList({ stages }: { stages: Stage[] }) {
  const actual = STAGE_ORDER.filter((stage) => stages.some((item) => item.stage === stage.stage));
  const specialists = actual.filter((s) => SPECIALIST_STAGES.has(s.stage));
  const sequential = actual.filter(
    (s) => !SPECIALIST_STAGES.has(s.stage) && s.stage !== "inquiry"
  );

  return (
    <div className="space-y-0.5">
      {actual.some((stage) => stage.stage === "inquiry") && <StepRow stage="inquiry" label="问诊" stages={stages} />}
      {specialists.length > 0 && <div className="ml-1 border-l-2 border-primary/20 pl-3">
        <p className="py-0.5 text-[11px] text-muted-foreground">依次会诊</p>
        {specialists.map((s) => (
          <StepRow key={s.stage} stage={s.stage} label={s.label} stages={stages} />
        ))}
      </div>}
      {sequential.map((s) => (
        <StepRow key={s.stage} stage={s.stage} label={s.label} stages={stages} />
      ))}
    </div>
  );
}

/** 字段渲染助手：仅当值非空时渲染一行。 */
function Field({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="whitespace-pre-wrap break-words text-[13px] leading-6">
      <span className="text-muted-foreground">{label}：</span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

/** 渲染单个会诊笔记。 */
function NoteCard({ note }: { note: ConsultationNote }) {
  const isReviewer =
    note.approved !== undefined ||
    (note.issues && note.issues.length > 0 && !note.syndrome);
  const specialist = note.specialist || "专家";

  return (
    <div className="rounded-xl border border-border/70 bg-background/70 p-3.5">
      <div className="mb-1.5 flex items-center gap-1.5">
        <Badge
          variant={isReviewer ? "outline" : "secondary"}
          className="text-[11px] font-normal"
        >
          {specialist}
        </Badge>
        {note.approved === true && (
          <Badge variant="secondary" className="text-[11px] text-primary">
            通过
          </Badge>
        )}
        {note.approved === false && (
          <Badge variant="secondary" className="text-[11px] text-destructive">
            未通过
          </Badge>
        )}
        {typeof note.confidence === "number" ? (
          <span className="text-xs text-muted-foreground" title="模型自评，未经临床校准，不代表诊断准确率">
            置信度 {Math.round(note.confidence * 100)}%
          </span>
        ) : null}
      </div>
      <Field label="证候" value={note.syndrome} />
      {note.recommended_formulas && note.recommended_formulas.length > 0 && (
        <div className="text-xs leading-relaxed">
          <span className="text-muted-foreground">推荐方剂：</span>
          <span className="text-foreground">
            {note.recommended_formulas.join("、")}
          </span>
        </div>
      )}
      {((note.safety_warnings?.length ?? 0) > 0 || /本草/.test(specialist)) && (
        <div className="my-2 rounded-lg border border-destructive/20 bg-destructive/5 p-2.5 text-[13px] leading-6 text-destructive" role="note" aria-label="本草安全警告">
          <span className="font-medium">安全警告：</span>
          <span>{note.safety_warnings?.length ? note.safety_warnings.map((warning) => warning.replace(/[；;。]+$/, "")).join("；") + "。" : "尚未确认个人用药适用性，请勿自行抓药或调整用药。"}</span>
        </div>
      )}
      {note.issues && note.issues.length > 0 && (
        <div className="text-xs leading-relaxed text-destructive">
          <span>问题：{note.issues.join("；")}</span>
        </div>
      )}
      <Field label="加减" value={note.modifications} />
      <Field label="建议" value={note.suggestions} />
      <Field label="分析摘要" value={note.reasoning} />
      {/辨证/.test(specialist) && typeof note.confidence === "number" && <p className="mt-2 text-[11px] text-muted-foreground">置信度为模型自评，不代表诊断准确率。</p>}
    </div>
  );
}

/**
 * 会诊轨迹块（方案 A：推理轨迹）。
 *
 * - 追问/纯问诊轮（仅问诊阶段、无笔记）：流式中单行"问诊中"，完成后不渲染
 *   （答案本身即是结果，无消失问题）。
 * - 会诊轮：流式中垂直展示实时步骤（并行专家簇缩进分组）；
 *   完成后自动折叠为永久摘要条，可展开查看步骤与会诊笔记。
 */
export function ConsultationTrace({
  stages,
  notes,
  streaming = false,
  elapsedMs,
}: {
  stages?: Stage[];
  notes?: ConsultationNote[];
  streaming?: boolean;
  elapsedMs?: number;
}) {
  const hasSpecialists = (stages ?? []).some((s) => SPECIALIST_STAGES.has(s.stage));
  const [open, setOpen] = useState(!streaming && (notes?.length ?? 0) >= 3);
  const wasStreaming = useRef(streaming);
  const contentId = useId();

  // 流式结束自动折叠（用户仍可手动展开）
  useEffect(() => {
    if (wasStreaming.current && !streaming) setOpen((notes?.length ?? 0) >= 3);
    wasStreaming.current = streaming;
  }, [streaming, notes?.length]);

  if (!stages?.length && !notes?.length) return null;

  // 追问/纯问诊轮：流式中单行提示，完成后由答案接管
  if (!hasSpecialists && !notes?.length) {
    const inquiry = stages?.some((stage) => stage.stage === "inquiry");
    const preflight = stages?.some((stage) => stage.stage === "pre_safety");
    if (!streaming) {
      if (!preflight) return null;
      const knowledge = stages?.some((stage) => stage.stage === "knowledge_search" || stage.stage === "knowledge_answer");
      return (
        <div className="mb-2 rounded-md bg-muted/60 px-3 py-2 text-xs text-muted-foreground" role="status">
          流程进度：{knowledge ? "知识解答" : inquiry ? "问诊资料确认" : "前置安全审查"} · 未进入专家分析
        </div>
      );
    }
    return (
      <div
        className="flex items-center gap-2 py-1 text-xs text-muted-foreground"
        role="status"
        aria-label="问诊中"
      >
        <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
        {preflight && !inquiry ? "前置安全审查中" : "问诊中"}
      </div>
    );
  }

  const doneCount = reconcileStages(stages ?? []).length;
  const summaryParts = [
    (notes?.length ?? 0) >= 3 ? `会诊过程 · ${notes!.length} 位专家` : doneCount > 0 ? `会诊过程 · ${doneCount} 步` : "会诊过程",
    typeof elapsedMs === "number" ? `${(elapsedMs / 1000).toFixed(1)}s` : null,
    notes?.length && notes.length < 3 ? `${notes.length} 位专家` : null,
  ].filter(Boolean);

  // 流式中：实时展开步骤
  if (streaming) {
    return (
      <div
        className="mb-2 rounded-md border border-border/60 bg-muted/40 px-3 py-2"
        role="status"
        aria-label="会诊进度"
        aria-live="polite"
      >
        <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-primary">
          <Stethoscope className="h-3.5 w-3.5" aria-hidden />
          多专家会诊中
        </div>
        <StepList stages={stages ?? []} />
      </div>
    );
  }

  // 完成后：折叠摘要条（永久保留，可展开）
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-2">
      <CollapsibleTrigger
        className="flex w-full items-center justify-between rounded-md bg-muted/60 px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-expanded={open}
        aria-controls={contentId}
      >
        <span className="flex items-center gap-1.5">
          <Stethoscope className="h-3.5 w-3.5" aria-hidden />
          {summaryParts.join(" · ")}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent id={contentId}>
        <div className="mt-1.5 space-y-1.5 pl-1">
          {stages?.length && (notes?.length ?? 0) < 3 ? <StepList stages={reconcileStages(stages)} /> : null}
          {notes?.map((n, i) => (
            <NoteCard key={i} note={n} />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
