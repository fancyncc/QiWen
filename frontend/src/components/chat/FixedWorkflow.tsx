"use client";

import { useState, type ReactNode } from "react";
import { Check, ChevronDown, CircleAlert, Clock3, GitBranch, Info, Loader2, MessageCircle, ScrollText } from "lucide-react";
import type { Stage, WorkflowPresentation } from "@/lib/types";
import { Fragment } from "react";
import { ClinicalPrescription, ClinicalSafety } from "./ClinicalPrescription";

const mainSteps = [
  { stage: "inquiry", label: "问诊" }, { stage: "search", label: "古籍检索（按需）" },
  { stage: "bianzheng", label: "辨证" }, { stage: "formula", label: "调理方案" },
  { stage: "safety", label: "配伍与用药校验" },
];

const outcomes: Record<string, string> = {
  passed: "配伍规则校验通过", conflict_retrying: "发现冲突，正在重新生成",
  conflict_unresolved: "仍存在冲突，未提供个人处方", not_checked: "未取得完整药材，未执行配伍校验",
  unknown: "校验资料不足，需人工复核",
};
const statuses: Record<string, string> = {
  awaiting_input: "等待补充信息", restricted: "方案待复核", completed: "方案已生成",
  answered: "本轮已回答", blocked: "需先处理安全问题", cancelled: "本轮已停止", failed: "本轮处理失败",
  unavailable: "模型服务暂不可用",
};

export function FixedWorkflowTrace({ stages = [], flow, streaming, elapsedMs, title = "固定工作流" }: {
  stages?: Stage[]; flow?: WorkflowPresentation; streaming: boolean; elapsedMs?: number; title?: string;
}) {
  const [open, setOpen] = useState(false);
  const latestStages = [...new Map(stages.map((stage) => [stage.stage, stage])).values()];
  const steps = flow?.steps ?? mainSteps.map((step) => {
    const actual = latestStages.find((item) => item.stage === step.stage ||
      (step.stage === "formula" && item.stage === "fangji") || (step.stage === "safety" && item.stage === "safety_check"));
    return { ...step, status: actual?.status ?? "not_run", outcome: actual?.outcome };
  });
  const waiting = flow?.status === "awaiting_input";
  const active = latestStages.filter((stage) => stage.status === "start").at(-1);
  const currentLabel: Record<string, string> = { prepare: "准备资料", document_parse: "解析附件", subject_resolve: "确认问诊对象", episode_resolve: "确认本次病程", profile_load: "读取资料", pre_safety: "安全预审", inquiry: "问诊", main_route: "整理问诊信息", evidence_ready: "整理分析资料", rag_decision: "检索决策", search: "古籍检索", evaluate_retrieval: "评估检索结果", rewrite_query: "调整检索", bianzheng: "辨证分析", formula: "生成调理方案", fangji: "生成调理方案", revise_formula: "调整调理方案", herb: "本草核对", bencao: "本草核对", reviewer: "独立审查", safety: "安全校验", safety_check: "安全校验", context_budget: "整理答复", episode_write: "整理问诊记录", finish: "整理答复" };
  const current = streaming ? active ? `${currentLabel[active.stage] ?? active.label}中` : "正在处理"
    : waiting ? `${steps.find((step) => step.status === "waiting")?.label ?? "问诊"} · 等待回答`
      : flow ? statuses[flow.status] : "流程记录";
  return <div className="mb-4 overflow-hidden rounded-xl border border-border bg-background/70" role="region" aria-label={`${title}进度`}>
    <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
      className="flex w-full flex-wrap items-center justify-between gap-2 px-4 py-3 text-left text-xs">
      <span className="flex items-center gap-2 font-medium text-primary"><GitBranch className="h-4 w-4" aria-hidden />{title}
        {typeof elapsedMs === "number" && <span className="font-normal text-muted-foreground">{(elapsedMs / 1000).toFixed(1)}s</span>}
      </span>
      <span className="flex items-center gap-2 text-muted-foreground">
        <span aria-live="polite">{current}</span>
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </span>
    </button>
    {open && <ol className="space-y-2 border-t border-border/70 px-4 py-3">
      {steps.map((step) => {
        const outcome = step.stage === "safety" ? (flow?.compatibility.status ?? step.outcome) : step.outcome;
        const warning = step.status === "failed" || step.status === "cancelled" || step.status === "blocked" || step.status === "unavailable" ||
          (step.status === "done" && outcome && outcome !== "passed");
        const active = streaming && step.status === "start";
        const Icon = active ? Loader2 : warning ? CircleAlert : step.status === "done" ? Check : Clock3;
        const state = (step.status === "done" && outcome ? outcomes[outcome] : undefined) ?? ({
          done: "已完成", start: streaming ? "进行中" : "未完成", waiting: "等待回答", not_run: "尚未执行",
          superseded: "上个候选已执行，本次尚未执行",
          unavailable: "服务异常，尚未完成",
          skipped: "本轮无需检索，已跳过", failed: "执行失败", cancelled: "已停止", blocked: "已拦截",
        }[step.status] ?? "尚未执行");
        return <li key={step.stage} className={`flex flex-wrap items-center justify-between gap-1.5 text-xs ${warning ? "text-destructive" : "text-muted-foreground"}`}>
          <span className="flex items-center gap-2"><Icon className={`h-3.5 w-3.5 shrink-0 ${active ? "animate-spin" : ""}`} aria-hidden />{step.label}</span>
          <span>{state}</span>
        </li>;
      })}
      {flow?.compatibility.status === "passed" && <li className="pt-1 text-[11px] text-muted-foreground">配伍规则通过不代表剂量已核验或适合个人使用。</li>}
      {flow?.compatibility.dosage_status && <li className="text-[11px] text-muted-foreground">剂量核验：{flow.compatibility.dosage_status === "verified" ? "已有核验记录，仍需核对个人适用性" : "尚未完成，需人工复核"}</li>}
    </ol>}
  </div>;
}

export function FollowupQuestion({ body, markdown }: { body: string; markdown: (value: string) => ReactNode }) {
  return <section className="w-full overflow-hidden rounded-xl border border-border bg-card" aria-label="补充问诊">
    <h2 className="flex items-center gap-2.5 border-b border-border/60 bg-secondary/40 px-4 py-3 text-sm font-semibold text-primary sm:px-5">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10"><MessageCircle className="h-4 w-4" aria-hidden /></span>补充问诊
    </h2>
    <div className="px-4 py-4 sm:px-5">
      <div className="break-words text-sm leading-7 [&>p:first-child]:mt-0 [&>p:last-child]:mb-0">{markdown(body)}</div>
      <div className="mt-4 flex items-start gap-2 rounded-lg bg-secondary/50 px-3 py-2.5 text-xs leading-6 text-muted-foreground">
        <Info className="mt-1 h-3.5 w-3.5 shrink-0" aria-hidden />
        <p>请直接在下方补充回答，将结合本次已提供的资料继续分析。暂不清楚的项目可以说明“不清楚”。</p>
      </div>
    </div>
  </section>;
}

export function ServiceUnavailableAnswer({ body, markdown }: { body: string; markdown: (value: string) => ReactNode }) {
  return <section className="rounded-xl border border-destructive/20 bg-destructive/5 p-4 sm:p-5" aria-label="模型服务暂不可用">
    <h2 className="mb-3 flex items-center gap-2 text-base font-semibold text-foreground"><CircleAlert className="h-4 w-4 text-destructive" aria-hidden />模型服务暂不可用</h2>
    <div className="text-sm leading-7">{markdown(body)}</div>
  </section>;
}

export function FixedWorkflowAnswer({ flow, body, markdown }: {
  flow: WorkflowPresentation; body: string; markdown: (value: string) => ReactNode;
}) {
  if (!flow.sections?.length) {
    if (flow.status === "awaiting_input") return <FollowupQuestion body={body} markdown={markdown} />;
    return <section className="rounded-xl border border-border bg-card p-4 sm:p-5" aria-label="本轮答复">
      <h2 className="mb-3 flex items-center gap-2 text-base font-semibold text-primary">
        <MessageCircle className="h-4 w-4" aria-hidden />{flow.intent === "requirements" ? "资料补充说明" : flow.intent === "chat" ? "知识解答" : "本轮答复"}
      </h2>
      <div className="text-sm leading-7">{markdown(body)}</div>
    </section>;
  }
  return <div className="space-y-4" aria-label="固定工作流调理方案">
    <div className="flex flex-wrap items-center justify-between gap-2 px-1">
      <h2 className="text-base font-semibold text-foreground">{flow.legacy ? "历史调理记录" : "本次调理方案"}</h2>
      <span className="rounded-full bg-secondary px-2.5 py-1 text-xs text-primary">{flow.status === "restricted" ? "适用性待复核" : "仅供参考"}</span>
    </div>
    {flow.legacy && <p className="rounded-lg bg-secondary/50 px-4 py-3 text-xs leading-6 text-muted-foreground">这是旧流程保存的答复。记录中的问题不会自动恢复问诊；可在下方补充信息，按当前流程继续分析。</p>}
    {flow.sections.map((section) => section.id === "disclaimer"
      ? <p key={section.id} className="px-1 text-xs leading-6 text-muted-foreground">{section.content}</p>
      : <Fragment key={section.id}><section aria-label={section.title} className="overflow-hidden rounded-xl border border-border bg-card">
        <h3 className="flex items-center gap-2 border-b border-border/60 bg-secondary/40 px-4 py-3 text-sm font-semibold text-primary sm:px-5">
          {section.id === "medication" && <ScrollText className="h-4 w-4" aria-hidden />}{section.title}
        </h3>
        <div className="break-words px-4 py-3 text-sm leading-7 sm:px-5 [&>p]:leading-7">
          {markdown(section.content)}
          {section.id === "diagnosis" && flow.diagnosis_details && <details className="mt-3 border-t border-border/60 pt-3">
            <summary className="cursor-pointer text-xs text-primary">展开详细辨证依据</summary>
            <div className="mt-3 text-muted-foreground">{markdown(flow.diagnosis_details)}</div>
          </details>}
        </div>
      </section>
      {section.id === "medication" && flow.prescription_reference && <ClinicalPrescription reference={flow.prescription_reference} />}
      {section.id === "medication" && <ClinicalSafety warnings={flow.prescription_reference?.safety_warnings || flow.safety_warnings || []} />}
      </Fragment>)}
  </div>;
}
