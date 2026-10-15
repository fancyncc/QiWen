import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageBubble } from "../MessageBubble";
import { FixedWorkflowTrace } from "../FixedWorkflow";
import { mapMessage } from "@/lib/backend-contract";
import type { WorkflowPresentation } from "@/lib/types";

const flow = (changes: Partial<WorkflowPresentation> = {}): WorkflowPresentation => ({
  mode: "fixed", intent: "consult", status: "awaiting_input", compatibility: { status: "not_checked" },
  steps: [
    { stage: "inquiry", label: "问诊", status: "waiting" },
    { stage: "search", label: "古籍检索（按需）", status: "not_run" },
    { stage: "bianzheng", label: "辨证", status: "not_run" },
    { stage: "formula", label: "调理方案", status: "not_run" },
    { stage: "safety", label: "配伍与用药校验", status: "not_run" },
  ], ...changes,
});

describe("固定工作流展示", () => {
  it("运行时保持收起，当前阶段随后端事件更新，也可手动展开和收起", () => {
    const { rerender } = render(<FixedWorkflowTrace streaming stages={[{ stage: "bianzheng", label: "辨证", status: "start" }]} />);
    expect(screen.getByRole("button", { name: /辨证分析中/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("尚未执行")).not.toBeInTheDocument();
    rerender(<FixedWorkflowTrace streaming stages={[
      { stage: "bianzheng", label: "辨证", status: "start" },
      { stage: "bianzheng", label: "辨证", status: "done" },
      { stage: "formula", label: "调理方案", status: "start" },
    ]} />);
    const progress = screen.getByRole("button", { name: /生成调理方案中/ });
    expect(progress).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(progress);
    expect(progress).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("进行中")).toBeInTheDocument();
    fireEvent.click(progress);
    expect(screen.queryByText("进行中")).not.toBeInTheDocument();
  });

  it.each(["fixed", "multi"] as const)("%s 的上游故障显示服务异常，不能伪装成等待病情资料", (mode) => {
    const data = flow({ mode, intent: "safety_rejected", status: "unavailable", steps: [
      { stage: "pre_safety", label: "前置安全检查", status: "unavailable" },
      { stage: "bianzheng", label: "辨证", status: "not_run" },
    ] });
    render(<MessageBubble message={{ role: "assistant", content: "安全预审暂时未能完成：模型上游服务异常（HTTP 500）。", flow: data }} />);
    expect(screen.getByRole("heading", { name: "模型服务暂不可用" })).toBeInTheDocument();
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "补充问诊" })).not.toBeInTheDocument();
    expect(screen.queryByText("等待补充信息")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /模型服务暂不可用/ }));
    expect(screen.getByText("服务异常，尚未完成")).toBeInTheDocument();
    expect(screen.getByText("尚未执行")).toBeInTheDocument();
    expect(screen.queryByText("已拦截")).not.toBeInTheDocument();
  });

  it("两种模式共用资料补充说明，查询不会显示处方或专家分析已完成", () => {
    render(<MessageBubble message={{ role: "assistant", content: "已记录的内容无需重复；本轮仅查询资料。",
      flow: flow({ mode: "multi", intent: "requirements", status: "answered", steps: [] }),
      notes: [{ specialist: "辨证", result: "旧结果" }, { specialist: "方剂", prescription_advice: "旧候选" }] }} />);
    expect(screen.getByRole("heading", { name: "资料补充说明" })).toBeInTheDocument();
    expect(screen.getByText(/本轮仅查询已有资料/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "处方建议" })).not.toBeInTheDocument();
    expect(screen.queryByText("旧候选")).not.toBeInTheDocument();
  });

  it.each(["bianzheng", "formula", "herb"])("多 Agent 在 %s 追问时展示等待状态，不显示尚未形成的处方", (stage) => {
    const steps = ["bianzheng", "formula", "herb", "reviewer", "safety"];
    const index = steps.indexOf(stage);
    const data = flow({ mode: "multi", steps: steps.map((step, position) => ({
      stage: step, label: ({ bianzheng: "辨证", formula: "方剂", herb: "本草", reviewer: "独立审查", safety: "安全校验" } as Record<string, string>)[step],
      status: position < index ? "done" : position === index ? "waiting" : "not_run",
    })) });
    render(<MessageBubble message={{ role: "assistant", content: "请补充本次发病前有无明显诱因？", flow: data,
      notes: [{ specialist: "辨证", result: "尚未形成最终分析" }, { specialist: "方剂", prescription_advice: "上一候选的方案" }] }} />);
    expect(screen.getByRole("region", { name: "多 Agent 会诊进度" })).toBeInTheDocument();
    const progress = screen.getByRole("button", { name: /多 Agent 会诊.*等待回答/ });
    expect(progress).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("尚未执行")).not.toBeInTheDocument();
    fireEvent.click(progress);
    expect(screen.getByText("等待回答")).toBeInTheDocument();
    expect(screen.getAllByText("尚未执行")).toHaveLength(steps.length - index - 1);
    expect(screen.getByRole("heading", { name: "补充问诊" })).toBeInTheDocument();
    expect(screen.getByText("请补充本次发病前有无明显诱因？")).toBeInTheDocument();
    expect(screen.queryByText("上一候选的方案")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "处方建议" })).not.toBeInTheDocument();
  });

  it("后端 consult 显示补充问诊，正文提到辨证也不能打勾或展示方案", () => {
    render(<MessageBubble message={mapMessage({ role: "assistant", intent_type: "consult", content: "辨证完成后还需要什么？请回答测试问题。",
      trace: { mode: "fixed", presentation: flow(), events: [{ stage: "inquiry", label: "问诊", status: "done" }] } })} />);
    expect(screen.getByRole("heading", { name: "补充问诊" })).toBeInTheDocument();
    expect(screen.getByText("问诊 · 等待回答")).toBeInTheDocument();
    expect(screen.getByText(/请直接在下方补充回答/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /固定工作流/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("尚未执行")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /固定工作流/ }));
    expect(screen.getByText("等待回答")).toBeInTheDocument();
    expect(screen.getAllByText("尚未执行")).toHaveLength(4);
    expect(screen.queryByText("本次调理方案")).not.toBeInTheDocument();
    expect(screen.queryByText("多专家会诊中")).not.toBeInTheDocument();
  });

  it("正式方案分段显示，详细辨证依据可折叠，历史独立审查如实保留", () => {
    const data = flow({ intent: "diagnose", status: "restricted", legacy: true,
      steps: [{ stage: "reviewer", label: "独立审查（历史流程）", status: "done" }],
      compatibility: { status: "unknown" }, diagnosis_details: "完整的测试依据。",
      sections: [
        { id: "diagnosis", title: "辨证分析", content: "简短分析。" },
        { id: "medication", title: "中药调理参考", content: "个人适用性待复核。" },
        { id: "non_drug", title: "非药物建议", content: "日常建议。" },
        { id: "medical_guidance", title: "就医指引", content: "面诊指引。" },
        { id: "disclaimer", title: "参考声明", content: "参考声明正文。" },
      ] });
    render(<MessageBubble message={{ role: "assistant", content: "旧的长文及内部日志", flow: data }} />);
    for (const name of ["辨证分析", "中药调理参考", "非药物建议", "就医指引"]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
    expect(screen.getByText("展开详细辨证依据").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByRole("heading", { name: "历史调理记录" })).toBeInTheDocument();
    expect(screen.getByText(/这是旧流程保存的答复/)).toBeInTheDocument();
    expect(screen.queryByText("旧的长文及内部日志")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /固定工作流/ }));
    expect(screen.getByText("独立审查（历史流程）")).toBeInTheDocument();
  });

  it.each([
    ["passed", "配伍规则校验通过"], ["conflict_retrying", "发现冲突，正在重新生成"],
    ["conflict_unresolved", "仍存在冲突，未提供个人处方"], ["not_checked", "未取得完整药材，未执行配伍校验"],
  ])("按实际校验结果区分 %s", (status, label) => {
    render(<FixedWorkflowTrace flow={flow({ status: "restricted", steps: [{ stage: "safety", label: "配伍与用药校验", status: "done" }],
      compatibility: { status } })} streaming={false} />);
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByText(label)).toBeInTheDocument();
    if (status !== "passed") expect(screen.queryByText("配伍规则校验通过")).not.toBeInTheDocument();
  });
});
