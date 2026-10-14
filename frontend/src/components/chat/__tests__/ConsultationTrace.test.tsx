import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConsultationTrace } from "@/components/chat/ConsultationTrace";
import type { ConsultationNote, Stage } from "@/lib/types";

const SPECIALIST_STAGES: Stage[] = [
  { stage: "inquiry", label: "问诊", status: "done" },
  { stage: "bianzheng", label: "辨证", status: "done" },
  { stage: "fangji", label: "方剂", status: "start" },
];

const NOTES: ConsultationNote[] = [
  { specialist: "辨证专家", syndrome: "阴虚火旺", confidence: 0.9 },
];

describe("ConsultationTrace", () => {
  it("无阶段且无笔记时不渲染", () => {
    const { container } = render(<ConsultationTrace streaming={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("追问轮（仅问诊阶段）：流式中显示单行「问诊中」，完成后不渲染", () => {
    const stages: Stage[] = [{ stage: "inquiry", label: "问诊", status: "start" }];
    const { container, rerender } = render(
      <ConsultationTrace stages={stages} streaming={true} />
    );
    expect(screen.getByText("问诊中")).toBeInTheDocument();
    rerender(<ConsultationTrace stages={stages} streaming={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("会诊轮流式中：只展示实际启动的步骤", () => {
    const { container } = render(
      <ConsultationTrace stages={SPECIALIST_STAGES} streaming={true} />
    );
    expect(screen.getByText("多专家会诊中")).toBeInTheDocument();
    expect(screen.getByText("依次会诊")).toBeInTheDocument();
    for (const label of ["问诊", "辨证", "方剂"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.queryByText("本草")).not.toBeInTheDocument();
    // 存在 active 阶段 -> 有旋转图标
    expect(container.querySelectorAll("svg.animate-spin").length).toBeGreaterThan(0);
  });

  it("会诊轮完成后：折叠摘要条（步数/用时/专家数），默认不显示笔记详情", () => {
    const { container } = render(
      <ConsultationTrace
        stages={SPECIALIST_STAGES}
        notes={NOTES}
        streaming={false}
        elapsedMs={8200}
      />
    );
    expect(screen.getByText(/会诊过程 · 3 步 · 8.2s · 1 位专家/)).toBeInTheDocument();
    // 笔记详情默认折叠
    expect(screen.queryByText("阴虚火旺")).not.toBeInTheDocument();
    // An incomplete stage stays incomplete even when streaming has stopped.
    expect(container.querySelector("svg.animate-spin")).toBeNull();
  });

  it("点击摘要条展开：显示步骤与会诊笔记", () => {
    render(
      <ConsultationTrace
        stages={SPECIALIST_STAGES}
        notes={NOTES}
        streaming={false}
        elapsedMs={8200}
      />
    );
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByText(/阴虚火旺/)).toBeInTheDocument();
  });

  it("专家阶段完成即显示单行结论卡（渐进揭示）", () => {
    const stagesWithNote: Stage[] = [
      { stage: "inquiry", label: "问诊", status: "done" },
      {
        stage: "bianzheng",
        label: "辨证",
        status: "done",
        note: { specialist: "辨证", syndrome: "脾胃虚寒证", confidence: 0.85 },
      },
      { stage: "fangji", label: "方剂", status: "start" },
    ];
    render(<ConsultationTrace stages={stagesWithNote} streaming={true} />);
    expect(screen.getByText(/证候：脾胃虚寒证 · 置信度 85%/)).toBeInTheDocument();
    // 进行中的方剂无结论卡
    expect(screen.queryByText(/推荐方/)).not.toBeInTheDocument();
  });

  it("结论卡安全警告用警示色", () => {
    const stagesWithWarning: Stage[] = [
      {
        stage: "bencao",
        label: "本草",
        status: "done",
        note: { specialist: "本草", safety_warnings: ["甘草反海藻"] },
      },
    ];
    const { container } = render(
      <ConsultationTrace stages={stagesWithWarning} streaming={true} />
    );
    expect(screen.getByText(/安全警告：甘草反海藻/)).toBeInTheDocument();
    expect(container.querySelector(".text-destructive")).not.toBeNull();
  });

  it("仅收到审查事件时，不虚构辨证、方剂、本草已经执行", () => {
    const partial: Stage[] = [
      { stage: "reviewer", label: "安全审查", status: "done" },
      { stage: "safety_check", label: "安全校验", status: "done" },
    ];
    render(<ConsultationTrace stages={partial} notes={NOTES} streaming={false} />);
    expect(screen.getByText(/会诊过程 · 2 步/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByText("安全审查")).toBeInTheDocument();
    expect(screen.getByText("安全校验")).toBeInTheDocument();
    expect(screen.queryByText("方剂")).not.toBeInTheDocument();
    expect(screen.queryByText("本草")).not.toBeInTheDocument();
  });

  it("仅笔记无阶段（历史会话）时显示摘要条", () => {
    render(<ConsultationTrace notes={NOTES} streaming={false} />);
    expect(screen.getByText(/会诊过程/)).toBeInTheDocument();
  });

  it("安全预审结束却未进入专家时保留流程位置", () => {
    render(<ConsultationTrace stages={[{ stage: "pre_safety", label: "前置安全审查", status: "done" }]} />);
    expect(screen.getByText("流程进度：前置安全审查 · 未进入专家分析")).toBeInTheDocument();
    expect(screen.queryByText("辨证")).not.toBeInTheDocument();
  });
});
