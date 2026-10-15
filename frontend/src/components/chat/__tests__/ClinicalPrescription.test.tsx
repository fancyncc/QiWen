import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageBubble } from "../MessageBubble";
import { consultationPresentation } from "@/lib/consultation-presentation";
import { mapMessage } from "@/lib/backend-contract";
import type { PrescriptionReference } from "@/lib/types";

const reference: PrescriptionReference = {
  name: "来源明确的参考方", source: "方剂文献", rationale: "已确认症状与选方条件的对应。", approved: false,
  dose_notice: "单味药文献参考范围不是本次个人拟用量，具体每剂与每日口径待确认。", uncertainties: ["脉象未知，不能据此确诊。"],
  safety_warnings: ["过敏风险：已记录青霉素过敏，相关风险待医师核对。", "茯苓条目禁忌：相关人群慎用。"],
  herbs: [{ herb_id: "h1", name: "茯苓", preparation: "", verification_status: "unverified", dosage: { min: 999, max: 999, unit: "g" },
    reference_dose: { min: 10, max: 15, unit: "g", kind: "literature_range", basis: "单味饮片文献范围，口径待医师核定", preparation: "茯苓饮片",
      source: "药典条目", url: "https://shennongalpha.westlake.edu.cn/knowledge/nmm-01ei", quote: "10～15g。", checked_at: "2026-10-15" },
    effect: "健脾宁心。" },
    { herb_id: "h2", name: "待核对药物", preparation: "", verification_status: "unverified", dosage: { min: null, max: null, unit: "g" } }],
};

describe("独立处方与临床内容", () => {
  it.each(["multi", "fixed"] as const)("%s 显示来源范围、单独的拟用量和安全说明，刷新与复制保留同一数据", (mode) => {
    const data = mapMessage({ role: "assistant", intent_type: "diagnose", content: "程序内部校验详情不属于临床结果。",
      flow: { mode, intent: "diagnose", status: "restricted", steps: [], compatibility: { status: "passed" },
        prescription_reference: reference,
        ...(mode === "fixed" ? { sections: [{ id: "diagnosis", title: "辨证结论", content: "具体的临床依据与病机解释。" },
          { id: "medication", title: "选方与处方建议", content: "说明治法和选方理由。" }] } : {}) },
      notes: [{ expert: "辨证", content: "具体的临床依据与病机解释。", summary: "临床摘要。" },
        { expert: "方剂", content: "说明治法和选方理由。", prescription_advice: "说明治法和选方理由。" },
        { expert: "本草", content: "针对本方说明配伍与药物风险。" }],
    });
    render(<MessageBubble message={data} />);
    expect(screen.getByText("具体的临床依据与病机解释。")).toBeInTheDocument();
    const card = screen.getByRole("region", { name: "独立处方参考" });
    expect(within(card).getByRole("table", { name: "处方组成、参考范围与本次拟用量" })).toBeInTheDocument();
    expect(within(card).getByRole("columnheader", { name: "本次拟用量" })).toBeInTheDocument();
    expect(within(card).getByText("10–15 g")).toBeInTheDocument();
    expect(within(card).getByRole("link", { name: "剂量来源" })).toHaveAttribute("href", reference.herbs[0].reference_dose!.url);
    expect(screen.queryByText(/999/)).not.toBeInTheDocument();
    expect(within(card).getByText("暂无可核对记录")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "安全说明" })).toHaveTextContent("青霉素过敏");
    expect(screen.queryByText("程序内部校验详情不属于临床结果。")).not.toBeInTheDocument();
    const copied = consultationPresentation(data);
    expect(copied).toContain("## 处方参考（待医师复核）");
    expect(copied).toContain("10–15 g");
    expect(copied).toContain(reference.herbs[0].reference_dose!.url);
    expect(copied).toContain("茯苓饮片");
    expect(copied).toContain("脉象未知，不能据此确诊。");
    expect(copied).not.toContain("999");
    if (mode === "multi") {
      fireEvent.click(screen.getByRole("button", { name: /会诊过程 · 3 位专家/ }));
      expect(screen.getByText("针对本方说明配伍与药物风险。")).toBeInTheDocument();
    }
  });
});
