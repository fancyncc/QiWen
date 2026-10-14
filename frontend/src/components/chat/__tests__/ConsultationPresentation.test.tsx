import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "../MessageBubble";
import { mapMessage, mapNote } from "@/lib/backend-contract";
import { consultationPresentation, withoutSourceSection, presentNote } from "@/lib/consultation-presentation";

const message = () => mapMessage({ role: "assistant",
  content: "## 初步辨证\n旧内容\n## 检索出处\n古籍原文\n## 安全校验\nverification_status 为 unverified\n## 仍待补充\n检索工具没有返回\n",
  notes: [
    { expert: "辨证", content: "详细分析", syndrome: "证候方向待面诊确认", confidence: 0.72, summary: "根据已提供的症状作初步分析，舌脉资料仍有缺项。" },
    { expert: "方剂", content: "可核对的方剂知识条目。", summary: "比较已检索条目与症状的重合及缺失依据。", prescription_advice: "当前尚不能形成个人处方，需由执业中医师复核。" },
    { expert: "本草", content: "药材功效与禁忌的知识核对。", public_warnings: ["已记录青霉素过敏，需要核对拟用药物风险。"] },
  ],
});

describe("会诊结果展示", () => {
  it("直接展开三个专家摘要，保留置信度与本草警告，正文只显示两个结果部分", () => {
    const data = message();
    render(<MessageBubble message={data} />);
    expect(screen.getByText(/会诊过程 · 3 位专家/)).toBeInTheDocument();
    for (const title of ["辨证", "方剂", "本草"]) expect(screen.getByText(title, { exact: true })).toBeInTheDocument();
    expect(screen.getByText("置信度 72%")).toBeInTheDocument();
    expect(screen.getByRole("note", { name: "本草安全警告" })).toHaveTextContent("青霉素过敏");
    expect(screen.getByRole("heading", { name: "辨证结果" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "处方建议" })).toBeInTheDocument();
    for (const hidden of ["检索出处", "古籍原文", "安全校验", "仍待补充", "verification_status 为 unverified"]) {
      expect(screen.queryByText(hidden)).not.toBeInTheDocument();
    }
    expect(data.content).toContain("verification_status"); // Audit source is unchanged.
  });
  it("旧记录没有置信度时不显示置信度，不编造百分比", () => {
    const data = message();
    delete data.notes![0].confidence;
    render(<MessageBubble message={data} />);
    expect(screen.queryByText(/置信度/)).not.toBeInTheDocument();
    expect(screen.queryByText(/95%/)).not.toBeInTheDocument();
  });
  it("旧记录的长分析在专家卡中压缩，完整结果保留，空白本草条目不占行", () => {
    const full = "来自用户描述的分析依据。".repeat(50);
    const note = { specialist: "辨证", reasoning: full };
    expect(presentNote(note).reasoning!.length).toBeLessThan(360);
    expect(consultationPresentation({ role: "assistant", content: "原答复", notes: [note, { specialist: "方剂" }] })).toContain(full);
    expect(presentNote({ specialist: "本草", reasoning: "桂枝：功效说明。\n芍药：\n生姜：" }).reasoning).not.toContain("芍药");
  });
  it("纯安全拦截仍显示重要警告，移除出处时保留后续正文", () => {
    const data = { role: "assistant" as const, content: "## 安全警告\n正在出现危险症状时请及时就医。\n## 检索出处\n原文\n## 提醒\n保留提醒" };
    expect(consultationPresentation(data)).toContain("危险症状");
    expect(withoutSourceSection(data.content)).toContain("保留提醒");
    expect(withoutSourceSection(data.content)).not.toContain("原文");
  });
  it("历史技术日志整理为临床警告，历史和流式共用的适配器保留新增字段", () => {
    const legacy = mapNote({ expert: "本草", content: "条目说明。", warning: 'verification_status 为 unverified；过敏原“青霉素”尚未覆盖自动校验；四诊资料不完整' });
    expect(legacy.safety_warnings?.join(" ")).toContain("青霉素过敏");
    expect(legacy.safety_warnings?.join(" ")).not.toContain("verification_status");
    expect(mapNote({ expert: "辨证", content: "说明", confidence: 95 }).confidence).toBeUndefined();
    expect(message().notes?.[1].prescription_advice).toContain("不能形成个人处方");
  });
});
