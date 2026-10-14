import type { ConsultationNote, MessageItem } from "./types";

/** Presentation only: the original response, source evidence and reviews stay on the server. */
export function withoutSourceSection(text: string): string {
  return text.replace(/^#{1,4}\s*检索出处[^\n]*\n[\s\S]*?(?=^#{1,4}\s|$(?![\s\S]))/gm, "").trim();
}

export function publicExpertText(text: string): string {
  return withoutSourceSection(text.replace(/<think\b[^>]*>[\s\S]*?<\/think[^>]*>/gi, "").replace(/<think\b[^>]*>[\s\S]*$/gi, ""))
    .replace(/[；;]?\s*知识来源[：:][^\n]*/g, "")
    .replace(/\n{3,}/g, "\n\n").trim();
}

export function publicWarnings(values: string[]): string[] {
  const warnings: string[] = [];
  for (const value of values) {
    // Old records contain mixed clinical warnings and internal validation logs.
    if (/过敏/.test(value)) {
      const allergen = value.match(/过敏原[“「"]([^”」"]+)[”」"]/)?.[1];
      warnings.push(allergen ? `已记录${allergen}过敏，拟用药物的相关风险需医师或药师核对。`
        : /verification_status|unverified|内置演示种子|资料为空|未收录完整/.test(value)
          ? "已记录过敏史，拟用药物的相关风险需医师或药师核对。" : value);
    }
    if (/十八反|十九畏/.test(value) && value.length < 250) warnings.push(value);
    if (/四诊|舌.*脉|脉.*舌/.test(value)) warnings.push("舌脉等资料尚未补全，本次辨证仅为初步判断。");
    if (/verification_status|unverified|内置演示种子|资料为空|信息为空|未收录完整/.test(value)) {
      warnings.push("部分药材资料尚未完成核验，不能据此自行用药。");
    } else if (!/过敏|十八反|十九畏|四诊|舌.*脉|脉.*舌/.test(value)) warnings.push(value);
  }
  return [...new Set(warnings.map(publicExpertText).filter(Boolean))];
}

export function consultationPresentation(message: MessageItem): string {
  if (message.role === "user") return message.content;
  const diagnosis = message.notes?.find((note) => /辨证/.test(note.specialist));
  const formula = message.notes?.find((note) => /方剂/.test(note.specialist));
  if (!diagnosis || !formula) return withoutSourceSection(message.content);
  const result = publicExpertText(diagnosis.result || diagnosis.reasoning || diagnosis.syndrome || "现有资料尚不足以确定证型。");
  const advice = publicExpertText(formula.prescription_advice || "当前尚不能形成可执行的个人处方，请由执业中医师结合面诊资料复核。");
  return `## 辨证结果\n\n${result}\n\n## 处方建议\n\n${advice}`;
}

export function presentNote(note: ConsultationNote): ConsultationNote {
  let summary = note.reasoning ? publicExpertText(note.reasoning) : undefined;
  if (summary && /辨证/.test(note.specialist) && summary.length > 220) {
    const excerpt = summary.slice(0, 220);
    const end = Math.max(excerpt.lastIndexOf("。"), excerpt.lastIndexOf("；"));
    summary = (end >= 100 ? excerpt.slice(0, end + 1) : excerpt + "…") + "\n详细分析见下方辨证结果。";
  }
  if (summary) summary = summary.replace(/\n{2,}/g, "\n");
  if (summary && /本草/.test(note.specialist)) summary = summary.replace(/^\s*[\p{Script=Han}]{1,12}：\s*$/gmu, "").trim();
  return { ...note,
    reasoning: summary,
    safety_warnings: note.safety_warnings ? publicWarnings(note.safety_warnings) : undefined,
  };
}
