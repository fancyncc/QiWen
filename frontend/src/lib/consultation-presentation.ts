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

export function consultationSections(message: MessageItem): { diagnosis: string; advice: string; herbs?: string } | null {
  if (message.role === "user" || message.flow?.status === "awaiting_input" || message.flow?.status === "unavailable"
    || message.flow?.intent === "requirements" || message.intent_type === "consult") return null;
  const diagnosis = message.notes?.find((note) => /辨证/.test(note.specialist));
  const formula = message.notes?.find((note) => /方剂/.test(note.specialist));
  const herb = message.notes?.find((note) => /本草/.test(note.specialist));
  if (!diagnosis || !formula) return null;
  let result = publicExpertText(diagnosis.result || diagnosis.reasoning || diagnosis.syndrome || "现有资料尚不足以确定证型。");
  const uncertainties = diagnosis.uncertainties?.map(publicExpertText).filter(Boolean);
  if (uncertainties?.length) result += `\n\n### 判断局限\n\n${uncertainties.map((item) => `- ${item}`).join("\n")}`;
  const advice = publicExpertText(formula.prescription_advice || "当前尚不能形成可执行的个人处方，请由执业中医师结合面诊资料复核。");
  const herbs = herb ? publicExpertText(herb.reasoning || herb.result || "") : undefined;
  return { diagnosis: result, advice, herbs };
}

export function consultationPresentation(message: MessageItem): string {
  if (message.role === "user") return message.content;
  if (message.flow?.mode === "fixed" && message.flow.sections?.length) {
    return message.flow.sections.map((section) => `## ${section.title}\n\n${section.content}${section.id === "medication" ? prescriptionText(message) : ""}`).join("\n\n");
  }
  const sections = consultationSections(message);
  return sections ? `## 辨证结果\n\n${sections.diagnosis}\n\n## 处方建议\n\n${sections.advice}${prescriptionText(message)}${sections.herbs ? `\n\n## 本草核对\n\n${sections.herbs}` : ""}`
    : withoutSourceSection(message.content);
}

function prescriptionText(message: MessageItem): string {
  const reference = message.flow?.prescription_reference;
  const warnings = reference?.safety_warnings || message.flow?.safety_warnings || [];
  const lines: string[] = [];
  if (reference) {
    lines.push("## 处方参考（待医师复核）", "", reference.name, "", reference.dose_notice || "个人适用性与剂量待医师确认。", "",
      "| 药物 | 文献参考范围 | 本次拟用量 | 来源 |", "| --- | --- | --- | --- |");
    for (const item of reference.herbs) {
      const range = item.reference_dose;
      const literature = range ? `${range.min}–${range.max} ${range.unit}；${range.basis}` : "暂无可核对记录";
      const planned = item.verification_status === "verified" && item.dosage.min !== null
        ? `${item.dosage.min}–${item.dosage.max} ${item.dosage.unit}（口径与个人适用性待确认）` : "待医师核定";
      const preparation = item.preparation || range?.preparation;
      lines.push(`| ${item.name}${preparation ? `（来源炮制／用法要求：${preparation}）` : ""} | ${literature} | ${planned} | ${range ? `[${range.source}](${range.url})` : "待补充"} |`);
    }
    lines.push("", `方剂出处：${reference.source}`);
    if (reference.uncertainties.length) lines.push("", "### 尚待确认的适用条件", "", ...reference.uncertainties.map((item) => `- ${item}`));
  }
  if (warnings.length) lines.push("", "## 安全说明", "", ...publicWarnings(warnings).map((item) => `- ${item}`));
  return lines.length ? "\n\n" + lines.join("\n") : "";
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
