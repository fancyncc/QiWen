import type { ConsultationNote, MessageItem, Stage, ThreadInfo, UserProfile, ProfileFact, MedicalRecord } from "./types";
import { publicWarnings } from "./consultation-presentation";

/** 适配自有后端的存储字段，页面继续使用现有展示模型。 */
export interface BackendThread {
  id: string;
  mode?: "fixed" | "multi";
  title: string;
  created_at: string;
}
export interface BackendNote {
  expert: string;
  content: string;
  warning?: string;
  status?: string;
  syndrome?: string;
  confidence?: number | null;
  summary?: string;
  public_warnings?: string[];
  prescription_advice?: string;
}
export interface BackendStage {
  stage: string;
  label: string;
  status: string;
  note?: ConsultationNote;
}
export interface BackendMessage {
  role: "user" | "assistant";
  content: string;
  notes?: BackendNote[];
  trace?: { events?: BackendStage[]; elapsedMs?: number } | null;
}
export interface BackendProfile {
  user_id: string;
  constitution: string;
  allergies: string;
  history: string;
  safety_revision?: number;
  facts?: ProfileFact[];
  medical_records?: MedicalRecord[];
}

export const mapThread = (value: BackendThread): ThreadInfo => ({
  thread_id: value.id, title: value.title, created_at: value.created_at,
  mode: value.mode ?? "fixed",
});
export const mapNote = (value: BackendNote): ConsultationNote => ({
  specialist: value.expert,
  reasoning: value.summary || value.content,
  ...(value.summary && /辨证/.test(value.expert) ? { result: value.content } : {}),
  ...(value.status ? { status: value.status } : {}),
  ...(value.syndrome ? { syndrome: value.syndrome } : {}),
  ...(typeof value.confidence === "number" && Number.isFinite(value.confidence) && value.confidence >= 0 && value.confidence <= 1
    ? { confidence: value.confidence } : {}),
  ...(value.prescription_advice ? { prescription_advice: value.prescription_advice } : {}),
  ...((value.public_warnings?.length || value.warning)
    ? { safety_warnings: value.public_warnings?.length ? value.public_warnings : publicWarnings([value.warning!]) } : {}),
});
export function mapStage(value: BackendStage): Stage {
  const aliases: Record<string, string> = {
    formula: "fangji", herb: "bencao", safety: "safety_check", revise_formula: "fangji",
  };
  return {
    stage: aliases[value.stage] ?? value.stage,
    label: value.label,
    status: value.status === "done" ? "done" : "start",
    ...(value.note ? { note: value.note } : {}),
  };
}
export const mapMessage = (value: BackendMessage): MessageItem => ({
  role: value.role,
  content: value.content,
  notes: value.notes?.map(mapNote),
  stages: value.trace?.events?.map(mapStage),
  elapsedMs: value.trace?.elapsedMs,
});
export const mapProfile = (value: BackendProfile): UserProfile => ({
  patient_id: value.user_id,
  constitution: value.constitution,
  allergies: value.allergies,
  past_history: value.history ? [value.history] : [],
  ...(value.safety_revision !== undefined ? { revision: value.safety_revision } : {}),
  ...(value.facts ? { facts: value.facts } : {}),
  medical_records: value.medical_records ?? [],
});

export function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (item.loc?.includes("password")) return "密码长度须为 8 至 128 位";
      if (item.loc?.includes("username")) return "用户名须为 3 至 32 位字母、数字或下划线";
      return typeof item.msg === "string" ? item.msg : fallback;
    }).join("；");
  }
  return fallback;
}
