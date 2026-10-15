/**
 * 页面展示类型。自有后端的字段通过 backend-contract.ts 转换。
 */

export interface ChatRequest {
  message: string;
  mode?: "fixed" | "multi";
  thread_id?: string;
  files?: string[];
}

export interface ChatResponse {
  response: string;
  thread_id: string;
  intent_type: "chat" | "consult" | "diagnose" | "safety_rejected" | string;
  symptoms: string[];
}

export interface ThreadInfo {
  thread_id: string;
  mode?: "fixed" | "multi";
  title: string;
  created_at: string;
}

export interface MessageItem {
  role: "user" | "assistant";
  content: string;
  intent_type?: string;
  mode?: "fixed" | "multi";
  flow?: WorkflowPresentation;
  notes?: ConsultationNote[];
  /** 会诊阶段轨迹（per-message 持久，流式中实时更新，完成后保留为摘要条） */
  stages?: Stage[];
  /** 单轮响应用时（毫秒，来自 done 事件） */
  elapsedMs?: number;
}

export interface WorkflowPresentation {
  mode: "fixed" | "multi";
  intent: string;
  status: "awaiting_input" | "blocked" | "answered" | "restricted" | "completed" | "cancelled" | "failed" | "unavailable";
  steps: { stage: string; label: string; status: string; outcome?: string }[];
  compatibility: { status: string; personal_plan_approved?: boolean; dosage_status?: string };
  sections?: { id: string; title: string; content: string }[];
  diagnosis_details?: string;
  legacy?: boolean;
  prescription_reference?: PrescriptionReference;
  safety_warnings?: string[];
}

export interface PrescriptionReference {
  name: string;
  source: string;
  rationale: string;
  indication?: string;
  approved: false;
  dose_notice?: string;
  uncertainties: string[];
  safety_warnings: string[];
  herbs: {
    herb_id: string; name: string; preparation: string; effect?: string; cautions?: string;
    verification_status: string;
    dosage: { min: number | null; max: number | null; unit: string };
    reference_dose?: {
      min: number; max: number; unit: string; kind: "literature_range";
      basis: string; preparation: string; source: string; url: string; quote: string; checked_at: string;
    } | null;
  }[];
}

export interface UserProfile {
  patient_id: string;
  constitution: string;
  allergies: string;
  past_history: string[];
  revision?: number;
  facts?: ProfileFact[];
  medical_records?: MedicalRecord[];
}

export interface MedicalRecord {
  id: string;
  record_date: string | null;
  institution: string;
  diagnosis: string;
  treatment: string;
  outcome: string;
  created_at: string;
  updated_at: string;
}
export type MedicalRecordInput = Pick<MedicalRecord, "record_date" | "institution" | "diagnosis" | "treatment" | "outcome"> & {
  revision: number;
  confirm_self: boolean;
};

export interface ProfileFact {
  id: string;
  category: string;
  entity: string;
  subject: string;
  assertion: "present" | "absent" | "uncertain";
  temporal_scope: string;
  details: string;
  source: { quote: string; thread_id?: string; collected_at: string };
}
export interface ProfileSuggestion {
  id: string;
  fact: ProfileFact;
  conflict: boolean;
  previous: string;
}
export interface SuggestionDecision {
  id: string;
  action: "accept" | "dismiss" | "other";
  confirm_self?: boolean;
  replace?: boolean;
  entity?: string;
  details?: string;
}

export interface ProfileUpdate {
  constitution?: string;
  allergies?: string;
  history?: string;
  revision?: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UploadResponse {
  path: string;
  filename: string;
}

/** 会诊笔记条目（多智能体专家输出，对应后端 consultation_notes）。 */
export interface ConsultationNote {
  specialist: string;
  uncertainties?: string[];
  status?: string;
  prescription_advice?: string;
  result?: string;
  syndrome?: string;
  recommended_formulas?: string[];
  herb_notes?: unknown[];
  safety_warnings?: string[];
  reasoning?: string;
  confidence?: number;
  modifications?: string;
  approved?: boolean;
  issues?: string[];
  suggestions?: string;
  [k: string]: unknown;
}

/** 会诊阶段（用于时间线展示）。note=该阶段完成时的会诊笔记（渐进揭示）。 */
export interface Stage {
  stage: string;
  label: string;
  status: "start" | "done" | "failed" | "cancelled";
  outcome?: string;
  note?: ConsultationNote;
}

/** SSE 流式事件（POST /api/chat?stream=true） */
export type ChatStreamEvent =
  | { type: "token"; content: string }
  | {
      type: "stage";
      stage: string;
      label: string;
      status: Stage["status"];
      outcome?: string;
      note?: ConsultationNote;
    }
  | {
      type: "done";
      thread_id: string;
      notes?: ConsultationNote[];
      diagnosis?: string;
      elapsed_ms?: number;
      flow?: WorkflowPresentation;
    }
  | { type: "error"; message: string };
