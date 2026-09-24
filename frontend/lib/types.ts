export type User = { id: string; username: string };
export type Thread = {
  id: string;
  title: string;
  mode: "fixed" | "multi";
  created_at: string;
  updated_at: string;
};
export type Note = { expert: string; content: string };
export type StageEvent = {
  type: "stage";
  stage: string;
  label: string;
  status: "start" | "done";
};
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  notes?: Note[];
  intent_type?: string | null;
  symptoms?: string[];
  trace?: {
    events: StageEvent[];
    elapsedMs?: number;
    mode: "fixed" | "multi";
  };
};
export type Profile = {
  user_id: string;
  constitution: string;
  allergies: string;
  history: string;
};
export type Upload = { id: string; name: string; media_type: string };
export type AuthResult = { access_token: string; user: User };

