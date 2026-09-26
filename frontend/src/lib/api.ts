import { useAuthStore } from "@/stores/auth";
import { errorMessage, mapMessage, mapProfile, mapThread } from "./backend-contract";
import type { BackendMessage, BackendProfile, BackendThread } from "./backend-contract";
import type {
  ChatRequest,
  ChatResponse,
  MessageItem,
  ProfileUpdate,
  ThreadInfo,
  TokenResponse,
  UploadResponse,
  UserProfile,
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "") + "/api";

/** 带 HTTP 状态码的错误，便于上层按状态区分处理。 */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** 从 auth store 取 Bearer 头（非交互场景用 getState，避免 hook 限制）。 */
function authHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** 401 时清登录态并跳登录页。 */
function handleUnauthorized() {
  useAuthStore.getState().logout();
  if (typeof window !== "undefined") {
    window.location.href = "/login";
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...options.headers,
    },
  });

  if (res.status === 401 && path !== "/login" && path !== "/register") {
    handleUnauthorized();
    throw new ApiError(401, "未登录或登录已过期");
  }
  if (!res.ok) {
    let msg = `请求失败 (${res.status})`;
    try {
      const d = await res.json();
      msg = errorMessage(d.detail, msg);
    } catch {
      /* 非 JSON 错误体 */
    }
    throw new ApiError(res.status, msg);
  }
  return res.status === 204 ? undefined as T : res.json();
}

export const api = {
  // 认证
  login: (username: string, password: string) =>
    request<TokenResponse>("/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  register: (username: string, password: string) =>
    request<TokenResponse>("/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  // 会话
  listThreads: async () => (await request<BackendThread[]>("/threads")).map(mapThread),
  createThread: async (title: string): Promise<ThreadInfo> => {
    const thread = await request<BackendThread>("/threads", {
      method: "POST", body: JSON.stringify({ mode: "fixed" }),
    });
    if (title && title !== thread.title) {
      return mapThread(await request<BackendThread>(`/threads/${thread.id}`, {
        method: "PUT", body: JSON.stringify({ title }),
      }));
    }
    return mapThread(thread);
  },
  renameThread: (id: string, title: string) =>
    request(`/threads/${id}`, { method: "PUT", body: JSON.stringify({ title }) }),
  deleteThread: (id: string) => request(`/threads/${id}`, { method: "DELETE" }),
  getMessages: async (id: string): Promise<MessageItem[]> =>
    (await request<BackendMessage[]>(`/threads/${id}/messages`)).map(mapMessage),

  // 画像
  getProfile: async (_username: string): Promise<UserProfile> => {
    const user = await request<{ id: string }>("/me");
    return mapProfile(await request<BackendProfile>(`/profiles/${user.id}`));
  },
  updateProfile: async (_username: string, data: ProfileUpdate): Promise<UserProfile> => {
    const user = await request<{ id: string }>("/me");
    return mapProfile(await request<BackendProfile>(`/profiles/${user.id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }));
  },

  // 对话（非流式，流式见 lib/stream.ts）
  chat: (req: ChatRequest) =>
    request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(req) }),

  // 文件上传（multipart，不带 Content-Type 让浏览器自动设置 boundary）
  upload: async (file: File): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: fd,
    });
    if (res.status === 401) {
      handleUnauthorized();
      throw new ApiError(401, "未登录");
    }
    if (!res.ok) {
      let msg = "上传失败";
      try {
        const d = await res.json();
        msg = errorMessage(d.detail, msg);
      } catch {
        /* ignore */
      }
      throw new ApiError(res.status, msg);
    }
    const uploaded: { id: string; name: string } = await res.json();
    return { path: uploaded.id, filename: uploaded.name };
  },
};
