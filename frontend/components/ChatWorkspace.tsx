"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { API_BASE, ApiError, api } from "@/lib/api";
import type { Message, Profile, StageEvent, Thread, Upload } from "@/lib/types";
import { useAuth } from "@/stores/auth";
import LeafIcon from "./LeafIcon";
import MessageBubble from "./MessageBubble";

const EXAMPLES = [
  "最近总是睡不好，白天容易疲劳",
  "我想了解中医如何整理咳嗽症状",
  "介绍固定流程与多专家会诊的区别",
];
const DISCLAIMER = "仅供学习与研究参考，不能代替执业医师面诊，也不应作为自行抓药的依据。";

function visibleError(error: unknown) {
  return error instanceof Error ? error.message : "操作失败，请稍后重试";
}

function mergeAssistantMessages(messages: Message[]) {
  const merged: Message[] = [];
  for (const message of messages) {
    const previous = merged[merged.length - 1];
    if (message.role === "assistant" && previous?.role === "assistant") {
      merged[merged.length - 1] = {
        ...message,
        content: [previous.content, message.content].filter(Boolean).join("\n\n"),
        notes: message.notes?.length ? message.notes : previous.notes,
        trace: message.trace || previous.trace,
      };
    } else merged.push(message);
  }
  return merged;
}

function DialogFrame({ title, children, onClose }: { title: string; children: React.ReactNode; onClose: () => void }) {
  useEffect(() => {
    const listener = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [onClose]);
  return <div className="dialog-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="dialog-card" role="dialog" aria-modal="true" aria-label={title}>
      <div className="dialog-heading"><h2>{title}</h2><button type="button" aria-label="关闭弹窗" onClick={onClose}>×</button></div>
      {children}
    </section>
  </div>;
}

export default function ChatWorkspace() {
  const pathname = usePathname();
  const routeThreadId = pathname?.startsWith("/chat/") ? decodeURIComponent(pathname.slice(6).split("/")[0]) : null;
  const router = useRouter();
  const queryClient = useQueryClient();
  const { token, user, hydrated, logout } = useAuth();
  const [mode, setMode] = useState<"fixed" | "multi">("fixed");
  const [draft, setDraft] = useState("");
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [liveMessages, setLiveMessages] = useState<Message[] | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [dialog, setDialog] = useState<{ kind: "rename" | "delete"; thread: Thread } | null>(null);
  const [renameText, setRenameText] = useState("");
  const [showProfile, setShowProfile] = useState(false);
  const [profileDraft, setProfileDraft] = useState({ constitution: "", allergies: "", history: "" });
  const controller = useRef<AbortController | null>(null);
  const activeRun = useRef(0);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const threadQuery = useQuery({
    queryKey: ["threads", user?.id],
    queryFn: () => api<Thread[]>("/api/threads", token),
    enabled: hydrated && !!token && !!user,
  });
  const profileQuery = useQuery({
    queryKey: ["profile", user?.id],
    queryFn: () => api<Profile>(`/api/profiles/${user!.id}`, token),
    enabled: hydrated && !!token && !!user,
  });
  const messagesQuery = useQuery({
    queryKey: ["messages", routeThreadId],
    queryFn: () => api<Message[]>(`/api/threads/${routeThreadId}/messages`, token),
    enabled: hydrated && !!token && !!routeThreadId,
  });
  const threads = threadQuery.data || [];
  const currentThread = threads.find(item => item.id === routeThreadId);
  const messages = useMemo(() => mergeAssistantMessages(liveMessages || messagesQuery.data || []), [liveMessages, messagesQuery.data]);

  useEffect(() => {
    if (hydrated && (!token || !user)) router.replace("/login");
  }, [hydrated, token, user, router]);
  useEffect(() => {
    const errors = [threadQuery.error, profileQuery.error, messagesQuery.error];
    if (errors.some(error => error instanceof ApiError && error.status === 401)) {
      logout();
      queryClient.clear();
      toast.error("登录已失效，请重新登录");
      router.replace("/login");
    }
  }, [threadQuery.error, profileQuery.error, messagesQuery.error, logout, queryClient, router]);
  useEffect(() => {
    if (currentThread) setMode(currentThread.mode);
  }, [currentThread?.id, currentThread?.mode]);
  useEffect(() => {
    setLiveMessages(null);
    setUploads([]);
    setMobileOpen(false);
  }, [routeThreadId]);
  useEffect(() => {
    const input = inputRef.current;
    if (input) {
      input.style.height = "auto";
      input.style.height = Math.min(Math.max(input.scrollHeight, 44), 180) + "px";
    }
  }, [draft]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: sending ? "smooth" : "instant" }); }, [messages, sending]);
  useEffect(() => () => { activeRun.current++; controller.current?.abort(); }, []);

  function newChat() {
    if (sending) return;
    setLiveMessages(null);
    setDraft("");
    setUploads([]);
    setMobileOpen(false);
    router.push("/chat");
  }

  function openThread(thread: Thread) {
    if (sending) return;
    setMobileOpen(false);
    router.push(`/chat/${thread.id}`);
  }

  function openProfile() {
    const current = profileQuery.data;
    setProfileDraft({
      constitution: current?.constitution || "",
      allergies: current?.allergies || "",
      history: current?.history || "",
    });
    setMobileOpen(false);
    setShowProfile(true);
  }

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token || !user) return;
    try {
      await api(`/api/profiles/${user.id}`, token, { method: "PATCH", body: JSON.stringify(profileDraft) });
      await queryClient.invalidateQueries({ queryKey: ["profile", user.id] });
      toast.success("画像已保存");
      setShowProfile(false);
    } catch (error) { toast.error(visibleError(error)); }
  }

  async function renameThread(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dialog || dialog.kind !== "rename" || !token || !renameText.trim()) return;
    try {
      await api(`/api/threads/${dialog.thread.id}`, token, {
        method: "PUT", body: JSON.stringify({ title: renameText.trim() }),
      });
      await queryClient.invalidateQueries({ queryKey: ["threads", user?.id] });
      setDialog(null);
      toast.success("会话已重命名");
    } catch (error) { toast.error(visibleError(error)); }
  }

  async function deleteThread() {
    if (!dialog || dialog.kind !== "delete" || !token) return;
    try {
      await api(`/api/threads/${dialog.thread.id}`, token, { method: "DELETE" });
      queryClient.removeQueries({ queryKey: ["messages", dialog.thread.id] });
      await queryClient.invalidateQueries({ queryKey: ["threads", user?.id] });
      if (routeThreadId === dialog.thread.id) router.replace("/chat");
      setDialog(null);
      toast.success("会话已删除");
    } catch (error) { toast.error(visibleError(error)); }
  }

  function signOut() {
    controller.current?.abort();
    logout();
    queryClient.clear();
    router.replace("/login");
  }

  async function upload(file: File) {
    if (!token) return;
    if (uploads.length >= 5) return toast.error("一次最多添加 5 个文件");
    const data = new FormData();
    data.append("file", file);
    setUploading(true);
    try {
      const result = await api<Upload>("/api/upload", token, { method: "POST", body: data });
      setUploads(current => [...current, result]);
      toast.success("文件已添加");
    } catch (error) { toast.error(visibleError(error)); }
    finally { setUploading(false); }
  }

  const updateAssistant = useCallback((id: string, updater: (message: Message) => Message) => {
    setLiveMessages(current => current?.map(message => message.id === id ? updater(message) : message) || null);
  }, []);

  async function send(text = draft) {
    const content = text.trim();
    if (!content || !token || sending) return;
    const run = ++activeRun.current;
    const aborter = new AbortController();
    controller.current = aborter;
    setSending(true);
    setDraft("");
    const userMessage: Message = { id: `local-user-${run}`, role: "user", content };
    const assistantId = `local-assistant-${run}`;
    const assistant: Message = {
      id: assistantId, role: "assistant", content: "", notes: [],
      trace: { events: [], mode: currentThread?.mode || mode },
    };
    setLiveMessages([...(liveMessages || messagesQuery.data || []), userMessage, assistant]);
    let threadId = routeThreadId;
    const stageEvents: StageEvent[] = [];
    let elapsedMs: number | undefined;
    try {
      if (!threadId) {
        const thread = await api<Thread>("/api/threads", token, {
          method: "POST", body: JSON.stringify({ mode }),
        });
        threadId = thread.id;
        await queryClient.invalidateQueries({ queryKey: ["threads", user?.id] });
      }
      const response = await fetch(API_BASE + "/api/chat?stream=true", {
        method: "POST",
        signal: aborter.signal,
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ message: content, thread_id: threadId, files: uploads.map(file => file.id) }),
      });
      if (!response.ok || !response.body) {
        let detail = "无法连接到问答服务";
        try { detail = (await response.json()).detail || detail; } catch { /* Keep default. */ }
        throw new ApiError(detail, response.status);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          const line = block.split("\n").find(row => row.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));
          if (event.error) throw new Error(event.error);
          if (event.token) updateAssistant(assistantId, item => ({ ...item, content: item.content + event.token }));
          if (event.type === "stage") {
            stageEvents.push(event as StageEvent);
            updateAssistant(assistantId, item => ({ ...item, trace: { events: [...stageEvents], mode: currentThread?.mode || mode } }));
          }
          if (event.done) {
            completed = true;
            elapsedMs = event.elapsed_ms;
            updateAssistant(assistantId, item => ({ ...item, notes: event.notes || [], trace: { events: [...stageEvents], elapsedMs, mode: currentThread?.mode || mode } }));
          }
        }
      }
      if (!completed) throw new Error("连接中断，回答尚未完成");
      const saved = await api<Message[]>(`/api/threads/${threadId}/messages`, token);
      if (activeRun.current !== run) return;
      const latestAssistant = [...saved].reverse().find(message => message.role === "assistant");
      const withTrace = saved.map(message => message.id === latestAssistant?.id
        ? { ...message, trace: { events: [...stageEvents], elapsedMs, mode: currentThread?.mode || mode } }
        : message);
      queryClient.setQueryData(["messages", threadId], withTrace);
      setLiveMessages(withTrace);
      setUploads([]);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["threads", user?.id] }),
        queryClient.invalidateQueries({ queryKey: ["profile", user?.id] }),
      ]);
      if (!routeThreadId) router.replace(`/chat/${threadId}`);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        signOut();
        toast.error("登录已失效，请重新登录");
      } else if ((error as Error).name === "AbortError") {
        toast.message("已停止生成");
      } else toast.error(visibleError(error));
      if (threadId && activeRun.current === run) {
        try {
          const saved = await api<Message[]>(`/api/threads/${threadId}/messages`, token);
          setLiveMessages(saved);
          queryClient.setQueryData(["messages", threadId], saved);
          if (!routeThreadId) router.replace(`/chat/${threadId}`);
        } catch { /* Keep the visible local message. */ }
      }
    } finally {
      if (activeRun.current === run) setSending(false);
      if (controller.current === aborter) controller.current = null;
    }
  }

  if (!hydrated || !token || !user) return <div className="route-loading"><span className="leaf-loading"><LeafIcon size={34} /></span><span>正在载入会话…</span></div>;

  return <div className="chat-shell">
    <aside className={`sidebar ${mobileOpen ? "open" : ""}`} aria-label="会话与账户侧栏">
      <div className="sidebar-brand"><span className="sidebar-leaf"><LeafIcon size={26} /></span><div><strong>岐问</strong><small>中医知识助手</small></div><button className="sidebar-close" onClick={() => setMobileOpen(false)} aria-label="关闭导航">×</button></div>
      <button className="new-thread-button" onClick={newChat}><span>＋</span>新对话</button>
      <div className="sidebar-section-label">会话记录 <span>{threads.length}</span></div>
      <div className="sidebar-threads">
        {threads.length === 0 && <p className="sidebar-empty">还没有会话，从一个问题开始。</p>}
        {threads.map(thread => <div className={`thread-item ${routeThreadId === thread.id ? "active" : ""}`} key={thread.id}>
          <button className="thread-main" title={thread.title} onClick={() => openThread(thread)}><span className="thread-bullet">◦</span><span>{thread.title}</span></button>
          <button className="thread-more" title="更多操作" aria-label={`管理${thread.title}`} onClick={() => { setDialog({ kind: "rename", thread }); setRenameText(thread.title); }}>···</button>
        </div>)}
      </div>
      <div className="sidebar-bottom">
        <section className="profile-summary">
          <div className="profile-heading"><span>患者画像</span><button onClick={openProfile} aria-label="编辑画像与过敏史">编辑</button></div>
          <p><strong>体质</strong><span>{profileQuery.data?.constitution || "尚未填写"}</span></p>
          <p><strong>过敏史</strong><span>{profileQuery.data?.allergies || "尚未填写"}</span></p>
        </section>
        <div className="account-row"><span className="account-avatar">{user.username.charAt(0).toUpperCase()}</span><div><strong>{user.username}</strong><small>个人空间</small></div><button onClick={signOut} aria-label="退出登录" title="退出登录">↗</button></div>
      </div>
    </aside>
    {mobileOpen && <button className="sidebar-backdrop" aria-label="关闭导航" onClick={() => setMobileOpen(false)} />}
    <main className="chat-main">
      <header className="chat-topbar">
        <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="打开导航">☰</button>
        <div className="chat-breadcrumb"><span>岐问</span><i>/</i><strong>{currentThread?.title || "新对话"}</strong></div>
        <span className="mode-badge">{(currentThread?.mode || mode) === "multi" ? "多专家会诊" : "固定流程"}</span>
      </header>
      <div className="chat-scroll">
        {messages.length === 0 ? <section className="empty-chat">
          <div className="empty-leaf"><LeafIcon size={37} /></div>
          <h1>灵医问诊</h1>
          <p>描述症状、上传相关资料。岐问会整理线索，展示知识检索与审查过程。</p>
          {!routeThreadId && <div className="mode-select" role="group" aria-label="会诊模式">
            <button className={mode === "fixed" ? "selected" : ""} onClick={() => setMode("fixed")}><strong>固定流程</strong><small>循序分析</small></button>
            <button className={mode === "multi" ? "selected" : ""} onClick={() => setMode("multi")}><strong>多专家会诊</strong><small>并行讨论</small></button>
          </div>}
          <div className="example-list">{EXAMPLES.map((example, index) => <button key={example} onClick={() => { setDraft(example); inputRef.current?.focus(); }}><span>0{index + 1}</span>{example}<b>↗</b></button>)}</div>
        </section> : <div className="message-list">
          {messages.map(message => <MessageBubble key={message.id} message={message} username={user.username} mode={currentThread?.mode || mode} streaming={sending && message.id.startsWith("local-assistant-")} />)}
          <div ref={bottomRef} />
        </div>}
      </div>
      <div className="composer-area">
        <div className="composer-inner">
          {uploads.length > 0 && <div className="attachment-list">{uploads.map(file => <span key={file.id}>▧ {file.name}<button aria-label={`移除${file.name}`} onClick={() => setUploads(items => items.filter(item => item.id !== file.id))}>×</button></span>)}</div>}
          <div className="composer-box">
            <textarea
              ref={inputRef}
              value={draft}
              onChange={event => setDraft(event.target.value)}
              onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }}
              placeholder="描述症状，或提出关于中医知识的问题…"
              rows={1}
              disabled={sending}
              aria-label="输入问题"
            />
            <div className="composer-toolbar">
              <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" hidden onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ""; }} />
              <button className="upload-button" onClick={() => fileInputRef.current?.click()} disabled={sending || uploading || uploads.length >= 5}><span>＋</span>{uploading ? "上传中…" : "上传资料"}</button>
              <span className="keyboard-hint">Enter 发送 · Shift + Enter 换行</span>
              {sending
                ? <button className="stop-button" onClick={() => controller.current?.abort()}>■ 停止生成</button>
                : <button className="send-button" onClick={() => void send()} disabled={!draft.trim()}>发送 ↗</button>}
            </div>
          </div>
          <p className="chat-disclaimer">{DISCLAIMER}</p>
        </div>
      </div>
    </main>
    {dialog?.kind === "rename" && <DialogFrame title="重命名会话" onClose={() => setDialog(null)}>
      <form onSubmit={renameThread}><label className="dialog-label" htmlFor="rename-title">会话名称</label><input id="rename-title" autoFocus value={renameText} onChange={event => setRenameText(event.target.value)} maxLength={80} required />
        <div className="dialog-actions"><button type="button" onClick={() => { if (dialog) setDialog({ kind: "delete", thread: dialog.thread }); }} className="danger-text">删除会话</button><span /><button type="button" onClick={() => setDialog(null)}>取消</button><button className="button-primary" type="submit">保存</button></div>
      </form>
    </DialogFrame>}
    {dialog?.kind === "delete" && <DialogFrame title="删除会话" onClose={() => setDialog(null)}>
      <p className="dialog-description">确定删除“{dialog.thread.title}”吗？聊天记录将无法恢复。</p>
      <div className="dialog-actions"><button onClick={() => setDialog(null)}>取消</button><button className="button-danger" onClick={() => void deleteThread()}>确认删除</button></div>
    </DialogFrame>}
    {showProfile && <DialogFrame title="编辑患者画像" onClose={() => setShowProfile(false)}>
      <p className="dialog-description">画像会在你的不同会话中使用。请只填写希望用于演示的信息。</p>
      <form onSubmit={saveProfile} className="profile-form">
        <label htmlFor="constitution">体质信息</label><textarea id="constitution" value={profileDraft.constitution} onChange={event => setProfileDraft({ ...profileDraft, constitution: event.target.value })} maxLength={500} placeholder="如：容易怕冷、睡眠情况等" />
        <label htmlFor="allergies">过敏史</label><textarea id="allergies" value={profileDraft.allergies} onChange={event => setProfileDraft({ ...profileDraft, allergies: event.target.value })} maxLength={500} placeholder="已知的药物或食物过敏" />
        <label htmlFor="history">既往史</label><textarea id="history" value={profileDraft.history} onChange={event => setProfileDraft({ ...profileDraft, history: event.target.value })} maxLength={1000} placeholder="相关既往健康信息" />
        <div className="dialog-actions"><button type="button" onClick={() => setShowProfile(false)}>取消</button><button className="button-primary" type="submit">保存画像</button></div>
      </form>
    </DialogFrame>}
  </div>;
}

