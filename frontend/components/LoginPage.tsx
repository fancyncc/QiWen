"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import LeafIcon from "./LeafIcon";
import { api } from "@/lib/api";
import type { AuthResult } from "@/lib/types";
import { useAuth } from "@/stores/auth";

export default function LoginPage() {
  const router = useRouter();
  const { hydrated, token, setAuth } = useAuth();
  const [tab, setTab] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (hydrated && token) router.replace("/chat");
  }, [hydrated, token, router]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim() || !password) return toast.error("请输入用户名和密码");
    if (tab === "register" && password !== confirm) return toast.error("两次输入的密码不一致");
    if (password.length < 8) return toast.error("密码至少需要 8 位");
    setBusy(true);
    try {
      const result = await api<AuthResult>(
        tab === "login" ? "/api/login" : "/api/register",
        null,
        { method: "POST", body: JSON.stringify({ username: username.trim(), password }) },
      );
      setAuth(result.access_token, result.user);
      toast.success(tab === "login" ? "登录成功" : "注册成功");
      router.replace("/chat");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "操作失败，请稍后重试");
    } finally {
      setBusy(false);
    }
  }

  if (!hydrated) return <div className="route-loading"><span className="leaf-loading"><LeafIcon size={34} /></span><span>正在打开岐问…</span></div>;
  return <main className="login-page">
    <div className="login-decoration decoration-left" aria-hidden="true" />
    <div className="login-decoration decoration-right" aria-hidden="true" />
    <section className="login-card" aria-label="登录与注册">
      <div className="login-brand">
        <div className="leaf-emblem"><LeafIcon size={31} /></div>
        <div className="eyebrow">QIWEN · TRADITIONAL KNOWLEDGE</div>
        <h1>岐问</h1>
        <p>中医知识辅助问答与会诊演示</p>
      </div>
      <div className="auth-tabs" role="tablist" aria-label="账号操作">
        <button role="tab" aria-selected={tab === "login"} className={tab === "login" ? "active" : ""} onClick={() => setTab("login")}>登录</button>
        <button role="tab" aria-selected={tab === "register"} className={tab === "register" ? "active" : ""} onClick={() => setTab("register")}>注册</button>
      </div>
      <form onSubmit={submit} className="login-form">
        <label htmlFor="username">用户名</label>
        <input id="username" autoComplete="username" value={username} onChange={event => setUsername(event.target.value)} placeholder="请输入用户名" minLength={3} maxLength={32} required />
        <label htmlFor="password">密码</label>
        <input id="password" type="password" autoComplete={tab === "login" ? "current-password" : "new-password"} value={password} onChange={event => setPassword(event.target.value)} placeholder={tab === "register" ? "至少 8 位密码" : "请输入密码"} minLength={8} required />
        {tab === "register" && <>
          <label htmlFor="confirm">确认密码</label>
          <input id="confirm" type="password" autoComplete="new-password" value={confirm} onChange={event => setConfirm(event.target.value)} placeholder="请再次输入密码" minLength={8} required />
        </>}
        <button className="button-primary login-submit" disabled={busy} type="submit">{busy ? (tab === "login" ? "登录中…" : "注册中…") : (tab === "login" ? "登录" : "注册并进入")}</button>
      </form>
      <p className="login-footnote">仅供学习与研究参考 · 不代替执业医师面诊</p>
    </section>
    <p className="login-copyright">岐问 QIWEN · 理 · 法 · 方 · 药</p>
  </main>;
}

