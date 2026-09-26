"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Leaf } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";

/** 登录/注册页（Tabs 切换），成功后存 token 并跳 /chat。 */
export default function LoginPage() {
  const router = useRouter();
  const setAuth = useAuthStore((s) => s.setAuth);
  const [tab, setTab] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function showError(message: string) {
    setError(message);
    toast.error(message);
  }

  async function handleLogin() {
    setError("");
    if (!username || !password) return showError("请填写用户名和密码");
    setLoading(true);
    try {
      const res = await api.login(username, password);
      setAuth(res.access_token, username);
      router.replace("/chat");
    } catch (e) {
      showError(e instanceof ApiError ? e.message : "登录失败，请检查网络后重试");
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister() {
    setError("");
    if (!username || !password) return showError("请填写用户名和密码");
    if (password.length < 6) return showError("密码至少 6 位");
    if (password !== confirm) return showError("两次密码不一致");
    setLoading(true);
    try {
      await api.register(username, password);
      toast.success("注册成功，正在登录...");
      const res = await api.login(username, password);
      setAuth(res.access_token, username);
      router.replace("/chat");
    } catch (e) {
      showError(e instanceof ApiError ? e.message : "注册失败，请检查网络后重试");
    } finally {
      setLoading(false);
    }
  }

  const labelCls = "text-sm font-medium leading-none";

  return (
    <div className="auth-page flex flex-1 items-center justify-center px-4 py-10 sm:px-6">
      <Card className="auth-card w-full max-w-md gap-6 rounded-3xl py-8 ring-border/80 sm:py-10">
        <CardHeader className="space-y-3 px-6 text-center sm:px-8">
          <div className="mx-auto mb-2 flex h-16 w-16 items-center justify-center rounded-2xl border border-primary/15 bg-secondary shadow-sm">
            <Leaf className="h-7 w-7 text-primary" aria-hidden />
          </div>
          <CardTitle className="text-2xl tracking-wide">岐问 · 中医诊疗助手</CardTitle>
          <p className="text-sm leading-relaxed text-muted-foreground">基于中医理法方药的智能诊疗对话</p>
        </CardHeader>
        <CardContent className="px-6 sm:px-8">
          <Tabs value={tab} onValueChange={(value) => { setTab(value); setError(""); }}>
            <TabsList className="grid w-full grid-cols-2 rounded-xl p-1 group-data-horizontal/tabs:h-11">
              <TabsTrigger className="rounded-lg data-active:bg-card data-active:text-primary" value="login">登录</TabsTrigger>
              <TabsTrigger className="rounded-lg data-active:bg-card data-active:text-primary" value="register">注册</TabsTrigger>
            </TabsList>

            {error && (
              <p role="alert" className="mt-4 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </p>
            )}

            <TabsContent value="login" className="mt-5 space-y-5 [&_input]:h-11 [&_input]:bg-background/60 [&_input]:px-3 [&_input]:focus-visible:ring-ring/20">
              <div className="space-y-2">
                <label htmlFor="u" className={labelCls}>用户名</label>
                <Input id="u" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="请输入用户名" autoComplete="username" />
              </div>
              <div className="space-y-2">
                <label htmlFor="p" className={labelCls}>密码</label>
                <Input id="p" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="请输入密码" autoComplete="current-password" onKeyDown={(e) => e.key === "Enter" && handleLogin()} />
              </div>
              <Button className="h-11 w-full rounded-xl shadow-sm" onClick={handleLogin} disabled={loading}>
                {loading ? "登录中..." : "登录"}
              </Button>
            </TabsContent>

            <TabsContent value="register" className="mt-5 space-y-5 [&_input]:h-11 [&_input]:bg-background/60 [&_input]:px-3 [&_input]:focus-visible:ring-ring/20">
              <div className="space-y-2">
                <label htmlFor="ru" className={labelCls}>用户名</label>
                <Input id="ru" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="请输入用户名" autoComplete="username" />
              </div>
              <div className="space-y-2">
                <label htmlFor="rp" className={labelCls}>密码</label>
                <Input id="rp" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="至少 6 位" autoComplete="new-password" />
              </div>
              <div className="space-y-2">
                <label htmlFor="rc" className={labelCls}>确认密码</label>
                <Input id="rc" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="再次输入密码" autoComplete="new-password" />
              </div>
              <Button className="h-11 w-full rounded-xl shadow-sm" onClick={handleRegister} disabled={loading}>
                {loading ? "注册中..." : "注册"}
              </Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
