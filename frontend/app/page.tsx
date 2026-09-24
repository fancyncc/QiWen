"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/stores/auth";

export default function Home() {
  const router = useRouter();
  const { token, hydrated } = useAuth();
  useEffect(() => {
    if (hydrated) router.replace(token ? "/chat" : "/login");
  }, [hydrated, token, router]);
  return <div className="route-loading"><span className="leaf-loading">❧</span><span>正在打开岐问…</span></div>;
}
