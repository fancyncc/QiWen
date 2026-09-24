"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Toaster } from "sonner";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";
import { useAuth } from "@/stores/auth";

export default function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: { queries: { staleTime: 20_000, refetchOnWindowFocus: false, retry: 1 } },
  }));

  useEffect(() => {
    let active = true;
    async function hydrate() {
      await useAuth.persist.rehydrate();
      const state = useAuth.getState();
      if (!state.token) {
        const legacy = localStorage.getItem("qiwen_token");
        if (legacy) {
          try {
            const user = await api<User>("/api/me", legacy);
            if (active) state.setAuth(legacy, user);
          } catch { /* The old session has expired. */ }
          localStorage.removeItem("qiwen_token");
          localStorage.removeItem("qiwen_active_thread");
        }
      }
      if (active) useAuth.getState().setHydrated(true);
    }
    void hydrate();
    return () => { active = false; };
  }, []);

  return <QueryClientProvider client={client}>
    {children}
    <Toaster position="top-center" richColors closeButton />
  </QueryClientProvider>;
}

