"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/sonner";
import { useEffect, useState, type ReactNode } from "react";
import { useAuthStore } from "@/stores/auth";

/** 全局 Provider：TanStack Query + Toaster（sonner）。 */
export function Providers({ children }: { children: ReactNode }) {
  const username = useAuthStore((state) => state.username);
  return <AccountProviders key={username ? `user:${username}` : "signed-out"}>{children}</AccountProviders>;
}

function AccountProviders({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false },
        },
      })
  );
  // Changing accounts unmounts this provider, cancels old queries and destroys
  // their cache. A late response can never populate the next account's client.
  useEffect(() => () => client.clear(), [client]);
  return (
    <QueryClientProvider client={client}>
      {children}
      <Toaster richColors position="top-center" />
    </QueryClientProvider>
  );
}
