import { useEffect } from "react";
import { QueryClient, useQueryClient } from "@tanstack/react-query";
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { Providers } from "../providers";
import { useAuthStore } from "@/stores/auth";

vi.mock("@/components/ui/sonner", () => ({ Toaster: () => null }));
afterEach(() => { cleanup(); useAuthStore.getState().logout(); });

it("logout and account changes destroy cached private data and isolate late queries", async () => {
  const clients: QueryClient[] = [];
  function Probe() {
    const client = useQueryClient();
    useEffect(() => { clients.push(client); }, [client]);
    return null;
  }
  useAuthStore.getState().setAuth("test-a", "account-a");
  render(<Providers><Probe /></Providers>);
  const first = clients.at(-1)!;
  first.setQueryData(["threads", "account-a"], ["private title"]);
  let finish!: (value: string) => void;
  const pending = first.fetchQuery({ queryKey: ["pending"], queryFn: () => new Promise<string>((resolve) => { finish = resolve; }) }).catch(() => undefined);
  act(() => useAuthStore.getState().logout());
  expect(first.getQueryCache().getAll()).toHaveLength(0);
  act(() => useAuthStore.getState().setAuth("test-b", "account-b"));
  const second = clients.at(-1)!;
  expect(second).not.toBe(first);
  expect(second.getQueryData(["threads", "account-a"])).toBeUndefined();
  await act(async () => { finish("late private value"); await pending; });
  expect(second.getQueryData(["pending"])).toBeUndefined();
});
