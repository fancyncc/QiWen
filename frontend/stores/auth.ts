"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { User } from "@/lib/types";

type AuthState = {
  token: string | null;
  user: User | null;
  hydrated: boolean;
  setAuth: (token: string, user: User) => void;
  logout: () => void;
  setHydrated: (ready: boolean) => void;
};

export const useAuth = create<AuthState>()(persist(
  set => ({
    token: null,
    user: null,
    hydrated: false,
    setAuth: (token, user) => set({ token, user }),
    logout: () => set({ token: null, user: null }),
    setHydrated: hydrated => set({ hydrated }),
  }),
  {
    name: "qiwen-auth",
    storage: createJSONStorage(() => localStorage),
    partialize: state => ({ token: state.token, user: state.user }),
    skipHydration: true,
  },
));

