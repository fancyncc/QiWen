import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import LoginPage from "../page";
import { useAuthStore } from "@/stores/auth";

const { replace } = vi.hoisted(() => ({ replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));

beforeEach(() => {
  useAuthStore.getState().logout();
  replace.mockClear();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function enterCredentials() {
  fireEvent.change(screen.getByLabelText("用户名"), { target: { value: "legacy_test" } });
  fireEvent.change(screen.getByLabelText("密码"), { target: { value: "wrong-password" } });
  fireEvent.click(screen.getByRole("button", { name: "登录" }));
}

describe("login failure feedback", () => {
  it("shows the API credential error without clearing auth or redirecting", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "用户名或密码错误" }), { status: 401 })));
    const logout = vi.spyOn(useAuthStore.getState(), "logout");
    render(<LoginPage />);
    enterCredentials();
    expect(await screen.findByRole("alert")).toHaveTextContent("用户名或密码错误");
    expect(logout).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();
  });

  it("keeps a visible error when the network request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<LoginPage />);
    enterCredentials();
    expect(await screen.findByRole("alert")).toHaveTextContent("登录失败，请检查网络后重试");
    expect(replace).not.toHaveBeenCalled();
  });

  it("shows missing credentials and does not send a request", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    render(<LoginPage />);
    fireEvent.click(screen.getByRole("button", { name: "登录" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("请填写用户名和密码");
    expect(fetch).not.toHaveBeenCalled();
  });
});
