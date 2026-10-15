import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { MessageItem } from "@/lib/types";
import { ChatWindow } from "../ChatWindow";

const state = vi.hoisted(() => ({ messages: [] as MessageItem[] }));
vi.mock("@/hooks/useChat", () => ({ useChat: () => ({ messages: state.messages, streaming: false }) }));
vi.mock("@/stores/auth", () => ({ useAuthStore: (select: (value: { username: string }) => unknown) => select({ username: "test" }) }));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: [], isPending: false }), useQueryClient: () => ({}) }));
vi.mock("../ProfileSuggestions", () => ({ ProfileSuggestions: () => null }));
vi.mock("../ProfileFacts", () => ({ ProfileFacts: () => null }));
vi.mock("../MessageBubble", () => ({ MessageBubble: ({ message }: { message: MessageItem }) => <p>{message.content}</p> }));

afterEach(() => { cleanup(); vi.restoreAllMocks(); state.messages = []; });

it("new replies scroll the message viewport without scrolling the page or composer", () => {
  const scrolled: HTMLElement[] = [];
  const scrollTo = vi.fn(function (this: HTMLElement) { scrolled.push(this); });
  vi.stubGlobal("scrollTo", vi.fn());
  Object.defineProperty(HTMLElement.prototype, "scrollTo", { configurable: true, value: scrollTo });
  const { rerender } = render(<ChatWindow threadId="test-thread" />);
  const viewport = screen.getByRole("region", { name: "问诊消息" });
  Object.defineProperty(viewport, "scrollHeight", { configurable: true, value: 3200 });
  state.messages = [{ role: "assistant", content: "长会诊结果" }];
  rerender(<ChatWindow threadId="test-thread" />);
  expect(scrollTo).toHaveBeenLastCalledWith({ top: 3200, behavior: "smooth" });
  expect(scrolled.every((element) => element === viewport)).toBe(true);
  expect(window.scrollTo).not.toHaveBeenCalled();
  expect(screen.getByPlaceholderText("请描述您的症状...")).toBeInTheDocument();
  vi.unstubAllGlobals();
  delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollTo;
});
