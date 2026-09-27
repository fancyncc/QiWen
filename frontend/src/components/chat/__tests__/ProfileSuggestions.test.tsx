import { beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ProfileSuggestions } from "../ProfileSuggestions";

const mocked = vi.hoisted(() => ({ get: vi.fn(), resolve: vi.fn(), invalidate: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: { getProfileSuggestions: mocked.get, resolveProfileSuggestions: mocked.resolve } }));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({ invalidateQueries: mocked.invalidate }) }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));

const item = (conflict = false) => ({
  id: "proposal-one", conflict, previous: conflict ? "无过敏" : "",
  fact: { id: "source-fact", category: "allergy", entity: "甘草", assertion: "present", temporal_scope: "current",
    details: "服用后起疹子", source: { quote: "对甘草过敏，服用后起疹子", collected_at: "2026-09-27" } },
});
beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  mocked.get.mockResolvedValue({ revision: 3, items: [item()] });
  mocked.resolve.mockResolvedValue({ revision: 4 });
  mocked.invalidate.mockResolvedValue(undefined);
});

describe("长期画像确认", () => {
  it("仅展示来源，不自动保存；确认时明确声明属于本人", async () => {
    render(<ProfileSuggestions threadId="one" streaming={false} refreshKey={2} />);
    await screen.findByRole("dialog");
    expect(screen.getByText(/来源：对甘草过敏/)).toBeInTheDocument();
    expect(mocked.resolve).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("甘草内容"), { target: { value: "青霉素" } });
    fireEvent.click(screen.getByRole("button", { name: "确认所选资料属于我，并保存" }));
    await waitFor(() => expect(mocked.resolve).toHaveBeenCalledWith(3, [
      expect.objectContaining({ id: "proposal-one", action: "accept", confirm_self: true, entity: "青霉素" }),
    ]));
  });
  it("关闭弹窗等同暂不保存，不会接受默认选择", async () => {
    render(<ProfileSuggestions threadId="one" streaming={false} refreshKey={2} />);
    await screen.findByRole("dialog");
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() => expect(mocked.resolve).toHaveBeenCalledWith(3, [
      expect.objectContaining({ action: "dismiss", confirm_self: false }),
    ]));
  });
  it("资料属于他人时不确认本人，也不加入本人档案", async () => {
    render(<ProfileSuggestions threadId="one" streaming={false} refreshKey={2} />);
    await screen.findByRole("dialog");
    fireEvent.change(screen.getByLabelText("处理甘草"), { target: { value: "other" } });
    fireEvent.click(screen.getByRole("button", { name: "确认不加入本人画像" }));
    await waitFor(() => expect(mocked.resolve).toHaveBeenCalledWith(3, [
      expect.objectContaining({ action: "other", confirm_self: false }),
    ]));
  });
  it("冲突必须明确勾选更正才能保存", async () => {
    mocked.get.mockResolvedValue({ revision: 3, items: [item(true)] });
    render(<ProfileSuggestions threadId="one" streaming={false} refreshKey={2} />);
    await screen.findByRole("dialog");
    const save = screen.getByRole("button", { name: "确认所选资料属于我，并保存" });
    expect(save).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "我确认更正对应的旧资料" }));
    expect(save).not.toBeDisabled();
    fireEvent.click(save);
    await waitFor(() => expect(mocked.resolve).toHaveBeenCalledWith(3, [
      expect.objectContaining({ replace: true }),
    ]));
  });
  it("运行过程中不弹窗，迟到的另一会话结果不能显示", async () => {
    let complete!: (data: unknown) => void;
    mocked.get.mockImplementationOnce(() => new Promise((resolve) => { complete = resolve; }));
    const view = render(<ProfileSuggestions threadId="one" streaming={false} refreshKey={2} />);
    view.rerender(<ProfileSuggestions threadId="two" streaming={true} refreshKey={3} />);
    complete({ revision: 3, items: [item()] });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(mocked.resolve).not.toHaveBeenCalled();
  });
});
