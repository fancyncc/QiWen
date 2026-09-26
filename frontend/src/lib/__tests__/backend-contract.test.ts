import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { mapMessage } from "../backend-contract";
import { useAuthStore } from "@/stores/auth";

describe("恢复后端接口适配", () => {
  beforeEach(() => {
    useAuthStore.setState({ token: "original-token" });
    vi.unstubAllGlobals();
  });
  it("会话 id 转成页面 thread_id，重命名提交 title", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200,
      json: async () => [{ id: "thread", title: "旧会话", created_at: "today" }] });
    vi.stubGlobal("fetch", fetchMock);
    expect((await api.listThreads())[0].thread_id).toBe("thread");
    await api.renameThread("thread", "改名");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ title: "改名" });
  });
  it("删除会话接受 204 空响应", async () => {
    const json = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 204, json }));
    await expect(api.deleteThread("thread")).resolves.toBeUndefined();
    expect(json).not.toHaveBeenCalled();
  });
  it("画像使用当前登录用户 id，不把用户名当数据库 id", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ id: "user-id" }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({
        user_id: "user-id", constitution: "", allergies: "", history: "既往记录",
      }) });
    vi.stubGlobal("fetch", fetchMock);
    expect((await api.getProfile("coco")).past_history).toEqual(["既往记录"]);
    expect(fetchMock.mock.calls[1][0]).toBe("/api/profiles/user-id");
  });
  it("上传把记录 id 作为附件引用，保留文件名", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 201,
      json: async () => ({ id: "file-id", name: "病历.txt" }) }));
    expect(await api.upload(new File(["text"], "病历.txt"))).toEqual({
      path: "file-id", filename: "病历.txt",
    });
  });
  it("恢复历史的专家笔记、阶段和耗时", () => {
    expect(mapMessage({ role: "assistant", content: "答案", notes: [{ expert: "方剂", content: "笔记" }],
      trace: { events: [{ stage: "formula", label: "方剂", status: "done" }], elapsedMs: 123 },
    })).toEqual({ role: "assistant", content: "答案", notes: [{ specialist: "方剂", reasoning: "笔记" }],
      stages: [{ stage: "fangji", label: "方剂", status: "done" }], elapsedMs: 123 });
  });
  it("错误密码保留登录态并显示服务端错误", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 401,
      json: async () => ({ detail: "用户名或密码错误" }) }));
    await expect(api.login("coco", "wrongpassword")).rejects.toThrow("用户名或密码错误");
    expect(useAuthStore.getState().token).toBe("original-token");
  });
});
