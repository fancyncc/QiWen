import { beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MedicalRecords } from "../MedicalRecords";
import type { MedicalRecord, UserProfile } from "@/lib/types";

const mocked = vi.hoisted(() => ({ save: vi.fn(), remove: vi.fn(), invalidate: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: { saveMedicalRecord: mocked.save, deleteMedicalRecord: mocked.remove } }));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({ invalidateQueries: mocked.invalidate }) }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));
const record: MedicalRecord = { id: "record-1", record_date: "2025-09-10", institution: "测试医院", diagnosis: "既往胃炎",
  treatment: "当时接受治疗", outcome: "复诊好转", created_at: "2025-09-10", updated_at: "2025-09-10" };
const profile: UserProfile = { patient_id: "patient-1", constitution: "阳虚体质", allergies: "青霉素", past_history: [], revision: 3, medical_records: [] };
beforeEach(() => { cleanup(); vi.clearAllMocks(); mocked.save.mockResolvedValue({}); mocked.remove.mockResolvedValue({}); });

describe("既往诊疗记录", () => {
  it("空档案可以补充本人就诊经过，未填写诊断不能保存，取消不写入", async () => {
    render(<MedicalRecords profile={profile} />);
    fireEvent.click(screen.getByText("既往诊疗记录"));
    fireEvent.click(screen.getByRole("button", { name: "补充诊疗记录" }));
    expect(screen.getByRole("button", { name: "确认是本人记录并保存" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("既往诊断或就诊原因"), { target: { value: "既往胃炎" } });
    fireEvent.change(screen.getByLabelText("就诊日期"), { target: { value: "2025-09-10" } });
    fireEvent.change(screen.getByLabelText("治疗经过"), { target: { value: "按当时病历记录" } });
    fireEvent.click(screen.getByRole("button", { name: "确认是本人记录并保存" }));
    await waitFor(() => expect(mocked.save).toHaveBeenCalledWith("patient-1", expect.objectContaining({ diagnosis: "既往胃炎",
      record_date: "2025-09-10", treatment: "按当时病历记录", confirm_self: true, revision: 3 }), undefined));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "补充诊疗记录" }));
    fireEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(mocked.save).toHaveBeenCalledTimes(1);
  });
  it("显示既往诊疗详情并能修改原记录", async () => {
    render(<MedicalRecords profile={{ ...profile, medical_records: [record] }} />);
    fireEvent.click(screen.getByText("既往诊疗记录"));
    expect(screen.getByText("治疗经过：当时接受治疗")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "编辑既往胃炎" }));
    expect(screen.getByLabelText("既往诊断或就诊原因")).toHaveValue("既往胃炎");
    fireEvent.change(screen.getByLabelText("疗效及复诊情况"), { target: { value: "后续稳定" } });
    fireEvent.click(screen.getByRole("button", { name: "确认是本人记录并保存" }));
    await waitFor(() => expect(mocked.save).toHaveBeenCalledWith("patient-1", expect.objectContaining({ outcome: "后续稳定" }), "record-1"));
  });
  it("删除必须确认，失败后保留记录并刷新画像", async () => {
    mocked.remove.mockRejectedValueOnce(new Error("画像已变化"));
    render(<MedicalRecords profile={{ ...profile, medical_records: [record] }} />);
    fireEvent.click(screen.getByText("既往诊疗记录"));
    fireEvent.click(screen.getByRole("button", { name: "删除既往胃炎" }));
    expect(mocked.remove).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "确认删除" }));
    await waitFor(() => expect(mocked.remove).toHaveBeenCalledWith("patient-1", "record-1", 3));
    await waitFor(() => expect(mocked.invalidate).toHaveBeenCalledWith({ queryKey: ["profile"] }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("既往胃炎")).toBeInTheDocument();
  });
});
