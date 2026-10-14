"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronRight, Pencil, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { MedicalRecord, UserProfile } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";

const emptyForm = { record_date: "", institution: "", diagnosis: "", treatment: "", outcome: "" };

export function MedicalRecords({ profile }: { profile?: UserProfile }) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<MedicalRecord | "new" | null>(null);
  const [deleting, setDeleting] = useState<MedicalRecord | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [busy, setBusy] = useState(false);
  const records = profile?.medical_records ?? [];

  function edit(record: MedicalRecord | "new") {
    setForm(record === "new" ? emptyForm : { record_date: record.record_date ?? "", institution: record.institution,
      diagnosis: record.diagnosis, treatment: record.treatment, outcome: record.outcome });
    setEditing(record);
  }
  async function save() {
    if (!profile || !editing || busy || !form.diagnosis.trim()) return;
    setBusy(true);
    try {
      await api.saveMedicalRecord(profile.patient_id, { ...form, record_date: form.record_date || null,
        revision: profile.revision ?? 0, confirm_self: true }, editing === "new" ? undefined : editing.id);
      await qc.invalidateQueries({ queryKey: ["profile"] });
      setEditing(null);
      toast.success("既往诊疗记录已保存");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存失败");
      await qc.invalidateQueries({ queryKey: ["profile"] });
    } finally { setBusy(false); }
  }
  async function remove() {
    if (!profile || !deleting || busy) return;
    setBusy(true);
    try {
      await api.deleteMedicalRecord(profile.patient_id, deleting.id, profile.revision ?? 0);
      await qc.invalidateQueries({ queryKey: ["profile"] });
      setDeleting(null);
      toast.success("诊疗记录已删除");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除失败");
      await qc.invalidateQueries({ queryKey: ["profile"] });
    } finally { setBusy(false); }
  }

  return <>
    <details className="group mt-3 rounded-lg border border-sidebar-border text-sm">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" aria-hidden />
        既往诊疗记录
        {records.length > 0 && <span className="ml-auto text-xs text-muted-foreground">{records.length} 条</span>}
      </summary>
      <div className="space-y-3 border-t border-sidebar-border p-2">
        <p className="text-xs leading-5 text-muted-foreground">保存本人既往诊断、治疗经过及疗效，本次症状保留在会话中。</p>
        {!!profile?.past_history.length && <div className="rounded-md bg-background/60 p-2 text-xs">
          <p className="mb-1 font-medium">已记录既往病史</p>
          <p className="whitespace-pre-wrap break-words">{profile.past_history.join("；")}</p>
        </div>}
        <div className="max-h-64 space-y-2 overflow-y-auto">
          {records.map((record) => <article key={record.id} className="space-y-1 rounded-md bg-background/60 p-2 text-xs">
            <div className="flex items-start justify-between gap-1">
              <p className="min-w-0 break-words font-medium">{record.diagnosis}</p>
              <div className="flex shrink-0 gap-1">
                <button className="rounded p-1 hover:bg-secondary" onClick={() => edit(record)} aria-label={`编辑${record.diagnosis}`}><Pencil className="h-3 w-3" /></button>
                <button className="rounded p-1 hover:bg-secondary" onClick={() => setDeleting(record)} aria-label={`删除${record.diagnosis}`}><Trash2 className="h-3 w-3" /></button>
              </div>
            </div>
            <p className="break-words text-muted-foreground">{record.record_date || "日期未填写"}{record.institution ? ` · ${record.institution}` : ""}</p>
            {record.treatment && <p className="whitespace-pre-wrap break-words">治疗经过：{record.treatment}</p>}
            {record.outcome && <p className="whitespace-pre-wrap break-words">疗效及复诊：{record.outcome}</p>}
            <p className="text-muted-foreground">本人补充 · 未经医学核实</p>
          </article>)}
          {!records.length && !profile?.past_history.length && <p className="py-1 text-xs text-muted-foreground">暂无既往诊疗记录</p>}
        </div>
        <Button variant="outline" size="sm" className="w-full" disabled={!profile} onClick={() => edit("new")}><Plus className="h-3.5 w-3.5" />补充诊疗记录</Button>
      </div>
    </details>
    <Dialog open={!!editing} onOpenChange={(open) => { if (!open && !busy) setEditing(null); }}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-lg" showCloseButton={!busy}>
        <DialogHeader>
          <DialogTitle>{editing === "new" ? "补充既往诊疗记录" : "编辑既往诊疗记录"}</DialogTitle>
          <DialogDescription>请填写本人真实就诊经历。保存仅记录本人提供的资料，不把本轮模型分析当作既往诊断。</DialogDescription>
        </DialogHeader>
        <form className="space-y-3" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1 text-sm">就诊日期<Input aria-label="就诊日期" type="date" value={form.record_date} disabled={busy} onChange={(e) => setForm({ ...form, record_date: e.target.value })} /></label>
            <label className="space-y-1 text-sm">就诊机构<Input aria-label="就诊机构" maxLength={120} value={form.institution} disabled={busy} onChange={(e) => setForm({ ...form, institution: e.target.value })} placeholder="可选" /></label>
          </div>
          <label className="block space-y-1 text-sm">既往诊断或就诊原因（必填）<Input aria-label="既往诊断或就诊原因" required maxLength={200} value={form.diagnosis} disabled={busy} onChange={(e) => setForm({ ...form, diagnosis: e.target.value })} placeholder="按当时诊断或病历填写" /></label>
          <label className="block space-y-1 text-sm">治疗经过<Textarea aria-label="治疗经过" maxLength={1000} value={form.treatment} disabled={busy} onChange={(e) => setForm({ ...form, treatment: e.target.value })} placeholder="当时的治疗、用药或其他处置" /></label>
          <label className="block space-y-1 text-sm">疗效及复诊情况<Textarea aria-label="疗效及复诊情况" maxLength={500} value={form.outcome} disabled={busy} onChange={(e) => setForm({ ...form, outcome: e.target.value })} placeholder="治疗后变化、复诊结论等" /></label>
          <DialogFooter>
            <Button type="button" variant="outline" disabled={busy} onClick={() => setEditing(null)}>取消</Button>
            <Button type="submit" disabled={busy || !form.diagnosis.trim()}>{busy ? "保存中…" : "确认是本人记录并保存"}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
    <Dialog open={!!deleting} onOpenChange={(open) => { if (!open && !busy) setDeleting(null); }}>
      <DialogContent showCloseButton={!busy}>
        <DialogHeader><DialogTitle>删除这条诊疗记录？</DialogTitle><DialogDescription>将移除“{deleting?.diagnosis}”这条记录。</DialogDescription></DialogHeader>
        <DialogFooter><Button variant="outline" disabled={busy} onClick={() => setDeleting(null)}>取消</Button><Button variant="destructive" disabled={busy} onClick={() => void remove()}>确认删除</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  </>;
}
