"use client";

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import type { ProfileSuggestion, SuggestionDecision } from "@/lib/types";
import { toast } from "sonner";

const LABELS: Record<string, string> = {
  allergy: "过敏史", constitution: "自述体质", history: "既往史",
  pregnancy_lactation: "孕哺状态", medication: "用药情况",
};
type Row = ProfileSuggestion & { action: SuggestionDecision["action"]; entity: string; details: string; replace: boolean };

export function ProfileSuggestions({ threadId, streaming, refreshKey }: {
  threadId: string; streaming: boolean; refreshKey: number;
}) {
  if (!threadId || streaming) return null;
  return <SuggestionDialog key={JSON.stringify([threadId, refreshKey])} threadId={threadId} />;
}

function SuggestionDialog({ threadId }: { threadId: string }) {
  const qc = useQueryClient();
  const [rows, setRows] = useState<Row[]>([]);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let stale = false;
    api.getProfileSuggestions(threadId).then((data) => {
      if (stale) return;
      setRevision(data.revision);
      setRows(data.items.map((item) => ({ ...item, action: "accept", entity: item.fact.entity, details: item.fact.details, replace: false })));
    }).catch((error) => {
      if (!stale) toast.error(error.message || "画像建议读取失败");
    });
    return () => { stale = true; };
  }, [threadId, reload]);

  function change(id: string, patch: Partial<Row>) {
    setRows((old) => old.map((row) => row.id === id ? { ...row, ...patch } : row));
  }
  async function submit(skipAll = false) {
    if (busy) return;
    setBusy(true);
    try {
      await api.resolveProfileSuggestions(revision, rows.map((row) => ({
        id: row.id, action: skipAll ? "dismiss" : row.action,
        confirm_self: !skipAll && row.action === "accept", replace: row.replace,
        ...(row.entity !== row.fact.entity ? { entity: row.entity } : {}),
        ...(row.details !== row.fact.details ? { details: row.details } : {}),
      })));
      setRows([]);
      await qc.invalidateQueries({ queryKey: ["profile"] });
      if (!skipAll) toast.success("已处理画像建议");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存失败");
      if (typeof error === "object" && error && "status" in error && error.status === 409) {
        setRows([]);
        setReload((old) => old + 1);
      }
    } finally { setBusy(false); }
  }
  const invalid = rows.some((row) => row.action === "accept" && (!row.entity.trim() || (row.conflict && !row.replace)));
  return (
    <Dialog open={rows.length > 0} onOpenChange={(open) => { if (!open && !busy) void submit(true); }}>
      <DialogContent className="sm:max-w-xl" showCloseButton={!busy}>
        <DialogHeader>
          <DialogTitle>是否加入你的长期画像？</DialogTitle>
          <DialogDescription>
            以下资料来自本次对话。请核对是否属于你本人，可修改或逐条排除。确认只记录用户自述，不代表医学核实；不保存也不会忽略本轮已知风险。
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[55vh] space-y-4 overflow-y-auto pr-1">
          {rows.map((row) => (
            <div key={row.id} className="space-y-2 rounded-xl border border-border bg-secondary/30 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{LABELS[row.fact.category] ?? "健康资料"} · {row.fact.assertion === "absent" ? "明确否认" : "自述"}</span>
                <select aria-label={"处理" + row.fact.entity} value={row.action} disabled={busy}
                  onChange={(e) => change(row.id, { action: e.target.value as Row["action"] })}
                  className="rounded-md border border-border bg-background px-2 py-1">
                  <option value="accept">加入本人画像</option><option value="dismiss">暂不保存</option><option value="other">属于他人</option>
                </select>
              </div>
              <label className="block text-xs text-muted-foreground">
                内容
                <Input aria-label={row.fact.entity + "内容"} maxLength={80} value={row.entity} disabled={busy || row.action !== "accept"}
                  onChange={(e) => change(row.id, { entity: e.target.value })} className="mt-1" />
              </label>
              <label className="block text-xs text-muted-foreground">
                补充说明
                <Input aria-label={row.fact.entity + "说明"} maxLength={500} value={row.details} disabled={busy || row.action !== "accept"}
                  onChange={(e) => change(row.id, { details: e.target.value })} className="mt-1" />
              </label>
              <p className="break-words text-xs leading-5 text-muted-foreground">来源：{row.fact.source.quote}</p>
              <p className="text-xs text-muted-foreground">时间范围：{({ current: "当前", past: "既往", ended: "已结束", unknown: "待确认" } as Record<string, string>)[row.fact.temporal_scope] ?? "待确认"}</p>
              {row.conflict && <div className="rounded-lg border border-amber-300 bg-amber-50 p-2 text-xs text-amber-950">
                <p>与已有资料不一致：{row.previous || "已有风险记录"}。请核对后明确更正。</p>
                <label className="mt-2 flex items-center gap-2">
                  <input type="checkbox" checked={row.replace} disabled={busy || row.action !== "accept"}
                    onChange={(e) => change(row.id, { replace: e.target.checked })} />
                  我确认更正对应的旧资料
                </label>
              </div>}
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={() => void submit(true)}>全部暂不保存</Button>
          <Button disabled={busy || invalid} onClick={() => void submit()}>
            {busy ? "保存中…" : rows.some((row) => row.action === "accept") ? "确认所选资料属于我，并保存" : "确认不加入本人画像"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
