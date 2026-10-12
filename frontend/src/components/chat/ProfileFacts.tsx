"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { toast } from "sonner";

export function ProfileFacts() {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const username = useAuthStore((state) => state.username) ?? "";
  const qc = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["profile", username], queryFn: () => api.getProfile(), enabled: open,
  });
  async function revoke(id: string) {
    if (busy || data?.revision === undefined) return;
    setBusy(true);
    try {
      await api.revokeProfileFact(id, data.revision);
      setRemoving(null);
      await qc.invalidateQueries({ queryKey: ["profile"] });
      toast.success("已撤销长期保存；本会话已有风险仍需核对");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "撤销失败");
      await qc.invalidateQueries({ queryKey: ["profile"] });
    } finally { setBusy(false); }
  }
  return <>
    <Button variant="ghost" size="sm" onClick={() => setOpen(true)}>已确认画像</Button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>已确认的长期资料</DialogTitle>
          <DialogDescription>以下为你确认保存的本人自述，保留原始来源。撤销长期保存不会自动否认本轮已知风险；手填资料可在侧栏编辑。</DialogDescription>
        </DialogHeader>
        <div className="max-h-[55vh] space-y-3 overflow-y-auto">
          {isLoading ? <p>加载中…</p> : error ? <p role="alert">画像读取失败，请重新打开。</p> : !data?.facts?.length ? <p className="text-muted-foreground">暂无已确认的对话资料。</p> : data.facts.map((fact) =>
            <div key={fact.id} className="rounded-xl border border-border bg-secondary/30 p-3">
              <p className="font-medium">{fact.assertion === "absent" ? "明确否认：" : ""}{fact.entity}</p>
              <p className="mt-1 text-sm">{fact.details}</p>
              <p className="mt-1 break-words text-xs text-muted-foreground">来源：{fact.source.quote}</p>
              <p className="mt-1 text-xs text-muted-foreground">{fact.source.collected_at} · 用户自述</p>
              {removing === fact.id ? <div className="mt-2 flex gap-2">
                <Button size="sm" variant="outline" disabled={busy} onClick={() => setRemoving(null)}>取消</Button>
                <Button size="sm" variant="destructive" disabled={busy} onClick={() => void revoke(fact.id)}>确认撤销</Button>
              </div> : <Button className="mt-2" size="sm" variant="outline" onClick={() => setRemoving(fact.id)}>撤销长期保存</Button>}
            </div>)}
        </div>
      </DialogContent>
    </Dialog>
  </>;
}
