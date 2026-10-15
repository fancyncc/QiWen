"use client";

import { Leaf, ShieldAlert } from "lucide-react";
import type { PrescriptionReference } from "@/lib/types";
import { publicWarnings } from "@/lib/consultation-presentation";

function quantity(dose: { min: number | null; max: number | null; unit: string }) {
  return dose.min === null || dose.max === null ? "待医师核定"
    : `${dose.min === dose.max ? dose.min : `${dose.min}–${dose.max}`} ${dose.unit}`;
}

export function ClinicalSafety({ warnings }: { warnings: string[] }) {
  const values = publicWarnings(warnings);
  if (!values.length) return null;
  const primary = values.filter((item) => /^(?:过敏风险|合并用药|特殊人群|已记录过敏|高风险药材)|十八反|十九畏|画像提示可能对|请勿/.test(item));
  const details = values.filter((item) => !primary.includes(item));
  return <section aria-label="安全说明" className="rounded-xl border border-destructive/25 bg-destructive/5 p-4 sm:p-5">
    <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-destructive"><ShieldAlert className="h-4 w-4" aria-hidden />安全说明</h3>
    <ul className="list-disc space-y-1.5 pl-5 text-sm leading-6">{(primary.length ? primary : details).map((item) => <li key={item}>{item}</li>)}</ul>
    {primary.length > 0 && details.length > 0 && <details className="mt-3 border-t border-destructive/15 pt-3">
      <summary className="cursor-pointer text-xs text-destructive">查看全部禁忌和待核对事项</summary>
      <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm leading-6">{details.map((item) => <li key={item}>{item}</li>)}</ul>
    </details>}
    <p className="mt-3 text-xs leading-6 text-muted-foreground">此为待医师复核的参考，不能据此自行抓药、服用或调整用药。</p>
  </section>;
}

export function ClinicalPrescription({ reference }: { reference: PrescriptionReference }) {
  if (!reference.herbs.length) return null;
  return <section aria-label="独立处方参考" className="overflow-hidden rounded-xl border border-primary/30 border-l-[3px] border-l-primary bg-secondary/20">
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-primary/15 px-4 py-3 sm:px-5">
      <h3 className="flex items-center gap-2 text-base font-semibold"><Leaf className="h-4 w-4 text-primary" aria-hidden />处方参考 · {reference.name}</h3>
      <span className="text-xs text-primary">待医师复核</span>
    </div>
    <div className="space-y-3 p-4 sm:p-5">
      <p className="text-xs leading-6 text-muted-foreground">{reference.dose_notice || "文献参考范围与本次拟用量分别展示，具体用量、口径及疗程需医师核定。"}</p>
      <div className="overflow-x-auto">
        <table aria-label="处方组成、参考范围与本次拟用量" className="w-full min-w-[36rem] text-left text-sm leading-6">
          <thead><tr className="border-b border-primary/20 text-muted-foreground">
            <th scope="col" className="px-2 py-2">药物／炮制</th><th scope="col" className="px-2 py-2">文献参考范围</th>
            <th scope="col" className="px-2 py-2">本次拟用量</th><th scope="col" className="px-2 py-2">药物作用与要求</th>
          </tr></thead>
          <tbody>{reference.herbs.map((item) => {
            const entry = item.reference_dose;
            const candidateVerified = item.verification_status === "verified";
            return <tr key={item.herb_id} className="border-b border-primary/10 align-top last:border-0">
              <th scope="row" className="px-2 py-3 font-medium">{item.name}<span className="block text-xs font-normal text-muted-foreground">{entry?.preparation || item.preparation || "炮制待核定"}</span></th>
              <td className="px-2 py-3 tabular-nums">{entry ? quantity(entry) : "暂无可核对记录"}
                {entry && <><span className="block max-w-48 text-xs text-muted-foreground">{entry.basis}</span>
                  <a className="text-xs text-primary underline underline-offset-2" href={entry.url} target="_blank" rel="noreferrer">剂量来源</a></>}
              </td>
              <td className="px-2 py-3 tabular-nums">{candidateVerified ? quantity(item.dosage) : "待医师核定"}
                {candidateVerified && <span className="block text-xs text-muted-foreground">已有剂量核验；每剂／每日及个人适用性仍待确认</span>}
              </td>
              <td className="max-w-64 px-2 py-3"><span>{item.effect || "配伍作用需结合选方说明核对"}</span>
                {(item.preparation || entry?.preparation) && <span className="block text-xs text-muted-foreground">来源炮制／用法要求：{item.preparation || entry?.preparation}</span>}
              </td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="text-xs leading-6 text-muted-foreground">方剂出处：{reference.source}</p>
      {reference.uncertainties.length > 0 && <details>
        <summary className="cursor-pointer text-xs text-primary">尚待确认的适用条件</summary>
        <ul className="mt-2 list-disc pl-5 text-sm leading-6">{reference.uncertainties.map((item) => <li key={item}>{item}</li>)}</ul>
      </details>}
    </div>
  </section>;
}
