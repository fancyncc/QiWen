import type { ReactNode } from "react";

type Herb = { name: string; dose?: string };
type Prescription = { name: string; herbs: Herb[] };

export function extractPrescription(content: string): Prescription | null {
  if (/药材清单已隐藏|未通过确定性校验|无法解析完整药材清单/.test(content)) return null;
  const json = content.match(/"herb_names"\s*:\s*(\[[^\]]*\])/);
  let names: unknown[] | null = null;
  if (json) {
    try { names = JSON.parse(json[1]); } catch { /* Continue with the text form. */ }
  }
  const section = content.split(/##\s*方剂知识示例/)[1] || "";
  const match = section.match(/\*\*([^*\n]+)\*\*\s*[：:]\s*([^\n]+)/);
  if (!names && match) names = match[2].split(/[、，,]/).map(item => item.trim());
  if (!names || names.length === 0) return null;
  const seen = new Set<string>();
  const herbs: Herb[] = [];
  for (const item of names) {
    const value = typeof item === "string" ? item : item && typeof item === "object" && "name" in item ? String(item.name) : "";
    const parsed = value.trim().match(/^([^\d\s]+?)(?:\s*(\d+(?:\.\d+)?\s*(?:克|g|G)))?$/);
    if (!parsed) continue;
    const name = parsed[1].trim();
    if (!name || seen.has(name)) continue;
    seen.add(name);
    herbs.push({ name, dose: parsed[2] });
  }
  return herbs.length ? { name: match?.[1] || "候选方剂", herbs } : null;
}

export default function PrescriptionCard({ content }: { content: string }): ReactNode {
  const prescription = extractPrescription(content);
  if (!prescription) return null;
  return <section className="prescription-card" aria-label="方剂知识示例">
    <div className="prescription-head"><span className="prescription-seal">方</span><div><small>方剂知识示例</small><strong>{prescription.name}</strong></div></div>
    <div className="herb-list">{prescription.herbs.map(herb => <span className="herb-tag" key={herb.name}>{herb.name}{herb.dose && <small>{herb.dose}</small>}</span>)}</div>
    <p>仅作知识展示，实际使用须经执业医师与药师审核。</p>
  </section>;
}
