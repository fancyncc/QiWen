"""Source-scoped clinical input shared by fixed and multi workflows."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .documents import model_excerpt
from .safety_context import applicable_facts
from . import symptoms
from .consultation_followup import answer_readings, scoped_answers, observation_ledger, inquiry_progress


class ContextBudgetError(RuntimeError):
    public_message = "本轮必要资料超过模型上下文容量，请分批上传病历或聚焦一个问题后重试。"


@dataclass
class ContextItem:
    id: str
    kind: str
    content: str
    required: bool = False


@dataclass
class ContextBundle:
    items: list[ContextItem] = field(default_factory=list)

    def render(self, budget: int = 10000) -> tuple[str, dict]:
        required = [item for item in self.items if item.required]
        optional = [item for item in self.items if not item.required]
        def block(item):
            return f"[{item.kind}；来源={item.id}]\n{item.content}"
        kept = list(required)
        size = sum(len(block(item)) + 1 for item in kept)
        if size > budget:
            raise ContextBudgetError()
        dropped = []
        for item in optional:
            cost = len(block(item)) + 1
            if size + cost <= budget:
                kept.append(item)
                size += cost
            else:
                dropped.append(item.id)
        return "\n".join(block(item) for item in kept), {
            "injected": [item.id for item in kept], "dropped": dropped,
            "drop_reason": "context_budget" if dropped else "", "chars": size,
        }


def clinical_bundle(state: dict) -> ContextBundle:
    safe = {"facts": [], "subject": "unknown", **state.get("safety_state", {})}
    # Legacy summary hints are never evidence, even if passed by an old caller.
    safe["facts"] = [f for f in safe["facts"] if f.get("source", {}).get("kind") != "summary"]
    subject = safe["subject"]
    scoped_symptoms = [f for f in state.get("symptom_facts", [])
                       if f.get("source", {}).get("kind") != "summary"
                       and (not state.get("thread_id") or f.get("source", {}).get("thread_id") in {None, "", state["thread_id"]})]
    facts = symptoms.effective(scoped_symptoms, subject)
    # Share source metadata once. Repeating the same message/date for every
    # symptom and inherited attribute can consume the entire model request.
    provenance, source_messages = [], {}
    for fact in facts:
        if fact.get("archived_episode"):
            continue
        source = fact.get("source", {})
        message_id = source.get("message_id", "")
        source_messages[message_id] = source.get("collected_at", "")
        refs = {}
        for field, ref in fact.get("evidence_refs", {}).items():
            source_messages[ref.get("message_id", "")] = ref.get("collected_at", "")
            if ref.get("fact_id") != fact["id"] or ref.get("message_id") != message_id:
                refs[field] = [ref.get("fact_id"), ref.get("message_id")]
        provenance.append({"id": fact["id"], "message_id": message_id, "evidence_refs": refs})
    items = [
        ContextItem(state.get("source_message_id") or "current", "本轮用户描述", state.get("message", ""), True),
        ContextItem("patient-scope", "咨询对象与病程", json.dumps({"subject": subject, "confirmed": safe.get("subject_confirmed", False),
                    "episode_id": state.get("episode_id", "")}, ensure_ascii=False), True),
        ContextItem("symptom-facts", "症状事实：保留时间、程度、变化和否认", symptoms.describe(facts, subject), True),
        ContextItem("symptom-sources", "症状来源引用：未单列字段继承同条事实的消息来源", json.dumps({"message_dates": source_messages, "facts": provenance},
                    ensure_ascii=False, separators=(",", ":")), True),
        ContextItem("safety-facts", "适用安全事实", json.dumps(applicable_facts(safe), ensure_ascii=False), True),
        ContextItem("profile", "患者画像：空白为未知，内容为用户自述", json.dumps(state.get("profile", {}), ensure_ascii=False), True),
        ContextItem("pending", "待回答问题", state.get("pending_question", ""), True),
    ]
    if state.get("answered_question"):
        items.append(ContextItem("answered-inquiry", "上一轮问题与本轮回答（问题不是症状事实）", json.dumps({
            "question": state["answered_question"], "answer": state.get("message", ""),
            "answer_message_id": state.get("source_message_id", ""),
        }, ensure_ascii=False), True))
    answers = scoped_answers(state)
    if answers:
        items.append(ContextItem("inquiry-answers", "已回答的问诊问题：保留未知，问题本身不是症状事实",
                                 json.dumps([{key: row[key] for key in ("question", "answer", "answer_message_id", "stage") if key in row}
                                             for row in answers], ensure_ascii=False), True))
        items.append(ContextItem("answer-readings", "用户回答释义：按问题逐项对应，均为自述，不代表医学核实；不得重复询问",
                                 json.dumps([{key: row[key] for key in ("question", "interpretation", "topics", "status", "answer_message_id")}
                                             for row in answer_readings(state)], ensure_ascii=False, separators=(",", ":")), True))
    observations = [{key: row[key] for key in ("topics", "status", "quote", "answer_message_id")}
                    for row in observation_ledger(state) if not row.get("question")]
    if observations:
        items.append(ContextItem("observations", "已答与主动报告的观察：未知和否认均需保留；问题不是阳性依据",
                                 json.dumps(observations, ensure_ascii=False, separators=(",", ":")), True))
    items.append(ContextItem("inquiry-progress", "本次就诊共享追问进度",
                             json.dumps(inquiry_progress(state), ensure_ascii=False), True))
    if state.get("analysis_only") or state.get("preliminary_analysis"):
        items.append(ContextItem("analysis-boundary", "本轮分析范围", json.dumps({
            "mode": "provisional_reference", "manual_review_reasons": state.get("manual_review_reasons", []),
            "instruction": "先追问辨证及选方所需的必要依据；追问后仍缺可选资料，可继续给出有依据的辨证、方剂和具体组成的处方参考，并逐项说明安全与剩余缺项。已明确的非草药过敏史保留待复核。不得确认个人用药适用性，所有候选剂量留空，不提供用法。"
        }, ensure_ascii=False), True))
    if state.get("file_text"):
        items.append(ContextItem("current-attachments", "病历相关原文摘录", model_excerpt(state["file_text"]), True))
    if state.get("pending_context") or state.get("episode_text"):
        items.append(ContextItem("episode-statements", "本次问诊原文补充", state.get("pending_context") or state["episode_text"]))
    from .classics import passage_excerpt
    for doc in state.get("docs", []):
        items.append(ContextItem(doc.get("id") or f"legacy-source:{doc['title']}/{doc['chapter']}", "公开古籍证据", f"{doc['title']} / {doc['chapter']}：{passage_excerpt(doc)}"))
    return ContextBundle(items)


def clinical_context(state: dict) -> str:
    text, details = clinical_bundle(state).render()
    from .model import invocation, current_node
    from .model import observe_input
    observe_input("clinical_context", {"text": text, **details})
    context = invocation.get()
    if context is not None:
        context.contexts.append({"node": current_node.get(), **details})
    return text


def project_observation(value, limit: int = 3500):
    """Keep valid JSON, IDs and risk fields; omit entire optional evidence rows."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    data = result.get("result")
    if not isinstance(data, dict):
        return result
    data = dict(data)
    # The raw observation remains in the owner-visible trace. Stage diagnostics
    # do not belong in the model's evidence and must not crowd out risk facts.
    data.pop("retrieval_trace", None)
    if isinstance(data.get("documents"), list):
        data["documents"] = [{key: row[key] for key in ("id", "title", "chapter", "excerpt") if key in row}
                             for row in data["documents"]]
    dropped = []
    for key in ("documents", "formulas"):
        if isinstance(data.get(key), list):
            accepted = []
            for row in data[key]:
                trial = {**data, key: [*accepted, row]}
                if len(json.dumps(trial, ensure_ascii=False)) <= limit:
                    accepted.append(row)
                else:
                    dropped.append(row.get("id", row.get("thread_id", row.get("name", key))))
            data[key] = accepted
    if dropped:
        data["omitted_evidence_ids"] = dropped
        data["omission_reason"] = "tool_result_budget"
    result["result"] = data
    return result
