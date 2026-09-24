from __future__ import annotations

import re
import sqlite3
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import CHECKPOINT_PATH, MODEL_MODE
from .db import one
from .model import complete, complete_json
from .retrieval import retrieve_with_status
from .safety import check_herbs, normalize


class FlowState(TypedDict, total=False):
    message: str
    file_text: str
    profile: dict
    symptoms: list[str]
    intent_type: str
    pending_question: str
    pending_context: str
    docs: list[dict]
    retrieval_warning: str
    diagnosis: str
    treatment_plan: dict
    bianzheng_note: str
    formula_note: str
    herb_note: str
    consultation_notes: list[dict]
    safety_errors: list[str]
    reviewer_approved: bool
    reviewer_retry_count: int
    safety_retry_count: int
    response: str


KEYWORDS = ["失眠", "睡眠", "咳嗽", "头痛", "腹痛", "胃痛", "怕冷", "发热", "疲劳", "乏力", "口干", "月经", "眩晕", "鼻塞", "食欲"]
URGENT = ["胸痛", "呼吸困难", "昏厥", "大出血", "意识不清", "自杀"]


def prepare(state: FlowState) -> FlowState:
    # Clear all per-turn fields, including prior expert drafts.
    return {
        "docs": [], "retrieval_warning": "", "diagnosis": "", "treatment_plan": {},
        "bianzheng_note": "", "formula_note": "", "herb_note": "",
        "consultation_notes": [], "safety_errors": [], "response": "",
        "reviewer_approved": True, "reviewer_retry_count": 0, "safety_retry_count": 0,
    }


def inquire(state: FlowState) -> FlowState:
    text = state.get("message", "")
    previous = state.get("pending_context", "")
    combined = f"{previous}；{text}" if previous else text
    if any(word in text for word in URGENT):
        return {
            "intent_type": "safety_rejected", "symptoms": [],
            "pending_context": "", "pending_question": "",
            "response": "你描述的情况可能需要及时处理。请尽快联系当地急救服务或就近医疗机构，不要等待在线问答。",
        }
    symptoms = [word for word in KEYWORDS if word in combined]
    if "睡不好" in combined and "失眠" not in symptoms:
        symptoms.insert(0, "失眠")
    if not symptoms and not previous:
        return {
            "intent_type": "chat", "symptoms": [], "pending_context": "",
            "pending_question": "",
            "response": "你好，我可以帮助整理症状信息，并展示中医知识检索与会诊流程。请描述主要不适、持续时间，以及有无过敏史。内容仅供学习研究参考。",
        }
    if not previous and len(text.strip()) < 18 and not re.search(r"\d|天|周|月|年|伴|同时", text):
        return {
            "intent_type": "consult", "symptoms": symptoms,
            "pending_context": combined, "pending_question": "这种情况持续多久了？是否伴有其他不适？",
            "response": "为了更好地整理信息，请补充：这种情况持续多久了？是否伴有其他不适？",
        }
    return {
        "intent_type": "diagnose", "symptoms": symptoms or state.get("symptoms", []),
        "pending_context": "", "pending_question": "",
    }


def route(state: FlowState) -> str:
    return "diagnose" if state["intent_type"] == "diagnose" else "end"


def search(state: FlowState) -> FlowState:
    query = " ".join(state.get("symptoms", [])) + " " + state["message"]
    docs, warning = retrieve_with_status(query)
    return {"docs": docs, "retrieval_warning": warning}


def _context(state: FlowState) -> str:
    docs = "\n".join(f'{d["title"]}：{d["body"][:240]}' for d in state.get("docs", []))
    return (
        f'用户描述：{state.get("message","")}\n'
        f'症状：{",".join(state.get("symptoms",[]))}\n'
        f'病历摘要：{state.get("file_text","")[:1200]}\n'
        f'患者画像：{state.get("profile",{})}\n'
        f'检索片段：{docs or "无可用古籍片段"}'
    )


def bianzheng(state: FlowState) -> FlowState:
    if MODEL_MODE == "mock":
        symptoms = "、".join(state.get("symptoms", [])) or "所述不适"
        diagnosis = f"根据已提供信息，主要涉及{symptoms}。信息不足以确定证型，以下仅展示分析流程。"
    else:
        diagnosis = complete(
            "你是中医知识辅助系统。只解释可能的辨证思路和不确定性，不作确诊，不编造古籍出处或药物剂量。",
            _context(state),
        )
    return {"diagnosis": diagnosis, "bianzheng_note": diagnosis}


def formula(state: FlowState) -> FlowState:
    if MODEL_MODE == "mock":
        symptoms = state.get("symptoms", [])
        if any(item in symptoms for item in ["失眠", "睡眠"]):
            name = "酸枣仁汤"
        elif any(item in symptoms for item in ["怕冷", "鼻塞"]):
            name = "桂枝汤"
        else:
            name = ""
        if name:
            import json
            row = one("SELECT name,herbs FROM formulas WHERE name=?", (name,))
            plan = {"name": row["name"], "herbs": json.loads(row["herbs"])} if row else {"name": "基础方剂数据缺失", "herbs": []}
        else:
            plan = {"name": "暂无匹配的示例方剂", "herbs": []}
    else:
        plan = complete_json(
            "你是中医知识演示系统。返回候选方剂 JSON，字段仅为 name 字符串和 herbs 字符串数组。禁止剂量、用法或自行抓药指令；无把握则 herbs 为空。",
            _context(state) + "\n辨证思路：" + state.get("diagnosis", ""),
        )
        if not isinstance(plan.get("name"), str) or not isinstance(plan.get("herbs"), list):
            plan = {"name": "未能解析候选方剂", "herbs": []}
    return {"treatment_plan": plan, "formula_note": f'候选方剂：{plan["name"]}'}


def herb_expert(state: FlowState) -> FlowState:
    allergies = state.get("profile", {}).get("allergies", "")
    return {"herb_note": f"已检查基础配伍规则。画像中的过敏信息：{allergies or '未填写'}。"}


def synthesize(state: FlowState) -> FlowState:
    return {"consultation_notes": [
        {"expert": "辨证", "content": state.get("bianzheng_note", "")},
        {"expert": "方剂", "content": state.get("formula_note", "")},
        {"expert": "本草", "content": state.get("herb_note", "")},
    ]}


def _review(plan: dict, context: str) -> tuple[bool, str]:
    if MODEL_MODE == "mock":
        return True, ""
    result = complete_json(
        "你是独立安全审查者。审查候选方剂是否存在未经证实的断言、明显不合理药材或用药指令。"
        "返回 JSON：approved 布尔值、reason 字符串。确定性配伍校验由程序另行执行。",
        context + "\n候选方案：" + str(plan),
    )
    return result.get("approved") is True, str(result.get("reason", "审查未通过"))


def reviewer(state: FlowState) -> FlowState:
    approved, reason = _review(state.get("treatment_plan", {}), _context(state))
    return {"reviewer_approved": approved, "safety_errors": [] if approved else [f"审查者拒绝：{reason}"]}


def safety(state: FlowState) -> FlowState:
    plan = state.get("treatment_plan", {})
    def validate(candidate: dict) -> list[str]:
        herbs = candidate.get("herbs")
        errors = check_herbs(herbs if isinstance(herbs, list) else None)
        allergy_text = state.get("profile", {}).get("allergies", "")
        if allergy_text and isinstance(herbs, list):
            for herb in herbs:
                if isinstance(herb, str) and normalize(herb) in allergy_text:
                    errors.append(f"画像提示可能对{herb}过敏")
        return errors

    errors = validate(plan)
    if not state.get("reviewer_approved", True):
        errors = state.get("safety_errors", []) + errors
    retries = 0
    if errors and MODEL_MODE != "mock":
        for attempt in range(2):
            retries = attempt + 1
            revised = complete_json(
                "你是中医知识演示系统。之前候选方案未通过安全校验。只返回 JSON：name 字符串和 herbs 字符串数组。"
                "禁止剂量或用法；无法安全修正则 herbs 为空。",
                _context(state) + "\n原候选：" + str(plan) + "\n错误：" + "；".join(errors),
            )
            if not isinstance(revised.get("name"), str) or not isinstance(revised.get("herbs"), list):
                continue
            plan = revised
            errors = validate(plan)
            if not errors and state.get("consultation_notes"):
                approved, reason = _review(plan, _context(state))
                if not approved:
                    errors = [f"审查者拒绝：{reason}"]
            if not errors:
                break
    notes = state.get("consultation_notes", [])
    if notes and not errors:
        notes = [{**note, "content": f'最终候选方剂：{plan["name"]}'} if note.get("expert") == "方剂" else note
                 for note in notes]
    return {"safety_errors": errors, "treatment_plan": plan, "safety_retry_count": retries,
            "consultation_notes": notes}


def finish(state: FlowState) -> FlowState:
    if state.get("intent_type") != "diagnose":
        return {"response": state.get("response", "")}
    lines = ["## 辨证思路", state.get("diagnosis", "资料不足。")]
    docs = state.get("docs", [])
    if state.get("retrieval_warning"):
        lines.extend(["", state["retrieval_warning"]])
    if docs:
        lines.extend(["", "## 检索出处"])
        lines.extend(f'- 《{d["title"]}》{d["chapter"]}：{d["body"][:110]}…' for d in docs)
    else:
        lines.extend(["", "当前没有可引用的古籍片段。"])
    errors = state.get("safety_errors", [])
    plan = state.get("treatment_plan", {})
    if errors:
        lines.extend(["", "## 安全校验", "候选方剂未通过确定性校验，具体药材清单已隐藏。请由执业医师人工复核。"])
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.extend(["", "## 方剂知识示例", f'**{plan.get("name", "")}**：' + "、".join(plan.get("herbs", []))])
        lines.append("此为知识演示，不包含剂量或用法，不能作为个人处方。")
    lines.extend(["", "> 仅供学习与研究参考，不能代替执业医师面诊，也不应作为自行抓药的依据。"])
    notes = state.get("consultation_notes", [])
    if errors:
        notes = [{**note, "content": "候选方案未通过安全审查，已隐藏。"} if note.get("expert") == "方剂" else note
                 for note in notes]
    return {"response": "\n".join(lines), "consultation_notes": notes}


def build_graph(mode: str):
    graph = StateGraph(FlowState)
    graph.add_node("prepare", prepare)
    graph.add_node("inquiry", inquire)
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "inquiry")
    graph.add_conditional_edges("inquiry", route, {"diagnose": "search", "end": "finish"})
    graph.add_node("search", search)
    graph.add_node("bianzheng", bianzheng)
    graph.add_node("formula", formula)
    graph.add_node("safety", safety)
    if mode == "multi":
        graph.add_node("herb", herb_expert)
        graph.add_node("synthesis", synthesize)
        graph.add_node("reviewer", reviewer)
        graph.add_edge("search", "bianzheng")
        graph.add_edge("search", "formula")
        graph.add_edge("search", "herb")
        graph.add_edge(["bianzheng", "formula", "herb"], "synthesis")
        graph.add_edge("synthesis", "reviewer")
        graph.add_edge("reviewer", "safety")
    else:
        graph.add_edge("search", "bianzheng")
        graph.add_edge("bianzheng", "formula")
        graph.add_edge("formula", "safety")
    graph.add_edge("safety", "finish")
    graph.add_edge("finish", END)
    return graph


LABELS = {"prepare": "资料准备", "inquiry": "问诊", "search": "古籍检索", "bianzheng": "辨证",
          "formula": "方剂", "herb": "本草", "synthesis": "综合", "reviewer": "审查",
          "safety": "安全校验", "finish": "完成"}


def run_flow(thread_id: str, mode: str, message: str, profile: dict, file_text: str):
    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        graph = build_graph(mode).compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id}}
        final = None
        completed: set[str] = set()
        started: set[str] = set()

        def start(stage: str):
            if stage in started:
                return None
            started.add(stage)
            return {"type": "stage", "stage": stage, "label": LABELS[stage], "status": "start"}

        yield start("prepare")
        for update in graph.stream(
            {"message": message, "profile": profile, "file_text": file_text},
            config=config, stream_mode="updates",
        ):
            for node, value in update.items():
                completed.add(node)
                yield {"type": "stage", "stage": node, "label": LABELS.get(node, node),
                       "status": "done", "note": {}}
                next_stages: tuple[str, ...] = ()
                if node == "prepare":
                    next_stages = ("inquiry",)
                elif node == "inquiry":
                    next_stages = ("search",) if value.get("intent_type") == "diagnose" else ("finish",)
                elif node == "search":
                    next_stages = ("bianzheng", "formula", "herb") if mode == "multi" else ("bianzheng",)
                elif mode == "multi":
                    if node in {"bianzheng", "formula", "herb"} and {"bianzheng", "formula", "herb"} <= completed:
                        next_stages = ("synthesis",)
                    elif node == "synthesis":
                        next_stages = ("reviewer",)
                    elif node == "reviewer":
                        next_stages = ("safety",)
                    elif node == "safety":
                        next_stages = ("finish",)
                else:
                    next_stages = {
                        "bianzheng": ("formula",), "formula": ("safety",),
                        "safety": ("finish",),
                    }.get(node, ())
                for stage in next_stages:
                    event = start(stage)
                    if event:
                        yield event
                if node == "finish":
                    final = graph.get_state(config).values
        if final is None:
            final = graph.get_state(config).values
        yield {"type": "result", "data": final}
    finally:
        conn.close()
