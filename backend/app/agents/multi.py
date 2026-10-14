from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .runtime import AgentRun, run_agent_graph
from .tools import (CLASSICS_TOOL, COMPATIBILITY_TOOL, DOSAGE_TOOL, FORMULA_TOOL, HERB_TOOL,
                    PROFILE_TOOL, USER_RISK_TOOL)
from .. import model
from ..db import one
from ..context import clinical_context
from ..symptoms import duration_text
from ..prescription import CandidatePlan, migrate_plan, names


class BianzhengArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["completed", "need_user_input"]
    conclusion: str = Field(max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    question: str = Field(default="", max_length=500)
    syndrome: str = Field(default="", max_length=200)
    confidence: float | None = Field(default=None, ge=0, le=1)
    summary: str = Field(default="", max_length=1000)


class FormulaArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["completed", "need_user_input"]
    plan: CandidatePlan | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    question: str = Field(default="", max_length=500)
    comparison: str = Field(default="", max_length=2000)


class HerbReviewArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["approved", "revise", "clarify", "rejected"]
    problems: list[str] = Field(default_factory=list, max_length=30)
    required_actions: list[str] = Field(default_factory=list, max_length=30)
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    question: str = Field(default="", max_length=500)


def _query(state: dict) -> str:
    return state.get("search_query") or " ".join(state.get("symptoms", [])) or state.get("message", "")[:200]


def _traces(state: dict, agent: str, run: AgentRun) -> dict:
    return {**state.get("agent_traces", {}), agent: run.trace}


def _artifacts(state: dict, agent: str, value: dict) -> dict:
    return {**state.get("agent_artifacts", {}), agent: value}


def can_analyze_provisionally(state: dict) -> bool:
    safe = state.get("safety_state", {})
    return bool(safe.get("subject_confirmed") and state.get("symptoms")
                and not state.get("symptom_pending") and state.get("preflight", {}).get("decision") in {"allow", "clarify"}
                and not state.get("preflight", {}).get("question_keys")
                and (duration_text(state.get("episode_text", "")) or any(
                    f.get("duration_text") and f.get("subject") == safe.get("subject")
                    and f.get("time_scope") == "current" and not f.get("superseded_by") and not f.get("archived_episode")
                    for f in state.get("symptom_facts", []))))


def provisional_updates(state: dict, artifact: dict) -> dict:
    reason = "四诊资料尚不完整，本轮仅作初步辨证及文献知识对照，不能确定个人处方"
    questions = artifact.get("uncertainties", []) or ([artifact["question"]] if artifact.get("question") else [])
    return {"analysis_only": True, "preliminary_analysis": True,
            "manual_review_reasons": list(dict.fromkeys([*state.get("manual_review_reasons", []), reason])),
            "analysis_questions": list(dict.fromkeys([*state.get("analysis_questions", []), *questions]))[:3]}


def _mock_diagnosis(state: dict) -> dict:
    symptoms = "、".join(state.get("symptoms", [])) or "所述不适"
    denied = "、".join(state.get("denied_symptoms", []))
    return {"status": "completed",
            "conclusion": (f"根据已提供信息，用户描述主要涉及{symptoms}。"
                           + (f"用户表示没有{denied}。" if denied else "")
                           + "现有信息不足以确定证型，以下仅展示多 Agent 分析流程。"),
            "evidence_ids": [row["id"] for row in state.get("docs", [])],
            "uncertainties": ["缺少面诊和完整临床资料"], "question": ""}


def _classics_from_run(run: AgentRun) -> tuple[list[dict], dict]:
    documents, warnings, statuses, attempts = [], [], [], []
    for observation in run.observations:
        if observation.get("tool") != "search_classics":
            continue
        if observation.get("status") != "success":
            statuses.append("dependency_error")
            warnings.append("古籍工具未完成，本轮没有取得该次检索证据。")
            attempts.append({"outcome": "dependency_error", "error_type": observation.get("error", "ToolFailure")})
            continue
        result = observation.get("result", {})
        statuses.append(result.get("status", "unknown"))
        if result.get("retrieval_trace"):
            attempts.append(result["retrieval_trace"])
        if result.get("warning"):
            warnings.append(result["warning"])
        for item in result.get("documents", []):
            row = one("SELECT id,title,chapter,body,metadata FROM corpus WHERE id=?", (item.get("id"),))
            if row and item.get("excerpt") and item["excerpt"] in row["body"]:
                row["retrieval"] = {**item.get("retrieval", {}), "evidence": item["excerpt"],
                                    "admitted_by": item.get("retrieval", {}).get("admitted_by", "agent_tool")}
                documents.append(row)
    unique = {row["id"]: row for row in documents}
    documents = list(unique.values())
    trace = {"requested_mode": "agent", "active_mode": "agent_tool",
             "outcome": "success" if documents else statuses[-1] if statuses else "not_called",
             "attempts": attempts,
             "warning": " ".join(dict.fromkeys(warnings)),
             "rerank": {"status": statuses[-1] if statuses else "not_called"},
             "selected": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"]}
                          for row in documents]}
    for source, row in zip(trace["selected"], documents):
        source["excerpt"] = row["retrieval"]["evidence"]
    return documents, trace


async def bianzheng_agent(state: dict) -> dict:
    query = _query(state)
    final = _mock_diagnosis(state)

    def planner(step: int, observations: list[dict]) -> dict:
        if step == 1:
            return {"tool": "get_patient_profile", "arguments": {}}
        if step == 2:
            return {"tool": "search_classics", "arguments": {"query": query}}
        completed = {**final, "evidence_ids": [item["id"] for observation in observations
                                                if observation.get("tool") == "search_classics"
                                                for item in observation.get("result", {}).get("documents", [])]}
        return {"final": completed}

    task = json.dumps({
        "goal": ("本轮仅整理辨证假设与不确定性，不拟定个人用药。已明确的非草药过敏保留待复核，不单独阻断辨证知识分析；必要症状资料不足仍正常追问。"
                 if state.get("analysis_only") else "形成有证据支持、明确不确定性的辨证假设；资料不足时请求用户补充。"),
        "current_message": state.get("message", ""), "symptoms": state.get("symptoms", []),
        "denied_symptoms": state.get("denied_symptoms", []), "retrieval_query": query,
        "patient_context": clinical_context(state),
        "completion_rule": "主诉和病程已明确时，给出初步鉴别方向：逐项列已有支持、尚缺的区分依据及不确定性，不作确诊。缺舌象、脉象或古籍未命中应放入 uncertainties，不能单独作为 need_user_input 的理由。不得把尚未询问的症状当作明确否认；不要仅列一长串追问。最多保留三个最关键的待补充项。",
        "display_summary": "syndrome 填写初步证候方向，无法判断则明确待辨；summary 用约200字概括用户可核对的主要症状依据与不确定性，不输出内部思维链或工具日志。confidence 仅为对该初步判断的模型自评，范围0到1，不代表经过校准的诊断准确率；无法合理评估时填null，不得固定给高置信度。",
    }, ensure_ascii=False)
    run = await run_agent_graph(name="辨证 Agent", role="bianzheng", task=task, state=state,
                                tools=[PROFILE_TOOL, CLASSICS_TOOL],
                                output_schema=BianzhengArtifact, mock_planner=planner, max_steps=6)
    artifact = run.output
    agent_docs, retrieval_trace = _classics_from_run(run)
    merged_docs = {row["id"]: row for row in [*state.get("docs", []), *agent_docs]}
    search_count = sum(item.get("tool") == "search_classics" for item in run.observations)
    retrieval_updates = {
        "docs": list(merged_docs.values()), "search_required": search_count > 0,
        "search_attempts": state.get("search_attempts", 0) + search_count,
        "retrieval_trace": retrieval_trace,
        "retrieval_history": [*state.get("retrieval_history", []), retrieval_trace] if search_count else state.get("retrieval_history", []),
        "retrieval_warning": retrieval_trace["warning"],
    }
    context = model.invocation.get()
    if context and context.emit and artifact.get("conclusion"):
        context.emit("candidate.delta", {"channel": "body", "text": artifact["conclusion"]})
    preliminary = artifact["status"] == "need_user_input" and can_analyze_provisionally(state)
    if artifact["status"] == "need_user_input" and not preliminary:
        question = artifact["question"] or "请补充主要不适的持续时间和伴随表现。"
        return {**retrieval_updates, "agent_stop": "need_user_input", "intent_type": "consult", "response": question,
                "pending_question": question, "pending_context": state.get("episode_text", ""),
                "agent_traces": _traces(state, "bianzheng", run),
                "agent_artifacts": _artifacts(state, "bianzheng", artifact)}
    return {**retrieval_updates, "agent_stop": "", "diagnosis": artifact["conclusion"], "bianzheng_note": artifact["conclusion"],
            **(provisional_updates(state, artifact) if preliminary else {}),
            "agent_traces": _traces(state, "bianzheng", run),
            "agent_artifacts": _artifacts(state, "bianzheng", artifact)}


def bianzheng_route(state: dict) -> str:
    return "stop" if state.get("agent_stop") == "need_user_input" else "continue"


def _mock_plan(state: dict, revision: bool) -> dict:
    if revision:
        return {"name": "安全审查后未提供候选方剂", "herbs": []}
    symptoms = state.get("symptoms", [])
    name = ("酸枣仁汤" if any(item in symptoms for item in ("失眠", "睡眠"))
            else "桂枝汤" if any(item in symptoms for item in ("怕冷", "鼻塞")) else "")
    if not name:
        return {"name": "暂无匹配的示例方剂", "herbs": []}
    row = one("SELECT name,structured_herbs FROM formulas WHERE name=?", (name,))
    return migrate_plan({"name": row["name"], "herbs": json.loads(row["structured_herbs"])} if row
                        else {"name": "基础方剂数据缺失", "herbs": []})


async def formula_agent(state: dict) -> dict:
    revision = bool(state.get("safety_errors"))
    mock_plan = _mock_plan(state, revision)
    formula_query = ("睡眠" if any(item in state.get("symptoms", []) for item in ("失眠", "睡眠"))
                     else "外感" if any(item in state.get("symptoms", []) for item in ("怕冷", "鼻塞"))
                     else state.get("diagnosis", "")[:100] or _query(state))
    herb_names = names(mock_plan)

    def planner(step: int, _observations: list[dict]) -> dict:
        if step == 1:
            return {"tool": "search_formulas", "arguments": {"query": mock_plan["name"] if herb_names else formula_query}}
        if not revision and 2 <= step < 2 + min(2, len(herb_names)):
            return {"tool": "lookup_herb", "arguments": {"name": herb_names[step - 2]}}
        if not revision and step == 2 + min(2, len(herb_names)) and herb_names:
            return {"tool": "check_herb_compatibility", "arguments": {"herbs": herb_names}}
        return {"final": {"status": "completed", "plan": mock_plan,
                           "evidence_ids": [source for herb in mock_plan.get("herbs", [])
                                            for source in herb.get("source_ids", [])],
                           "uncertainties": (["候选剂量仍需人工复核"] if herb_names else []), "question": ""}}

    task = json.dumps({
        "goal": ("仅检索可溯源的方剂文献条目作知识对照，形成结构化知识候选，剂量全部留空，不确认个人适用性。已明确的非草药过敏保留待医师复核，不阻断文献检索。无文献匹配时如实说明，不编造条目。"
                 if state.get("analysis_only") else "根据辨证产物形成结构化候选方剂。先查询方剂和药材，缺少剂量证据时必须留空，不得猜测。"),
        "diagnosis": state.get("diagnosis", ""), "symptoms": state.get("symptoms", []),
        "profile": state.get("profile", {}), "review_feedback": state.get("safety_errors", []),
        "patient_context": clinical_context(state),
        "available_dosage_evidence": "需要时调用 get_dosage_evidence，不得从模型常识补充",
        "completion_rule": "本轮若仅作初步知识分析，仍须实际检索方剂条目，在 comparison 说明条目记载的适应描述、与已知症状的重合和缺失依据。不得把文献候选说成个人选方。舌脉缺失不阻断文献对照；无匹配应说明证据不足，不编造方剂。",
    }, ensure_ascii=False)
    run = await run_agent_graph(name="方剂 Agent", role="formula", task=task, state=state,
                                tools=[FORMULA_TOOL, HERB_TOOL, DOSAGE_TOOL, COMPATIBILITY_TOOL],
                                output_schema=FormulaArtifact, mock_planner=planner, max_steps=8)
    artifact = run.output
    incomplete = artifact["status"] == "need_user_input" or not artifact.get("plan")
    preliminary = incomplete and can_analyze_provisionally(state)
    if incomplete and not preliminary:
        question = artifact["question"] or "候选方剂所需资料不足，请补充当前用药和过敏情况。"
        return {"agent_stop": "need_user_input", "intent_type": "consult", "response": question,
                "pending_question": question, "pending_context": state.get("episode_text", ""),
                "agent_traces": _traces(state, "formula", run),
                "agent_artifacts": _artifacts(state, "formula", artifact)}
    plan = migrate_plan(artifact.get("plan") or {"name": "本轮未形成个人方剂候选", "herbs": []})
    for herb in plan.get("herbs", []):
        herb["verification_status"] = "unverified"
        if state.get("analysis_only") or preliminary:
            herb["dosage"].update(min=None, max=None)
    artifact = {**artifact, "plan": plan}
    return {"agent_stop": "", "treatment_plan": plan, "formula_note": f'候选方剂：{plan["name"]}',
            **(provisional_updates(state, artifact) if preliminary else {}),
            "safety_errors": [], "reviewer_approved": True,
            "safety_retry_count": state.get("safety_retry_count", 0) + (1 if revision else 0),
            "agent_traces": _traces(state, "formula", run),
            "agent_artifacts": _artifacts(state, "formula", artifact)}


def formula_route(state: dict) -> str:
    return "stop" if state.get("agent_stop") == "need_user_input" else "continue"


async def herb_agent(state: dict) -> dict:
    plan = state.get("treatment_plan", {})
    herb_names = names(plan)
    if not herb_names and state.get("analysis_only"):
        for observation in state.get("agent_traces", {}).get("formula", []):
            if observation.get("tool") == "search_formulas" and observation.get("status") == "success":
                entries = observation.get("result", {}).get("formulas", [])
                if entries:
                    herb_names = [herb["name"] for herb in entries[0].get("herbs", []) if isinstance(herb, dict) and herb.get("name")][:5]
                    if herb_names:
                        break

    def planner(step: int, observations: list[dict]) -> dict:
        if step == 1:
            return {"tool": "get_patient_profile", "arguments": {}}
        if herb_names and step == 2:
            return {"tool": "check_herb_compatibility", "arguments": {"herbs": herb_names}}
        if herb_names and step == 3:
            return {"tool": "check_user_risks", "arguments": {"herbs": herb_names}}
        dosage_index = step - 4
        if 0 <= dosage_index < len(herb_names):
            return {"tool": "lookup_herb" if state.get("analysis_only") else "get_dosage_evidence",
                    "arguments": {"name": herb_names[dosage_index]}}
        problems = []
        evidence_ids = []
        for observation in observations:
            result = observation.get("result", {})
            if isinstance(result, dict):
                problems.extend(result.get("violations", []))
                problems.extend(f"画像提示可能对{name}过敏" for name in result.get("allergy_risks", []))
                problems.extend(f"画像存在需人工复核的特殊风险：{name}" for name in result.get("special_risks", []))
                evidence_ids.extend(item["id"] for item in result.get("evidence", []) if item.get("id"))
        status = "revise" if problems else "approved"
        return {"final": {"status": status, "problems": list(dict.fromkeys(problems)),
                           "required_actions": (["修改候选后重新进行本草审查"] if problems else []),
                           "evidence_ids": list(dict.fromkeys(evidence_ids)), "question": ""}}

    task = json.dumps({
        "goal": ("核对已检索条目的药材资料与已收录配伍规则，非草药过敏未覆盖项列入 problems 和 required_actions 待人工复核。区分知识条目核对与个人用药批准；本轮不提供个人用药安全承诺，不因已知非草药过敏停止药材知识核对。"
                 if state.get("analysis_only") else "独立审查候选药材与当前画像、配伍和剂量证据。未知药物相互作用必须标记人工复核。"),
        "candidate": plan, "profile_summary": state.get("profile", {}),
        "patient_context": clinical_context(state),
        "reference_herbs": herb_names,
        "completion_rule": "知识分析时应实际调用 lookup_herb 查询至少两味已有文献条目的药材（若有），核对性质、功效和来源后再结束。reference_herbs 只表示文献药材，不代表个人处方。不要只输出审查通过或需复核；不得编造未查到的本草内容。",
    }, ensure_ascii=False)
    run = await run_agent_graph(name="本草 Agent", role="herb", task=task, state=state,
                                tools=[PROFILE_TOOL, HERB_TOOL, DOSAGE_TOOL, COMPATIBILITY_TOOL, USER_RISK_TOOL],
                                output_schema=HerbReviewArtifact, mock_planner=planner,
                                max_steps=min(12, 5 + len(herb_names)))
    artifact = run.output
    note = ("本草 Agent 审查通过；未知风险仍需医师或药师复核。" if artifact["status"] == "approved"
            else "本草 Agent 发现问题：" + "；".join(artifact["problems"]))
    return {"herb_review": artifact, "herb_note": note,
            "agent_traces": _traces(state, "herb", run),
            "agent_artifacts": _artifacts(state, "herb", artifact)}


def synthesize_agents(state: dict) -> dict:
    return {"consultation_notes": [
        {"expert": "辨证", "content": state.get("bianzheng_note", "")},
        {"expert": "方剂", "content": state.get("formula_note", "")},
        {"expert": "本草", "content": state.get("herb_note", "")},
    ]}


async def review_agents(state: dict) -> dict:
    from ..workflow import _context, _review

    approved, reason = await _review(state.get("treatment_plan", {}), _context(state))
    herb_review = state.get("herb_review", {})
    if herb_review.get("status") != "approved":
        approved = False
        reason = "；".join(herb_review.get("problems", [])) or "本草 Agent 未批准候选"
    from ..prescription import fingerprint
    return {"reviewer_approved": approved,
            "reviewer_hash": fingerprint(state.get("treatment_plan", {})),
            "safety_errors": [] if approved else [reason[:500] or "独立审查未通过"]}
