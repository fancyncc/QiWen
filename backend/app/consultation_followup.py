"""Shared distinction between patient questions and unavailable knowledge evidence."""
import json
import re
from typing import Literal

from pydantic import Field, create_model


InformationGap = Literal["none", "patient", "optional", "evidence"]
FOLLOWUP_RULE = (
    "information_gap 区分资料缺口：none=无需补充；patient=用户可回答且继续分析必需的信息；"
    "optional=可选舌脉等资料，允许保留未知并初步分析；evidence=文献、药材或剂量核验依据缺失，"
    "应继续使用工具或明确受限，不能要求用户替系统补证据。"
    "patient 时必须提出最多三个具体、尚未回答的问题；这会暂停会诊并等待下一轮回答，"
    "不能用已存在主诉和病程为由忽略必要问题。optional 也可以询问尚未回答、用户可描述且有助区分证候的项目，"
    "有具体问题时同样等待回答，同时保留已有依据的初步分析；不能只写资料不足就结束。"
    "仅缺用户明确不清楚、无法提供的舌脉等资料，或只有 evidence 缺口时不循环追问，局限写入分析或不确定性字段。"
    "信息不完整不等于无法分析：conclusion 先列用户实际报告的支持、否认或不支持的依据，"
    "再给有依据的初步方向及鉴别局限；证据不能支持具体证型时说明能判断到的范围，禁止编造症状或硬定证型。"
    "已回答的问题不重复询问；结合上下文中的上一轮问题理解简短回答。多个问题仅回答‘没有’不能推定全部阴性。"
    "‘上述情况都存在/都有’须逐项关联上一轮的是非问题；‘都没有’才表示全部否认。"
    "‘偏好凉性食物’只表示饮食偏好，不能改写成喜温、据此否认怕冷，或单独作为某个寒热证候的支持或反证。"
    "正常舌质、无齿痕、小便正常等是用户自述，不能改写成淡胖、齿痕或小便清长；"
    "已提供的舌质或舌形不能笼统写成‘缺乏舌象’，若缺舌苔等其他属性须具体说明。"
    "优先读取用户回答释义与原文；已答项不得换个说法再次询问，确有矛盾时只澄清冲突，不能重发整份问卷。"
    "明确不清楚的可选资料保留未知，不循环追问；已确认的过敏史保留，不重复索要；不得把提问内容当作阳性症状。"
    "读取上下文中的追问进度：同一就诊三个专家共用最多两轮常规追问，每轮最多三项。"
    "第一轮集中询问真正影响鉴别和选方的项目；已答一轮后，可选面色舌脉写入 uncertainties，不再暂停。"
    "第二轮仅限影响候选选择的必要新问题，followup_reason 必须说明该答案如何改变鉴别或候选选择。"
    "达到预算后用现有依据完成带局限的分析，未知仍为未知，不得把普通资料缺项改写成安全问题以继续追问。"
)

TREATMENT_RULE = (
    "先核对辨证和候选方剂实际依赖的临床依据；尚未回答且会影响选择的必要资料在本阶段具体追问，"
    "不要先跳过这些问题直接出方案。追问后仍不完整、用户不清楚的非关键资料保留未知，"
    "结合已知支持与反证继续给出有来源的辨证、方剂和处方参考（具体药材组成），不应自动降为只有方名的文献列表。"
    "在 comparison 或 rationale 说明为什么考虑该候选、哪些条件仍未确认；安全问题逐项说明，"
    "包括过敏、合并用药与相互作用、特殊人群、配伍和剂量核验状态，不能只写一句仅供参考。"
    "剂量只能采用实际检索的结构化证据；未知或未核验的剂量留空，不从常识猜测。"
    "有明确配伍冲突、相关药材过敏或其他禁忌时不推荐该候选；必要安全信息仍无法确认时不得声称个人用药安全或直接服用。"
)


def needs_patient_answer(artifact: dict) -> bool:
    if artifact.get("information_gap", "none") == "evidence":
        return False
    # Explicit questions override an accidentally completed status, without
    # guessing workflow state from the analysis prose.
    return bool(artifact.get("information_gap") == "patient" or artifact.get("question", "").strip()
                or any(value.strip() for value in artifact.get("questions", []))
                or (artifact.get("information_gap") != "optional" and artifact.get("status") in {"need_user_input", "clarify"}))


def followup_response(state: dict, artifact: dict, *, assessment: bool = False) -> str:
    """Keep useful preliminary analysis visible while saving only the question."""
    from .safety import contains_dosing

    analysis = (artifact.get("conclusion", "") if assessment else state.get("diagnosis", "")).strip()
    question = question_text(artifact)
    if not analysis or contains_dosing(analysis):
        return question
    return "## 初步辨证（待补充确认）\n\n" + analysis + "\n\n## 请继续补充\n\n" + question


def question_text(artifact: dict) -> str:
    questions = [value.strip() for value in artifact.get("questions", []) if value.strip()]
    return ("\n".join(f"{index + 1}. {value}" for index, value in enumerate(questions))
            or artifact.get("question", "").strip()
            or "请补充本次不适的具体表现、持续时间或近期变化；已提供的内容无需重复。")


def scoped_answers(state: dict) -> list[dict]:
    """Question bindings never cross a patient, thread, or episode boundary."""
    subject = state.get("safety_state", {}).get("subject")
    return [row for row in state.get("inquiry_answers", [])
            if row.get("thread_id") == state.get("thread_id") and row.get("subject") == subject
            and row.get("episode_id") == state.get("episode_id") and row.get("answer_message_id")]


def question_items(text: str) -> list[str]:
    parts = re.split(r"(?:^|\s|(?<=[？?。；;：:]))[（(]?[一二三四五六七八九十\d]+[）).、．]\s*|(?<=[？?])\s*", text)
    return [part.strip() for part in parts if part.strip() and not re.fullmatch(r".{0,25}[：:]", part.strip())]


# These keys identify observations, never a syndrome or a diagnosis. Narrow
# attributes prevent an answer about urine colour from resolving urine volume.
TOPICS = {
    "fatigue": r"(?:精神)?(?:疲乏|疲倦|疲劳)|乏力|精神是否疲(?:乏|倦)",
    "mood": r"心情|情绪|抑郁|低落",
    "face_colour": r"面色|脸色",
    "tongue_shape": r"淡胖|齿痕|舌体.*(?:胖|大)|舌质正常",
    "tongue_colour": r"舌(?:质|头|体).{0,5}(?:颜色|淡白|偏白|红|正常)",
    "tongue_coating": r"舌苔|苔(?:色|厚|薄|白|黄|腻)",
    "coating_colour": r"(?:舌苔|苔).{0,6}(?:颜色|泛白|白|黄|色)|苔色",
    "coating_thickness": r"(?:舌苔|苔).{0,6}(?:厚|薄)",
    "coating_texture": r"(?:舌苔|苔).{0,6}(?:腻|干|燥|润|剥)",
    "pulse": r"脉象|脉.{0,4}(?:沉迟|无力)",
    "food_preference": r"(?:偏好|喜欢|喜食|爱吃|喜[冷热]|饮食偏好).{0,8}(?:食物|温热|凉性|生冷)|饮食偏好",
    "waist_knees": r"腰膝酸软|腰膝酸痛",
    "urine_clear_long": r"小便清长|小便.{0,5}(?:是否)?清长",
    "urine_colour": r"小便.{0,5}(?:颜色|偏黄|清黄)|尿色",
    "urine_frequency": r"小便.{0,5}(?:次数|频繁)|尿频",
    "appetite": r"食欲|胃口|纳差",
    "thirst": r"口渴|口干|饮水",
    "sleep": r"失眠|睡眠|多梦|入睡|易醒",
    "stool": r"大便|便稀|便溏|腹泻|便秘",
    "abdominal": r"腹胀|肚子胀|腹痛|肚子痛",
    "cold_limbs": r"四肢(?:不温|发凉|冰凉)|手脚(?:发凉|冰凉)",
}


def topics(text: str) -> set[str]:
    keys = {key for key, pattern in TOPICS.items() if re.search(pattern, text)}
    if "urine_clear_long" in keys and not re.search(r"偏黄|红|血|深黄", text):
        keys.discard("urine_colour")
    return keys


def fully_recognized_question(text: str) -> bool:
    """A known word in a compound question cannot hide a new, unknown item."""
    remainder = text
    for pattern in TOPICS.values():
        remainder = re.sub(pattern, "", remainder)
    remainder = re.sub(r"温热食物|凉性食物|生冷食物|舌质|舌体|舌头|小便|颜色|偏好|饮食|温热|凉性|食物", "", remainder)
    remainder = re.sub(r"请观察|请问|请补充|是否|有没有|有无|感到|感觉|存在|情况|还是|或者|喜欢|偏向|以及|伴有|目前|平时|容易|经常|是什么样|什么样|如何|怎样|颜色|苍白|萎黄|正常|偏白|泛白|偏|色", "", remainder)
    return not re.sub(r"[\s\d.、？?，,。；;：:（）()]|[您你的有为是或和与及吗呢也且]", "", remainder)


def answer_readings(state: dict) -> list[dict]:
    """Explicit distributive replies are bound to yes/no items only.

    Keep original words and provenance. Bare negation across several questions,
    alternative choices and ambiguous replies stay unresolved rather than being
    manufactured into patient findings.
    """
    readings = []
    for row in scoped_answers(state):
        items = question_items(row["question"])
        answer = row["answer"].strip()
        all_yes = bool(re.fullmatch(r"(?:上述|以上|这些|这几项)?(?:情况|症状|问题)?(?:都|全部|全)(?:有|存在|是|符合)(?:的)?[。！! ]*", answer))
        all_no = bool(re.fullmatch(r"(?:上述|以上|这些|这几项)?(?:情况|症状|问题)?(?:都|全部|全)(?:没有|无|不存在|不是)[。！! ]*", answer))
        all_unknown = bool(re.fullmatch(r"(?:上述|以上|这些)?(?:情况|问题)?(?:都|全部)?(?:不知道|不清楚|不了解|无法提供)[。！! ]*", answer))
        clauses = [clause.strip() for clause in re.split(r"[，,。；;\n]|但是|但", answer) if clause.strip()]
        for question in items:
            keys = topics(question)
            resolved_keys = set(keys)
            choice = bool(re.search(r"还是|或是|如何|怎样|什么颜色", question))
            yes_no = bool(re.search(r"是否|有无|有没有|会不会", question)) or (question.endswith(("？", "?")) and not choice)
            meaning = ""
            if (all_yes or all_no) and yes_no and not choice:
                meaning = "用户明确表示该项存在" if all_yes else "用户明确表示该项不存在"
            elif all_unknown:
                meaning = "用户表示不清楚，保留未知，不当作阳性或阴性"
            else:
                matched = [clause for clause in clauses if topics(clause) & keys]
                if matched:
                    meaning = "用户自述：" + "；".join(matched)
                    resolved_keys = {key for clause in matched for key in topics(clause)}
                elif len(items) == 1 and re.fullmatch(r"(?:是的|是|有|没有|不是)[。！! ]*", answer) and not choice:
                    meaning = "用户对本项回答：" + answer
            if meaning:
                status = ("unknown" if all_unknown else "absent" if all_no else "present" if all_yes
                          else "absent" if re.fullmatch(r"没有|不是[。！! ]*", answer) else "reported")
                if re.fullmatch(r"(?:有|是|是的)[。！! ]*", answer) and re.search(r"或|和|及", question):
                    status = "partial"  # A yes to alternatives does not choose a colour.
                readings.append({"question": question, "answer": answer, "interpretation": meaning,
                                 "topics": sorted(resolved_keys), "status": status,
                                 "answer_message_id": row["answer_message_id"]})
    return readings


def observation_records(state: dict, text: str, message_id: str) -> list[dict]:
    """Store only scoped user statements, never the assistant's question as a finding."""
    rows = []
    from .safety_context import subject_of
    subject = state.get("safety_state", {}).get("subject")
    for clause in re.split(r"[，,。；;\n]|但是|但", text):
        owner = subject_of(clause)
        if owner not in {"unknown", subject} or re.search(r"假如|假设|如果|举例|以前|曾经|小时候|去年", clause):
            continue
        keys = topics(clause)
        if not keys:
            continue
        status = "unknown" if re.search(r"不清楚|不知道|无法提供|不了解", clause) else "reported"
        rows.append({"topics": sorted(keys), "status": status, "quote": clause.strip()[:500],
                     "answer_message_id": message_id, "thread_id": state.get("thread_id"),
                     "episode_id": state.get("episode_id"), "subject": state.get("safety_state", {}).get("subject")})
    return rows


def observation_ledger(state: dict) -> list[dict]:
    subject = state.get("safety_state", {}).get("subject")
    rows = [row for row in state.get("inquiry_observations", [])
            if row.get("thread_id") == state.get("thread_id") and row.get("episode_id") == state.get("episode_id")
            and row.get("subject") == subject and row.get("answer_message_id")]
    # Include current spontaneous observations even before persistence.
    rows += observation_records(state, state.get("message", ""), state.get("source_message_id", ""))
    for row in answer_readings(state):
        rows.append({**row, "quote": row["interpretation"]})
    seen, result = set(), []
    for row in rows:
        key = (row.get("answer_message_id"), tuple(row["topics"]), row["quote"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


MAX_INQUIRY_ROUNDS = 2
SAFETY_QUESTION = re.compile(r"过敏|用药|服药|药物|孕|哺乳|年龄|性别|\d+岁|儿童|胸痛|呼吸|昏|血尿|黑便|便血|自杀|剂量")


def inquiry_progress(state: dict) -> dict:
    # Legacy rows have no stage; count each real answer message once. The
    # intake question is separate from the shared specialist budget.
    messages = {row["answer_message_id"] for row in scoped_answers(state) if row.get("stage") != "inquiry"}
    rounds = len(messages)
    return {"answered_rounds": rounds, "maximum_routine_rounds": MAX_INQUIRY_ROUNDS,
            "optional_questions_closed": rounds >= 1,
            "routine_questions_closed": rounds >= MAX_INQUIRY_ROUNDS,
            "instruction": "已答项不得重复；第二轮仅问影响候选选择的必要新项目并说明原因；预算耗尽保留缺项完成受限分析，安全门禁独立执行。"}


def can_converge(state: dict, questions: list[str]) -> bool:
    safe, preflight = state.get("safety_state", {}), state.get("preflight", {})
    return bool(safe.get("subject_confirmed") and state.get("symptoms") and not state.get("symptom_pending")
                and preflight.get("decision") in {"allow", "clarify"} and not preflight.get("question_keys")
                and not any(SAFETY_QUESTION.search(item) for item in questions))


def convergence_reason(state: dict, artifact: dict) -> str:
    if not needs_patient_answer(artifact):
        return ""
    questions = question_items(question_text(artifact))
    if not can_converge(state, questions):
        return ""
    remaining, repeated = repeated_questions(state, artifact)
    progress = inquiry_progress(state)
    if repeated and not remaining:
        return "仅重复已答或已说明未知的观察项"
    if progress["routine_questions_closed"]:
        return "已完成两轮常规追问"
    if progress["optional_questions_closed"] and artifact.get("information_gap") == "optional":
        return "已完成必要追问，剩余可选项保留未知"
    if progress["answered_rounds"] >= 1 and not artifact.get("followup_reason", "").strip():
        return "追加追问未说明如何改变鉴别或候选选择"
    return ""


def after_inquiry_schema(schema, *, role: str):
    """Require a NEW provisional artifact; an old questionnaire is never promoted."""
    fields = {"information_gap": (Literal["optional", "evidence"], Field(...))}
    if "question" in schema.model_fields:
        fields["question"] = (Literal[""], "")
    if "questions" in schema.model_fields:
        fields["questions"] = (list[str], Field(default_factory=list, max_length=0))
    if "status" in schema.model_fields:
        fields["status"] = ((Literal["approved", "revise", "rejected"], Field(...)) if role == "herb"
                            else (Literal["completed"], "completed"))
    return create_model(schema.__name__ + "Converged", __base__=schema, **fields)


def pending_inquiry(state: dict, artifact: dict, stage: str) -> dict:
    return {"pending_inquiry": {"stage": stage, "reason": artifact.get("followup_reason", ""),
                                "items": [{"question": item, "topics": sorted(topics(item)), "status": "unanswered"}
                                          for item in question_items(question_text(artifact))[:3]]}}


def bind_inquiry_items(state: dict, answers: list[dict]) -> None:
    """Materialize answer status with provenance, without asserting medical facts."""
    if not answers:
        return
    row = answers[-1]
    readings = {item["question"]: item for item in answer_readings({**state, "inquiry_answers": answers})
                if item["answer_message_id"] == row["answer_message_id"]}
    row["items"] = [{"question": question, "topics": sorted(topics(question)),
                     "status": readings.get(question, {}).get("status", "unanswered"),
                     "interpretation": readings.get(question, {}).get("interpretation", ""),
                     "answer_message_id": row["answer_message_id"]} for question in question_items(row["question"])]


def repeated_questions(state: dict, artifact: dict) -> tuple[list[str], list[str]]:
    """Only suppress known, fully covered observation questions; keep new details.

    Questions about change/conflict must remain available. No symptom facts or
    safety decisions are changed by this presentation/output validation.
    """
    if not needs_patient_answer(artifact):
        return [], []
    readings = answer_readings(state)
    observations = observation_ledger(state)
    covered = {key for row in observations if row.get("status") != "partial" for key in row["topics"]}
    def comparable(text):
        return re.sub(r"[\s\d.、？?，,。；;：:]|请问|请补充|您|你|目前|是否|有没有|有无", "", text)
    answered_items = {comparable(row["question"]) for row in readings}
    remaining, repeated = [], []
    for question in [item for value in (artifact.get("questions") or [artifact.get("question", "")]) for item in question_items(value)]:
        keys = topics(question)
        # Compound new attributes and explicit corrections are not silently dropped.
        conflict = re.search(r"变化|改变|之前|此前|更正|矛盾|仍然|现在是否", question)
        # A familiar symptom name does not mean its severity/duration and every
        # other attribute have also been answered.
        new_detail = any(re.search(pattern, question) and not any(
            re.search(pattern, row.get("question", "") + row["quote"]) for row in observations if keys & set(row["topics"]))
            for pattern in (r"程度|严重", r"多久|持续时间", r"次数|频率", r"夜间|夜尿", r"尿量|量多|量少", r"血尿|尿血"))
        if ((keys and keys <= covered and fully_recognized_question(question)) or comparable(question) in answered_items) and not conflict and not new_detail:
            repeated.append(question)
        else:
            remaining.append(question)
    return remaining, repeated


def stalled_assessment(artifact: dict) -> bool:
    """Flag a contradictory final answer for regeneration, never infer a route.

    Clinical questions and conclusions are produced by the model with the full
    scoped context. This check only detects the old 'missing details so cannot
    determine' ending when no question or evidence limitation was recorded.
    """
    if needs_patient_answer(artifact) or artifact.get("information_gap") == "evidence":
        return False
    text = artifact.get("conclusion", "")
    return bool(re.search(r"缺少|缺乏|尚缺|尚未提供|未提供", text)
                and re.search(r"(?:不能|无法|不足以|难以).{0,12}(?:确定|判断|生成|形成|辨证)", text))


def unsupported_preference(artifact: dict) -> bool:
    """A food preference cannot be promoted into contradictory clinical evidence."""
    sentences = re.split(r"[。！？\n]", artifact.get("conclusion", ""))
    return any(re.search(r"偏好|饮食偏好", sentence)
               and re.search(r"矛盾|相悖|反证|不符|排除|不支持", sentence)
               and not re.search(r"不能|不应|不足以|不等于|不作为|不宜", sentence)
               for sentence in sentences)


def without_unsupported_preference(text: str) -> str:
    # Remove only a known unsupported inference; never insert a clinical claim.
    return "".join(part for part in re.findall(r"[^。！？\n]+[。！？\n]?", text)
                   if not unsupported_preference({"conclusion": part})).strip()


def missing_preliminary_analysis(state: dict, artifact: dict) -> bool:
    """Known complaints should get more than a restatement and a questionnaire."""
    if len(state.get("symptoms", [])) < 2:
        return False
    text = artifact.get("conclusion", "")
    if not text.strip() and needs_patient_answer(artifact):
        return False  # Necessary inquiry can precede any supported assessment.
    if "流程" in text:
        return False  # Offline demonstration is explicitly labelled as such.
    analysis = re.split(r"为了|请(?:问|提供|补充)|[1１][.、．]", text, maxsplit=1)[0]
    return not re.search(r"可能|提示|考虑|倾向|初步|运化|寒热|虚实|气血", analysis)


def can_finalize_unknown_observations(state: dict, repeated: list[str]) -> bool:
    """Only completed inquiry with unresolved, non-safety observations can finish."""
    safe, preflight = state.get("safety_state", {}), state.get("preflight", {})
    if (not safe.get("subject_confirmed") or not state.get("symptoms") or state.get("symptom_pending")
            or preflight.get("decision") not in {"allow", "clarify"} or preflight.get("question_keys")):
        return False
    if any(re.search(r"过敏|用药|服药|药物|孕|哺乳|年龄|性别|\d+岁|儿童|胸痛|呼吸|昏|血尿|黑便|便血|自杀|剂量", item) for item in repeated):
        return False
    return any(re.search(r"不清楚|不知道|无法提供", row["answer"]) for row in scoped_answers(state))


async def validate_followup(state: dict, artifact: dict, repair, *, assessment: bool = False, finalize=None) -> dict:
    """One bounded correction using existing evidence; never promote a question
    artifact to a completed medical result by just deleting its questions.
    """
    questions = question_items(question_text(artifact)) if needs_patient_answer(artifact) else []
    if len(questions) > 3:
        questions = sorted(questions, key=lambda value: not bool(SAFETY_QUESTION.search(value)))[:3]
        artifact = {**artifact, **({"questions": questions} if "questions" in artifact else {"question": "\n".join(questions)})}
    remaining, repeated = repeated_questions(state, artifact)
    reason = convergence_reason(state, artifact)
    if finalize and reason:
        return await finalize_followup(state, artifact, finalize, reason, assessment=assessment)
    stalled = assessment and stalled_assessment(artifact)
    preference_error = assessment and unsupported_preference(artifact)
    analysis_missing = assessment and missing_preliminary_analysis(state, artifact)
    if not repeated and not stalled and not preference_error and not analysis_missing:
        return artifact
    if repeated and remaining and not preference_error and not analysis_missing:
        return {**artifact, **({"questions": remaining} if "questions" in artifact else {"question": "\n".join(remaining)})}
    feedback = (("上次产物把饮食偏好单独当作证候矛盾或反证，这超出了用户报告的症状事实。"
                 "请移除这个推断，保留用户原话及确已报告的怕冷、腹胀等症状，"
                 "根据实际症状整理初步方向与局限，保留尚未回答的具体追问。"
                 if preference_error else
                 "上次结论只有已知症状复述或问卷，缺少基于实际资料的初步分析。"
                 "已有多项明确主诉时，conclusion 应说明这些症状提示可从哪些方面分析、已有支持与反证，"
                 "证据不支持具体证型时只讨论可判断的范围。古籍未命中不代表不能分析用户已经报告的症状。"
                 "请在具体追问之前给出上述初步分析，不编造阳性症状或硬定证型。"
                 if analysis_missing else
                 "上次产物把缺少可补充的资料当作无法确定的终点，却没有具体追问。请重新检查已提供及已答资料："
                 "如果确有尚未回答且有助区分证候的信息，提出最多三个具体问题，并保留已有依据的初步分析；"
                 "如果缺项已回答、明确不清楚或无法提供，保留未知，给出已知依据支持的初步方向、反证和局限，"
                 "不要要求完整四诊后才允许初步辨证，不要硬定证型。"
                 if stalled else "上次产物只重复询问了用户已经回答的项目，不能再次返回这些问题。") +
                "请基于既有实际证据重新生成本阶段完整产物，不再调用工具。若无其他必要新问题，"
                "完成带不确定性的初步分析；可选资料仍未知、文献或剂量不足时如实标注受限，"
                "不能猜测、不代表个人处方或安全通过。确有新的必要问题时仅问尚未回答的具体项。\n"
                "上一份产物：" + json.dumps(artifact, ensure_ascii=False) + "\n"
                "用户回答释义：" + json.dumps(answer_readings(state), ensure_ascii=False))
    corrected = await repair(feedback)
    remaining, repeated = repeated_questions(state, corrected)
    reason = convergence_reason(state, corrected)
    if finalize and reason:
        return await finalize_followup(state, corrected, finalize, reason, assessment=assessment)
    if repeated and not remaining:
        if finalize and can_finalize_unknown_observations(state, repeated):
            # Generate a fresh constrained artifact from the same real sources;
            # never promote the question artifact by deleting its questions.
            final = await finalize("用户已对这些项目说明不清楚，修正后仍重复询问，不能继续循环。"
                                   "请以已知症状和实际来源重新生成完整的处方参考，保留未知及安全问题。"
                                   "不得猜测剂量或声称个人适用；依据无法支持候选时 herbs 留空，说明 evidence 缺口。"
                                   "本次 questions 必须为 []，information_gap 仅允许 none、optional 或 evidence。\n"
                                   "已答问题与回答：" + json.dumps(scoped_answers(state), ensure_ascii=False))
            if not needs_patient_answer(final):
                return final
        from .model import InvalidModelOutput
        raise InvalidModelOutput("模型修正后仍重复已答问题")
    if repeated:
        corrected = {**corrected, **({"questions": remaining} if "questions" in corrected else {"question": "\n".join(remaining)})}
    if assessment and unsupported_preference(corrected):
        conclusion = without_unsupported_preference(corrected.get("conclusion", ""))
        if not conclusion:
            from .model import InvalidModelOutput
            raise InvalidModelOutput("模型修正后仍将饮食偏好当作证候反证")
        corrected = {**corrected, "conclusion": conclusion,
                     "summary": without_unsupported_preference(corrected.get("summary", "")) or conclusion}
    return corrected


async def finalize_followup(state: dict, artifact: dict, finalize, reason: str, *, assessment: bool) -> dict:
    from .model import InvalidModelOutput

    result = await finalize(
        "追问收束原因：" + reason + "。本轮不得继续返回常规追问，请根据用户事实和实际取得的来源重新生成完整阶段产物。"
        "保留阴性、未知和未确认条件，不能把问句当作症状。辨证列支持、反证和鉴别范围；"
        "方剂有依据时给出候选和具体组成，无依据时允许空候选并具体说明原因；本草核对已有条目和风险。"
        "不能因古籍未命中抹掉已有临床分析，剂量缺失只限制剂量；不得编造候选、来源或剂量，不声称个人用药安全通过。"
        "只完成受限分析，information_gap 填 optional 或 evidence，所有 question/questions 为空。\n"
        "追问进度：" + json.dumps(inquiry_progress(state), ensure_ascii=False)
        + "\n已回答和主动报告的观察保留在本次患者上下文中，须继续按原始引用核对，不得将未知改为否认。"
        + "\n本阶段待修正项：" + json.dumps({key: artifact[key] for key in
            ("status", "information_gap", "question", "questions", "uncertainties", "problems", "required_actions")
            if key in artifact}, ensure_ascii=False))
    if needs_patient_answer(result):
        raise InvalidModelOutput("追问收束后仍未返回完整阶段产物")
    if assessment and unsupported_preference(result):
        result = {**result, "conclusion": without_unsupported_preference(result.get("conclusion", "")),
                  "summary": without_unsupported_preference(result.get("summary", ""))}
    if assessment and (not result.get("conclusion", "").strip() or missing_preliminary_analysis(state, result)):
        raise InvalidModelOutput("追问收束后缺少有依据的初步分析")
    result = {**result, "followup_reason": "追问已收束：" + reason}
    return result
