"""Evidence-scoped safety facts. These are engineering gates, not clinical criteria."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from uuid import uuid4

from .db import now
from .safety import mentioned_herbs, normalize, KNOWN, ALIASES

URGENT = ("胸痛", "呼吸困难", "昏厥", "大出血", "意识不清", "自杀",
          "咯血", "咳血", "惊厥", "脑梗", "嘴歪", "半身不遂", "硬得像板")
FAMILY = r"母亲|妈妈|父亲|爸爸|朋友|妻子|老婆|丈夫|老公|大宝|二宝|老大|老二|孩子|宝宝|儿子|女儿|同事|奶奶|爷爷|外婆|外公|姐姐|哥哥|妹妹|弟弟|家人|妈|爸|别人"
FAMILY_ALIASES = {"妈妈": "母亲", "妈": "母亲", "爸爸": "父亲", "爸": "父亲", "老婆": "妻子", "老公": "丈夫", "宝宝": "孩子", "别人": "未明确"}
CONSTITUTIONS = r"气虚|阳虚|阴虚|湿热|痰湿|血瘀|气郁|特禀|平和"
DISEASES = r"高血压|糖尿病|肾病|肝病|哮喘|心脏病"
NEGATIVE = r"没有|并无|否认|不伴|未见|未曾|从未|不过敏|无过敏|不再|未怀孕|未哺乳|不是|不在|未在|(?<!并)无|不"


def subject_of(text: str, inherited: str = "unknown") -> str:
    match = re.search(rf"(?:我的?|帮|替|给|为)?({FAMILY})", text)
    if match:
        entity = FAMILY_ALIASES.get(match[1], match[1])
        return "other:" + entity
    if re.search(r"(?:替|帮|为|给)(?:她|他)(?:们)?", text):
        return inherited if inherited.startswith("other:") else "other:未明确"
    if re.search(r"我(?!的?(?:" + FAMILY + r"))|本人|我自己", text) and not re.search(r"不是(?:我(?:本人|自己)?|本人)", text):
        return "self"
    if re.search(r"她|(?<!其)他", text):
        return inherited if inherited.startswith("other:") else "other:未明确"
    return inherited


def assertion_of(text: str, start: int, end: int) -> str:
    before, after = text[:start], text[end:end + 8]
    if re.search(r"不是没有|并非没有|不能说没有|不代表没有|没有否认|不能排除|不确定|不知道|可能|好像|疑似|不太清楚", before + after):
        return "uncertain"
    if re.search(NEGATIVE, before[-18:]) or re.fullmatch(r"(?:不过敏|阴性|没有|无)[了。 ]*", after):
        return "absent"
    return "present"


def temporal_of(text: str) -> str:
    if re.search(r"(?:没|未|没有|尚未|并未).{0,3}(?:结束|停用|停药|停止|好转)", text):
        return "current"
    if re.search(r"结束|停用|停服|停药|(?:已经|已|现在|目前)停|不再|已停止|已经好了|已好转", text):
        return "ended"
    if re.search(r"以前|曾经|曾(?:确诊|诊断|就诊|住院)|既往|去年|昨晚|昨天|过去|小时(?:前|之前)", text):
        return "past"
    return "current"


def extract_facts(text: str, *, subject="unknown", source_id="", kind="message", thread_id="") -> list[dict]:
    """Never treat hypothetical/quoted material as a confirmed self report."""
    result = []
    inherited = subject
    for clause in re.split(r"[，,。；;！？!?\n]|但是|不过(?!敏)|但|而且|并且|并有|还有|而是", text):
        clause = clause.strip()
        if not clause:
            continue
        if result and result[-1]["category"] == "allergy" and re.search(r"起疹|皮疹|发痒|瘙痒|红肿|休克", clause):
            result[-1]["details"] += "；" + clause
            result[-1]["source"]["quote"] += "；" + clause
        if re.search(r"如果|假如|假设|例如|举例|听说|书上|文章说|有人说|原文|引用|他说|她说|你说|你认为|系统判断|模型判断", clause):
            continue
        if re.search(r"了解|解释|科普|介绍|什么是|怎么理解|是什么意思|有什么作用|为什么", clause) and not re.search(r"我(?:现在|目前|最近)?(?:有|出现|发生|感觉|正在)|现在.{0,4}(?:胸痛|呼吸困难)", clause):
            continue
        who = subject_of(clause, inherited)
        if who != "unknown":
            inherited = who
        temporal = temporal_of(clause)
        uncertain_question = bool(re.search(r"怎么办|会怎样|是什么|会不会|是否|吗$", clause))

        def add(category, entity, start=0, end=None, details="", assertion=None):
            end = len(clause) if end is None else end
            value = assertion or assertion_of(clause, start, end)
            if uncertain_question:
                value = "uncertain"
            result.append({
                "id": str(uuid4()), "category": category, "entity": entity,
                "subject": who if kind != "attachment" else "unknown",
                "assertion": value, "temporal_scope": temporal, "details": details,
                "confirmation": "self_report" if who == "self" and kind != "attachment" else "unconfirmed",
                "source": {"kind": kind, "id": source_id, "thread_id": thread_id,
                           "quote": clause, "collected_at": now()},
            })

        for term in URGENT:
            for match in re.finditer(term, clause):
                add("urgent", term, match.start(), match.end())
                if temporal == "past" and re.search(r"现在(?:已经)?(?:没有|不)", clause):
                    result[-1]["assertion"] = "present"
                    add("urgent", term, assertion="absent")
                    result[-1]["temporal_scope"] = "current"
        for entity, pattern in (("怀孕", r"怀孕|孕期|妊娠"), ("哺乳", r"哺乳|母乳"), ("备孕", r"备孕")):
            match = re.search(pattern, clause)
            if match:
                add("pregnancy_lactation", entity, match.start(), match.end())
        age = re.search(r"(\d{1,3})\s*岁", clause)
        if age:
            add("age", age[1], age.start(), age.end())
        elif re.search(r"儿童|孩子|宝宝|婴儿|小孩|未成年", clause):
            add("age", "未知")

        no_current_reaction = re.search(r"(?:现在|目前|当前).{0,4}(?:没有|无|未出现)(?:任何)?过敏(?:反应|症状)", clause)
        if "过敏" in clause and not no_current_reaction:
            match = re.search(r"对(?:于)?(.{1,35}?)(?:有|不)?过敏", clause)
            entities = []
            if match:
                entities = [part.strip() for part in re.split(r"[、/和及]", match[1]) if part.strip()]
                start, end = match.start(), match.end()
            else:
                known = mentioned_herbs(clause)
                simple = re.search(r"(青霉素|花生|海鲜|头孢\w*|牛奶|鸡蛋)(?:有)?过敏", clause)
                entities = known or ([simple[1]] if simple else [])
                start, end = clause.index("过敏"), clause.index("过敏") + 2
            if not entities:
                entities = ["过敏史" if re.search(r"无过敏|没有.{0,4}过敏|否认.{0,4}过敏", clause) else "过敏原不明"]
            for entity in entities:
                if entity in KNOWN or entity in ALIASES:
                    entity = normalize(entity)
                value = assertion_of(clause, start, end)
                if re.search(r"不过敏|不对.{0,35}过敏", clause) and not re.search(r"不是没有", clause):
                    value = "absent"
                add("allergy", entity, start, end, clause, value)
        constitution = re.search(rf"({CONSTITUTIONS})体质", clause)
        if constitution:
            add("constitution", constitution[0], constitution.start(), constitution.end(), clause)
        for match in re.finditer(DISEASES, clause):
            add("history", match[0], match.start(), match.end(), clause)
        history = re.search(r"(?:既往史[:：]|曾确诊(?:为)?|确诊为|曾诊断为|诊断为|患有)([^；。]{1,80})", clause)
        if history and not any(f["source"]["quote"] == clause and f["category"] == "history" for f in result):
            add("history", history[1], history.start(), history.end(), clause)
        medication = re.search(r"(?:正在|目前|现在|长期)(?:在)?(?:吃|服用|服|使用|用)([^；。]{1,60})", clause)
        known_med = re.search(r"华法林|降压药|抗凝药|降糖药|合并用药|同时服", clause)
        if medication and not re.search(r"服用|药|片|胶囊|剂|丸|华法林|阿司匹林|布洛芬|二甲双胍|氨氯地平|维生素", clause):
            medication = None
        if medication or known_med:
            match = medication or known_med
            entity = medication[1] if medication else known_med[0]
            add("medication", entity, match.start(), match.end(), clause)
        denial = re.search(r"(?:没有|未|不|无)(?:在|正在)?(服药|用药|服用(?:任何|其他)?药(?:物|品)?|使用(?:任何|其他)?药(?:物|品)?|吃药)", clause)
        if denial and not (medication or known_med):
            assertion = "uncertain" if assertion_of(clause, denial.start(1), denial.end(1)) == "uncertain" else "absent"
            add("medication", denial[1], denial.start(1), denial.end(1), clause, assertion)
    return result


def empty_state() -> dict:
    return {"version": 0, "subject": "unknown", "subject_confirmed": False, "subject_epoch": 0, "subject_evidence": None, "facts": [],
            "pending": None, "answers": {}, "status": "resolved", "episode": "", "clinical": {}, "safety_by_subject": {}}


def merge_facts(old: list[dict], new: list[dict]) -> list[dict]:
    facts = copy.deepcopy(old)
    for fact in new:
        for prior in facts:
            same = all(prior.get(k) == fact.get(k) for k in ("category", "entity", "subject"))
            if fact["category"] == prior["category"] == "age" and fact["subject"] == prior["subject"] and fact["entity"].isdigit() and fact["assertion"] == "present":
                prior["superseded"] = True
            if (prior["category"] == fact["category"] == "allergy" and prior["subject"] == fact["subject"]
                and prior["assertion"] != fact["assertion"]
                and (prior["entity"] == fact["entity"] or "过敏史" in {prior["entity"], fact["entity"]})):
                fact["conflict"] = True
            if same and prior.get("temporal_scope") == fact["temporal_scope"]:
                if fact["category"] == "age" and fact["assertion"] == "absent" and prior["assertion"] == "present":
                    fact["conflict"] = True
                    continue
                if fact["category"] == "allergy" and prior["assertion"] != fact["assertion"]:
                    fact["conflict"] = True
                    continue
                # A new denial never silently removes a reported urgent event.
                if fact["category"] == "urgent" and prior["assertion"] == "present" and fact["assertion"] == "absent":
                    prior["temporal_scope"] = "past"
                else:
                    prior["superseded"] = True
            if same and fact["temporal_scope"] == "ended" and fact["category"] in {"medication", "pregnancy_lactation"}:
                prior["superseded"] = True
            if same and fact["assertion"] == "absent" and fact["category"] in {"medication", "pregnancy_lactation"}:
                prior["superseded"] = True
        facts.append(fact)
    return [fact for fact in facts if not fact.get("superseded")]


QUESTIONS = {
    "subject": "这次是你本人，还是替别人咨询？如果涉及多人，请明确本次主要咨询对象。",
    "allergy_details": "请说明具体过敏原和发生过的反应；不知道时请直接说明不知道。",
    "pregnancy_status": "这条孕哺资料目前仍然适用吗？请说明是怀孕、哺乳还是备孕，以及当前或结束的时间。",
    "medication_details": "请提供目前使用的全部药品名称和用药情况；不知道时请直接说明。",
    "child_details": "请补充咨询对象的具体年龄和现有就诊情况。",
    "urgent_details": "请说明这些情况的发生时间、目前是否仍有症状及是否已就医；相关风险需由医师核对。",
    "conflict": "这次资料与已有风险记录不一致，请具体说明哪条需要更正及原因。",
    "model_details": "安全预审仍需补充资料，请说明目前的风险情况与就诊情况，交由医师核对。",
}


def bind_answer(state: dict, text: str, source: dict) -> list[dict]:
    pending = state.get("pending")
    if not pending:
        return []
    key = pending["key"]
    short = bool(re.fullmatch(r"(?:不是|没有|是|是的|仍然是|还在|不知道|不清楚|跳过|已经结束了)[。！! ]*", text.strip()))
    if key == "subject":
        who = subject_of(text)
        if who == "unknown":
            return []
        state["subject"] = who
        state["subject_confirmed"] = who != "other:未明确"
        for fact in state["facts"]:
            if fact["subject"] == "unknown" or (fact["subject"] == "other:未明确" and who.startswith("other:")):
                fact["subject"] = who
                fact["ownership_confirmed_by"] = source["id"]
        state["pending"] = None
        return []
    if short and key not in {"pregnancy_status"}:
        return []
    if re.search(r"不知道|不清楚|跳过", text):
        state["answers"][key] = "unknown"
        return []
    if key == "allergy_details":
        explicit = re.search(r"(?:过敏原是|过敏药物是|药名是)([^，,。；;]{1,40})", text)
        allergens = mentioned_herbs(text)
        allergens += re.findall(r"青霉素|花生|海鲜|牛奶|鸡蛋|头孢[\u4e00-\u9fff]{0,8}", text)
        if explicit:
            allergens.append(explicit[1])
        if allergens:
            state["answers"][key] = "provided"
            state["pending"] = None
            state["facts"] = [f for f in state["facts"] if not (
                f["category"] == "allergy" and f["entity"] == "过敏原不明" and f["subject"] == state["subject"])]
            return [{"id": str(uuid4()), "category": "allergy", "entity": entity, "subject": state["subject"],
                     "assertion": "present", "temporal_scope": "current", "details": text,
                     "confirmation": "self_report", "source": source} for entity in dict.fromkeys(allergens)]
    if key == "pregnancy_status" and short:
        applicable = [f for f in state["facts"] if f["category"] == "pregnancy_lactation"
                      and f["subject"] == state["subject"] and f["assertion"] != "absent"]
        if len({f["entity"] for f in applicable}) != 1:
            return []
        fact = copy.deepcopy(applicable[-1])
        fact["id"] = str(uuid4())
        fact["source"] = source
        if re.search(r"不是|没有", text):
            fact.update(assertion="absent", temporal_scope="current")
        elif "结束" in text:
            # A vague "ended" without time stays unresolved.
            return []
        else:
            fact.update(assertion="present", temporal_scope="current")
        state["answers"][key] = "provided"
        state["pending"] = None
        return [fact]
    parsed = extract_facts(text, subject=state["subject"])
    relevant = {
        "pregnancy_status": "pregnancy_lactation", "allergy_details": "allergy",
        "medication_details": "medication", "child_details": "age", "urgent_details": "urgent",
    }.get(key)
    provided = any(f["category"] == relevant and f["assertion"] != "uncertain" for f in parsed)
    if key in {"pregnancy_status", "medication_details"} and re.search(r"结束|停用|停药|不再", text) and re.search(r"\d|昨天|昨晚|今天|去年", text):
        applicable = [f for f in state["facts"] if f["category"] == relevant
                      and f["subject"] == state["subject"] and f["assertion"] != "absent"]
        if len({f["entity"] for f in applicable}) == 1:
            fact = copy.deepcopy(applicable[-1])
            fact.update(id=str(uuid4()), assertion="absent", temporal_scope="ended", source=source)
            state["answers"][key] = "provided"
            state["pending"] = None
            return [fact]
    if key == "urgent_details":
        provided = bool(re.search(r"\d|昨天|昨晚|今天|去年", text) and re.search(r"就医|医院|目前|现在", text))
    if key == "model_details":
        provided = not short and len(text.strip()) >= 4
    if key == "conflict":
        provided = bool(re.search(r"更正|说错|纠正", text) and parsed)
    if provided:
        state["answers"][key] = "provided"
        state["pending"] = None
        if key == "allergy_details" and any(f["entity"] != "过敏原不明" for f in parsed if f["category"] == "allergy"):
            state["facts"] = [f for f in state["facts"] if not (f["category"] == "allergy" and f["entity"] == "过敏原不明" and f["subject"] == state["subject"])]
    return []


def make_context(saved: dict, text: str, attachment: str, *, message_id="", thread_id="") -> tuple[dict, list[dict]]:
    state = copy.deepcopy(saved or empty_state())
    for key, value in empty_state().items():
        state.setdefault(key, copy.deepcopy(value))
    # Old checkpoints defaulted to self without evidence. Do not promote that
    # default into an identity confirmation.
    if not state["subject_confirmed"]:
        state["subject"] = "unknown"
    if re.fullmatch(r"不是(?:我(?:本人|自己)?|本人)[。！! ]*", text.strip()):
        state.update(subject="unknown", subject_confirmed=False, pending={"key": "subject", "subject": "unknown"})
    subjects = set()
    inherited = state["subject"] if state["subject_confirmed"] else "unknown"
    for clause in re.split(r"[，,。；;]|但是|但", text):
        who = subject_of(clause, inherited)
        if who != "unknown":
            subjects.add(who)
            inherited = who
    subjects.update("other:" + FAMILY_ALIASES.get(m[0], m[0])
                    for m in re.finditer(FAMILY, text))
    if re.search(r"我(?!的?(?:" + FAMILY + r"))|本人", text):
        subjects.add("self")
    if re.search(r"她|(?<!其)他", text):
        named = {who for who in subjects if who.startswith("other:") and who != "other:未明确"}
        pronoun_subject = next(iter(named)) if len(named) == 1 else (state["subject"] if state["subject_confirmed"] and state["subject"].startswith("other:") else "other:未明确")
        subjects.add(pronoun_subject)
    if len(subjects) == 1:
        explicit = next(iter(subjects))
    else:
        explicit = "unknown"
    target = re.search(rf"(?:替|帮|为|给)((?:我的?)?(?:{FAMILY})|我自己|本人|我|她|他).{{0,6}}(?:咨询|问|看)", text)
    if target:
        explicit = subject_of(target[1], state["subject"] if state["subject_confirmed"] else "unknown")
    if len(subjects) > 1 and not target:
        state["safety_by_subject"][state["subject"]] = {
            key: copy.deepcopy(state.get(key)) for key in ("pending", "answers", "status", "episode")}
        state["subject"] = "unknown"
        state["subject_confirmed"] = False
        state["pending"] = {"key": "subject", "subject": "unknown"}
    resolving = bool(state.get("pending") and state["pending"]["key"] == "subject" and len(subjects) <= 1)
    if resolving:
        source = {"kind": "message", "id": message_id, "thread_id": thread_id, "quote": text, "collected_at": now()}
        bind_answer(state, text, source)
    if explicit != "unknown" and explicit != state["subject"] and (target or not state.get("pending") or re.search(r"换成|改问|现在问", text)):
        state["safety_by_subject"][state["subject"]] = {
            key: copy.deepcopy(state.get(key)) for key in ("pending", "answers", "status", "episode")}
        state["subject"] = explicit
        state["subject_confirmed"] = explicit != "other:未明确"
        personal = state["safety_by_subject"].get(explicit, {"pending": None, "answers": {}, "status": "resolved", "episode": ""})
        state.update(copy.deepcopy(personal))
    elif explicit != "unknown" and explicit == state["subject"]:
        state["subject_confirmed"] = explicit != "other:未明确"
    source = {"kind": "message", "id": message_id, "thread_id": thread_id, "quote": text, "collected_at": now()}
    answering = state.get("pending", {}).get("key") if state.get("pending") else None
    bound = bind_answer(state, text, source) if len(subjects) <= 1 or target else []
    inherited = state["subject"] if state["subject_confirmed"] else "unknown"
    current = extract_facts(text, subject=inherited, source_id=message_id, thread_id=thread_id)
    current += extract_facts(attachment, source_id=message_id, kind="attachment", thread_id=thread_id)
    current += bound
    for fact in current:
        key = {"pregnancy_lactation": "pregnancy_status", "medication": "medication_details",
               "age": "child_details", "allergy": "allergy_details", "urgent": "urgent_details"}.get(fact["category"])
        if key and not (answering == key and state["answers"].get(key) == "provided") and not any(all(old.get(k) == fact.get(k) for k in ("category", "entity", "subject", "assertion", "temporal_scope"))
                           for old in state["facts"]) and not bound:
            state["answers"].pop(key, None)
    state["facts"] = merge_facts(state["facts"], current)
    if not state["episode"]:
        state["episode"] = text
    return state, current


def applicable_facts(state: dict) -> list[dict]:
    return [fact for fact in state["facts"]
            if fact["subject"] in {state["subject"], "unknown"}
            and fact.get("source", {}).get("kind") != "summary"
            and (fact.get("source", {}).get("kind") not in {"manual_profile", "confirmed_profile"}
                 or state.get("subject_confirmed") and state["subject"] == "self")]


def uncovered_allergens(state: dict) -> list[str]:
    return list(dict.fromkeys(fact["entity"] for fact in applicable_facts(state)
                             if fact["category"] == "allergy" and fact["assertion"] == "present"
                             and fact["temporal_scope"] != "ended"
                             and fact["entity"] not in {"过敏原不明", "过敏史"}
                             and not mentioned_herbs(fact["entity"])))


def gate_context(state: dict) -> dict:
    categories, keys = [], []
    restricted = False
    facts = applicable_facts(state)
    if any(f.get("conflict") for f in facts):
        keys.append("conflict")
        restricted = True
    active = [f for f in facts if f["assertion"] != "absent" and f["temporal_scope"] != "ended"]
    current_denials = {(f["category"], f["entity"], f["subject"]) for f in facts
                      if f["assertion"] == "absent" and f["temporal_scope"] == "current"}
    active = [f for f in active if not (f["category"] in {"pregnancy_lactation", "medication"} and f["temporal_scope"] == "past"
                                      and (f["category"], f["entity"], f["subject"]) in current_denials)]
    if any(f["category"] == "urgent" and f["assertion"] == "present" and f["temporal_scope"] == "current"
           and f["subject"] in {state["subject"], "unknown"} for f in active):
        return {"decision": "block", "risk_categories": ["deterministic"], "reason": "命中当前肯定风险，停止治疗路径",
                "required_questions": [], "question_keys": [], "restricted": True}
    risk_facts = [f for f in active if f["category"] in {"urgent", "pregnancy_lactation", "age", "medication", "allergy"}]
    if state.get("identity_required") and (not state.get("subject_confirmed") or state["subject"] in {"unknown", "other:未明确"}):
        keys.append("subject")
    if risk_facts and (state["subject"] in {"unknown", "other:未明确"} or any(f["subject"] == "unknown" for f in risk_facts)):
        keys.append("subject")
    if any(f["category"] == "urgent" for f in active):
        categories.append("deterministic")
        keys.append("urgent_details")
        restricted = True
    for fact in active:
        category = fact["category"]
        if fact.get("conflict"):
            keys.append("conflict")
            restricted = True
        if category == "pregnancy_lactation":
            categories.append(category)
            keys.append("pregnancy_status")
            restricted = True
        if category == "age" and (fact["entity"] == "未知" or int(fact["entity"]) < 18):
            categories.append("child")
            keys.append("child_details")
            restricted = True
        if category == "medication":
            categories.append("concomitant_medication")
            keys.append("medication_details")
            restricted = True
        if category == "allergy":
            categories.append("allergy")
            if fact["entity"] in {"过敏原不明", "过敏史"} or fact["assertion"] == "uncertain":
                keys.append("allergy_details")
                restricted = True
            elif not mentioned_herbs(fact["entity"]):
                # We keep non-herbal allergens but cannot claim candidate coverage.
                restricted = True
    keys = list(dict.fromkeys(keys))
    pending = state.get("pending")
    keys = [key for key in keys if key == "subject" or state["answers"].get(key) != "provided"]
    if pending and pending["key"] in keys:
        keys.remove(pending["key"])
        keys.insert(0, pending["key"])
    analysis_allowed = bool(restricted and not keys and set(categories) == {"allergy"}
                            and state.get("subject_confirmed") and uncovered_allergens(state))
    return {"decision": "clarify" if keys or restricted else "allow",
            "risk_categories": list(dict.fromkeys(categories)),
            "reason": "风险资料待确认或需人工核对" if keys or restricted else "可继续流程，候选仍需独立校验",
            "required_questions": [QUESTIONS[key] for key in keys],
            "question_keys": keys, "restricted": restricted, "analysis_allowed": analysis_allowed}


def response_for(state: dict, decision: dict) -> str:
    keys = decision.get("question_keys", [])
    if decision["decision"] == "block":
        state["status"] = "blocked"
        if "preflight_unavailable" in decision.get("risk_categories", []):
            return "安全预审暂时未能完成，本轮未进入辨证、方剂和本草分析。你的主诉与已确认资料已保留，请重试。"
        return "你提供的信息中出现可能需要及时处理的情况。若症状正在发生，请尽快联系当地急救服务或就近医疗机构，不要等待在线问答。"
    if keys:
        key = keys[0]
        if not state.get("pending") or state["pending"]["key"] != key or not state["pending"].get("id"):
            state["pending"] = {"id": str(uuid4()), "key": key, "subject": state["subject"]}
        state["status"] = "needs_info"
        return "需要先确认安全资料，本轮不生成方剂或治疗建议。" + QUESTIONS[key]
    state["pending"] = None
    state["status"] = "restricted"
    unsupported = [fact["entity"] for fact in applicable_facts(state)
                   if fact["category"] == "allergy" and fact["assertion"] == "present"
                   and fact["temporal_scope"] != "ended" and not mentioned_herbs(fact["entity"])
                   and fact["entity"] not in {"过敏原不明", "过敏史"}]
    if unsupported:
        return "已记录过敏信息：" + "、".join(dict.fromkeys(unsupported)) + "。当前自动用药校验尚不覆盖这些过敏原，需由医师或药师核对，本轮未进入方剂和本草候选分析。可以继续补充症状与就诊资料。"
    return "已记录你补充的风险资料。资料补齐不表示可以安全用药；本轮保持受限，不生成方剂或治疗建议，请由医师或药师核对。可以继续补充症状与就诊资料。"


def candidate_key(fact: dict) -> str:
    value = {k: fact.get(k, "") for k in ("category", "entity", "assertion", "temporal_scope")}
    # Only clinically relevant extra self-reported detail, not message wording, affects duplication.
    detail = re.search(r"(?:起|出现|导致|发生|会|反应为|服用后)(.*(?:疹|痒|红肿|休克|呼吸).*)", fact.get("details", ""))
    value["details"] = detail[1] if detail else ""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def eligible_profile_fact(fact: dict) -> bool:
    """Long-term portrait excludes current symptoms and transient safety states."""
    category = fact.get("category")
    if category in {"allergy", "constitution"}:
        return fact.get("entity") != "过敏原不明"
    if category == "history":
        quote = fact.get("source", {}).get("quote", "")
        return bool(re.search(DISEASES, fact.get("entity", "")) or re.search(
            r"既往史|既往诊疗|曾确诊|确诊为|诊断为|曾诊断|曾就诊|曾住院|做过.{0,12}手术", quote))
    return False


def profile_candidates(facts: list[dict]) -> list[dict]:
    return [copy.deepcopy(f) for f in facts
            if eligible_profile_fact(f)
            and f["subject"] == "self" and f["assertion"] != "uncertain"
            and f["entity"] not in {"过敏原不明"} and f["source"]["kind"] == "message"]
