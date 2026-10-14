"""Source-backed ordinary symptom observations; these are not diagnoses.

Rules handle known expressions. Ambiguity is retained for a targeted question,
never resolved by giving all negations priority or by guessing severity.
"""
from __future__ import annotations

import copy
import re
from uuid import uuid4

from .db import now
from .safety_context import subject_of
from .terms import SYMPTOM_ALIASES


ALIASES = {alias: name for name, aliases in SYMPTOM_ALIASES.items() for alias in aliases}
MENTION = re.compile("|".join(re.escape(alias) for alias in sorted(ALIASES, key=len, reverse=True)))
BOUNDARY = re.compile(r"[，,。；;！？!?\n]|但是|不过(?!敏)|但|而且|并且|同时|还有|并有|而是")
DOUBLE = r"(?:并)?不是没有|并非没有|不是不|并非不"
UNCERTAIN = r"不能说没有|不代表没有|不能排除|不确定|不知道|不清楚|可能|好像|似乎|疑似|是否|有没有|会不会"
NEGATIVE = r"没有|并无|否认|不伴|未见|从未|未曾|不再|无|不"
TIME = re.compile(r"以前|曾经|既往|去年|昨晚|昨天|过去|\d+小时前|现在|目前|今天|最近|这次")
DEGREE = re.compile(r"不明显|不严重|不太严重|不太明显|不是很严重|不是很明显|不怎么|不经常|不频繁|很少|偶尔|偶有|轻微|明显|严重|剧烈|频繁|持续性|间歇性")
TREND = re.compile(r"(?:没有|未|没)(?:明显)?(?:加重|好转|减轻)|(?:有所|逐渐)?(?:加重|好转|减轻)")
TIMING = r"夜间|夜里|晚上|白天|晨起|早上|睡前|饭后|饭前"


def duration_text(text: str) -> str:
    match = re.search(r"(?:\d+|[一二两三四五六七八九十半]+)(?:[到至~～-](?:\d+|[一二两三四五六七八九十]+))?(?:个)?(?:小时|天|周|星期|礼拜|个月|月|年)", text)
    return match[0] if match else ""


def bind_details(facts: list[dict], text: str, *, subject: str, message_id: str, thread_id: str,
                 kind="message", start=0) -> list[dict]:
    """Bind a bare qualifier only to one current, explicitly present complaint."""
    clause = text.strip("。！! ")
    duration = r"(?:\d+|[一二两三四五六七八九十半]+)(?:个)?(?:小时|天|周|星期|礼拜|个月|月|年)"
    if not re.fullmatch(rf"(?:(?:{TIMING})(?:更|比较)?(?:{DEGREE.pattern}|{TREND.pattern})?|"
                        rf"(?:持续|已经|已有|有)?(?:了)?(?:有)?{duration}|(?:{DEGREE.pattern}|{TREND.pattern}))", clause):
        return []
    active = effective(facts, subject)
    active = [f for f in active if f["time_scope"] == "current" and f["assertion"] == "present"]
    if len(active) != 1 or subject == "unknown":
        return []
    fact = copy.deepcopy(active[0])
    fact.update(id=str(uuid4()), correction=False)
    fact["source"] = {"kind": kind, "message_id": message_id, "thread_id": thread_id,
                      "quote": clause, "start": start, "end": start + len(text), "collected_at": now()}
    ref = {"fact_id": fact["id"], "message_id": message_id, "collected_at": fact["source"]["collected_at"]}
    changes = {"duration_text": duration_text(clause),
               "timing_text": "、".join(dict.fromkeys(re.findall(TIMING, clause))),
               "degree_text": (match[0] if (match := DEGREE.search(clause)) else ""),
               "trend_text": (match[0] if (match := TREND.search(clause)) else "")}
    for key, value in changes.items():
        if value:
            fact[key] = value
            fact.setdefault("evidence_refs", {})[key] = ref
    return [fact]


def unmapped_mentions(clause: str, known: list):
    """Keep an unrecognized complaint as raw evidence, requiring confirmation."""
    physical = list(re.finditer(r"(?P<raw>(?:眼睛|头皮|手指|脚趾|肩膀|牙龈|喉咙|皮肤|小腿|后背).{0,12}?(?:酸涩|发紧|烧灼|灼热|抽动|抽痛|麻木|皲裂|不舒服))", clause))
    self_report = re.search(r"(?:我(?:现在|目前|最近)?(?:感觉|觉得|有|出现))(?P<raw>.{2,60})", clause)
    candidates = physical or ([self_report] if self_report and not known else [])
    accepted = []
    for match in candidates:
        if re.search(r"问题|建议|想法|账号|密码|文档|代码|模型|朋友|家人|需要|希望|知道|过敏|孕|哺乳|用药|体质|病史|病历", match["raw"]):
            continue
        raw = match["raw"].strip()
        duration = duration_text(raw)
        if duration:
            raw = raw[:raw.index(duration)].strip()
        if not raw or any(alias in raw for alias in ALIASES):
            continue
        start = clause.find(raw, match.start("raw"))
        end = start + len(raw)
        if any(start < item.end() and end > item.start() for item in known):
            continue
        accepted.append(re.compile(re.escape(raw)).search(clause, start, end))
    return accepted


def extract(text: str, *, subject="self", message_id="", thread_id="", kind="message") -> list[dict]:
    observations = []
    inherited = subject
    offset = 0
    for boundary in [*BOUNDARY.finditer(text), None]:
        end = boundary.start() if boundary else len(text)
        clause = text[offset:end]
        start_offset = offset
        offset = boundary.end() if boundary else len(text)
        if not clause.strip():
            continue
        if re.search(r"如果|假如|假设|例如|举例|听说|书上|文章说|原文|引用|你说|系统判断|模型判断", clause):
            continue
        if re.search(r"什么是|科普|解释|介绍|是什么意思", clause) and not re.search(r"我(?:现在|最近)?(?:有|感觉|出现)", clause):
            continue
        mentions = list(MENTION.finditer(clause))
        mentions = sorted([*mentions, *unmapped_mentions(clause, mentions)], key=lambda item: item.start())
        if not mentions and observations and re.fullmatch(r"\s*(?:现在|目前)(?:已经|已)?(?:没有(?:了)?|好了|不再有了?)\s*", clause):
            recent = [f for f in observations if f["source"]["quote"] == observations[-1]["source"]["quote"]]
            # An omitted object is bindable only when that clause named one symptom.
            if len({f["symptom"] for f in recent}) == 1:
                fact = copy.deepcopy(recent[-1])
                fact.update(id=str(uuid4()), assertion="absent", time_scope="current", degree_text="", trend_text="", duration_text="", correction=True)
                fact["source"].update(quote=clause.strip(), start=start_offset, end=end)
                observations.append(fact)
            continue
        if not mentions and observations:
            # Only the immediately preceding single complaint can own this clause.
            recent = [f for f in observations if f["source"]["quote"] == observations[-1]["source"]["quote"]]
            observations += bind_details(recent, clause, subject=inherited, message_id=message_id,
                                         thread_id=thread_id, kind=kind, start=start_offset)
        assertion_scope, time_scope = "present", "current"
        previous_end = 0
        for i, match in enumerate(mentions):
            prefix = clause[previous_end:match.start()]
            suffix = clause[match.end():mentions[i + 1].start() if i + 1 < len(mentions) else len(clause)]
            previous_end = match.end()
            # A postfix modifier belongs to the previous mention, not also to
            # the next mention in an unpunctuated string.
            if i:
                prefix = re.sub(r"^\s*(?:" + DEGREE.pattern + "|" + TREND.pattern + r")", "", prefix, count=1)
            who = subject_of(prefix, inherited)
            if who != "unknown":
                inherited = who
            temporal = list(TIME.finditer(prefix))
            if temporal:
                time_scope = "current" if temporal[-1][0] in {"现在", "目前", "今天", "最近", "这次"} else "past"
                assertion_scope = "present"
            # Only an enumeration continues the prior assertion scope. A new
            # predicate/person/time starts its own scope even without punctuation.
            enumeration = i > 0 and bool(re.fullmatch(r"[\s、和及与或]*", prefix))
            if not enumeration:
                assertion_scope = "present"
            if re.search(UNCERTAIN, prefix):
                assertion_scope = "uncertain"
            elif re.search(DOUBLE, prefix):
                assertion_scope = "present"
            elif (not re.search(r"不(?:太)?(?:明显|严重|怎么|经常|频繁)\s*$", prefix)
                  and re.search(r"(?:" + NEGATIVE + r")(?:明显|严重|剧烈|轻微|持续|反复|任何|什么|一点|再|出现|发生|伴有|伴随|的|\s)*$", prefix)):
                assertion_scope = "absent"
            value = assertion_scope
            post_time = re.match(r"\s*(?:是|属于)?(以前|曾经|既往|现在|目前)", suffix)
            if post_time:
                time_scope = "current" if post_time[1] in {"现在", "目前"} else "past"
            if ((i + 1 == len(mentions) and re.search(UNCERTAIN, suffix))
                    or re.fullmatch(r"\s*(?:不确定|不知道|不清楚)\s*", suffix)
                    or re.search(r"吗\s*$", clause)):
                value = "uncertain"
            elif re.match(r"\s*(?:" + DOUBLE + r")", suffix):
                value = "present"
            elif re.fullmatch(r"\s*(?:(?:现在|目前|已经|已)\s*)?(?:没有|无|不再有)(?:了)?[\s、]*", suffix) and (i + 1 == len(mentions) or "了" in suffix):
                value = "absent"
                if re.search(r"现在|目前|已经|已|了", suffix):
                    time_scope = "current"
            degree = DEGREE.search(suffix) or DEGREE.search(prefix)
            trend = TREND.search(suffix)
            # 'No worsening' is a change statement, not evidence of presence or
            # absence. Existing explicit facts may supply the missing presence.
            if trend and not re.search(r"有|出现|发生|仍|还|正在", prefix) and value == "present" and not degree:
                value = "unspecified"
            if re.search(r"时间不清|不知道何时|不清楚何时", prefix + suffix):
                time_scope = "unknown"
            observations.append({
                "id": str(uuid4()), "symptom": ALIASES.get(match[0], match[0]), "subject": who if kind == "message" else "unknown",
                "unmapped": match[0] not in ALIASES,
                "assertion": value, "time_scope": time_scope,
                "degree_text": degree[0] if degree else "", "trend_text": trend[0] if trend else "",
                "timing_text": "、".join(dict.fromkeys(re.findall(r"夜间|夜里|晚上|白天|晨起|早上|睡前|饭后|饭前", clause))),
                "duration_text": duration_text(suffix) or (duration_text(prefix) if i == 0 else ""),
                "correction": bool(re.search(r"更正|纠正|说错|实际|现在|目前|不再", prefix + suffix)),
                "source": {"kind": kind, "message_id": message_id, "thread_id": thread_id,
                           "quote": clause.strip(), "start": start_offset + match.start(),
                           "end": start_offset + match.end(), "collected_at": now()},
            })
    return observations


def merge(old: list[dict], new: list[dict]) -> list[dict]:
    """Keep evidence, marking explicit corrections instead of deleting history."""
    facts = copy.deepcopy(old)
    for fact in copy.deepcopy(new):
        identity = (fact["source"]["message_id"], fact["source"]["kind"], fact["source"]["start"], fact["source"]["end"])
        if identity[0] and any(identity == (f["source"]["message_id"], f["source"]["kind"], f["source"]["start"], f["source"]["end"]) for f in facts):
            continue
        if fact["correction"] and fact["assertion"] in {"present", "absent"}:
            for prior in facts:
                if (prior["symptom"], prior["subject"], prior["time_scope"]) == (fact["symptom"], fact["subject"], fact["time_scope"]):
                    prior["superseded_by"] = fact["id"]
        facts.append(fact)
    return facts


def view(facts: list[dict], subject: str) -> tuple[list[str], list[str], dict | None]:
    active = [f for f in facts if not f.get("superseded_by") and not f.get("archived_episode")]
    if subject == "unknown" and active:
        return [], [], {"kind": "subject", "symptom": "", "question": "本次主要为谁咨询？请说明你本人或具体代问对象。"}
    selected = [f for f in active if f["subject"] == subject]
    unknown = [f for f in active if f["subject"] == "unknown"]
    if unknown:
        return [], [], {"kind": "subject", "symptom": unknown[0]["symptom"],
                        "question": f"你提到的{unknown[0]['symptom']}属于谁？请说明你本人或具体代问对象。"}
    positive, denied = [], []
    pending = None
    for name in dict.fromkeys([*SYMPTOM_ALIASES, *(f["symptom"] for f in selected)]):
        current = [f for f in selected if f["symptom"] == name and f["time_scope"] == "current"]
        values = {f["assertion"] for f in current}
        unresolved = "uncertain" in values or {"present", "absent"} <= values
        if any(f.get("unmapped") and not f.get("mapping_confirmed") and f["assertion"] != "absent" for f in current):
            pending = pending or {"kind": "mapping", "symptom": name,
                                  "question": f"“{name}”暂未匹配已收录术语。请确认：这是你当前的不适描述吗？确认后将保留原文检索。"}
        elif unresolved:
            pending = pending or {"kind": "conflict" if {"present", "absent"} <= values else "assertion",
                                  "symptom": name, "question": f"请确认：你目前有{name}，还是明确没有{name}？"}
        elif "present" in values:
            positive.append(name)
        elif "absent" in values:
            denied.append(name)
        elif current:
            pending = pending or {"kind": "assertion", "symptom": name,
                                  "question": f"你描述了{name}的变化，目前是否有{name}？"}
        elif any(f["symptom"] == name for f in selected):
            pending = pending or {"kind": "time", "symptom": name,
                                  "question": f"你提到的{name}是既往情况吗？目前是否仍有{name}？"}
    return positive, denied, pending


def bind_answer(facts: list[dict], pending: dict | None, text: str, *, subject: str, message_id: str, thread_id: str) -> list[dict]:
    if not pending or pending["kind"] not in {"assertion", "time", "conflict", "mapping"}:
        return []
    answer = text.strip("。！! ")
    yes = answer in {"是", "是的", "有", "有的", "还有", "仍然有", "现在有"}
    no = answer in {"不是", "没有", "没有了", "现在没有", "现在没有了"}
    if not (yes or no):
        return []
    prior = next((f for f in reversed(facts) if f["symptom"] == pending["symptom"] and
                  f["subject"] == subject and not f.get("superseded_by")), {})
    return [{"id": str(uuid4()), "symptom": pending["symptom"], "subject": subject,
             "unmapped": pending["kind"] == "mapping", "mapping_confirmed": pending["kind"] == "mapping",
             "assertion": "present" if yes else "absent", "time_scope": "current",
             "degree_text": prior.get("degree_text", "") if yes else "",
             "trend_text": prior.get("trend_text", "") if yes else "",
             "duration_text": prior.get("duration_text", "") if yes else "", "correction": True,
             "attributes_source_id": prior.get("id", "") if yes else "",
             "source": {"kind": "message", "message_id": message_id, "thread_id": thread_id,
                        "quote": text, "start": 0, "end": len(text), "collected_at": now(),
                        "question": pending["question"]}}]


def effective(facts: list[dict], subject: str) -> list[dict]:
    """Compact repeated observations, keeping evidence for every carried field."""
    groups = {}
    for fact in facts:
        if fact["subject"] != subject or fact.get("superseded_by") or fact.get("archived_episode"):
            continue
        key = (fact["symptom"], fact["assertion"], fact["time_scope"])
        previous = groups.get(key)
        value = copy.deepcopy(fact)
        source_ref = {"fact_id": fact["id"], "message_id": fact.get("source", {}).get("message_id", ""),
                      "collected_at": fact.get("source", {}).get("collected_at", "")}
        refs = {"statement": source_ref}
        for field in ("duration_text", "degree_text", "trend_text", "timing_text"):
            if value.get(field):
                refs[field] = fact.get("evidence_refs", {}).get(field) or source_ref
            elif previous and previous.get(field):
                value[field] = previous[field]
                refs[field] = previous["evidence_refs"].get(field, source_ref)
        value["evidence_refs"] = refs
        groups[key] = value
    return list(groups.values())


def describe(facts: list[dict], subject: str) -> str:
    facts = effective(facts, subject)
    labels = {"present": "肯定", "absent": "否认", "uncertain": "待确认", "unspecified": "仅描述变化"}
    return "；".join(dict.fromkeys(
        f"{f['symptom']}（{f['time_scope']}、{labels[f['assertion']]}"
        + ("、" + "、".join(part for part in (f.get("degree_text"), f.get("trend_text"), f.get("duration_text"), f.get("timing_text")) if part)
           if any(f.get(k) for k in ("degree_text", "trend_text", "duration_text", "timing_text")) else "") + "）"
        for f in facts if f["subject"] == subject and not f.get("superseded_by") and not f.get("archived_episode")
    ))
