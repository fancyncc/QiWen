"""Shared query construction; evaluation labels never enter this module."""
from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from .retrieval_terms import NEGATIVE_SWEAT, POSITIVE_SWEAT, normalize_search, TERM_VERSION

STOPWORDS = frozenset("什么 如何 怎样 怎么 为什么 是否 哪些 哪个 多少 能否 可以 需要 请问 请 介绍 解释 概括 描述 理解 意思 含义 关系 区别 古籍 原文 出处 资料 文献 所说 关于 根据 里面 提到 说的 一下 一个 这个 那个 以及 应该 已经 最近 今天 三天 三周 典型 表现 说明 主治 形状 形态 变化 提纲".split())
CONCEPTS = "阴阳 阴虚 阳虚 气虚 血虚 阴盛 阳盛 气血 气滞 血瘀 津液 脏腑 经络 六经 卫气营血 五行 辨证 天癸 女子 男子 春三月 夏三月 秋三月 冬三月 阳气 阴气 正气 邪气 天日 生长 胃家实".split()
DOMAIN_SINGLE_TERMS = frozenset("怒喜忧思悲恐惊")
CLINICAL_SINGLE_TERMS = frozenset("喘渴咳痞哕呕")
_tokenizer_lock = threading.Lock()


def unique(items):
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


@lru_cache(maxsize=1)
def tokenizer():
    import jieba
    jieba.setLogLevel(logging.ERROR)
    instance = jieba.Tokenizer()
    words = list(CONCEPTS)
    resources = Path(__file__).resolve().parents[1] / "resources"
    symptoms = json.loads((resources / "symptom_terms.json").read_text(encoding="utf-8"))
    words.extend(alias for entry in symptoms["entries"] for alias in entry["aliases"])
    seed = json.loads((resources / "knowledge_seed.json").read_text(encoding="utf-8"))
    for category in ("herbs", "formulas"):
        for entry in seed[category]:
            words.extend([entry["name"], *entry.get("aliases", [])])
    instance.initialize()
    for word in unique(words):
        instance.add_word(word, freq=100000)
    return instance


def lexical_terms(question: str) -> list[str]:
    # Preserve named sources and quoted concepts, then segment ordinary prose.
    question = normalize_search(re.sub(r"《[^》]{2,30}》", " ", question))
    quoted = re.findall(r"[“\"]([^”\"]{2,30})[”\"]", question)
    with _tokenizer_lock:
        tokens = list(tokenizer().cut(question, HMM=True))
    # Product domain vocabulary, independent of benchmark labels.
    compounds = re.findall(r"[浮沉迟数滑涩弦紧缓洪微细芤革牢伏动促结代散弱濡长短大实虚阴阳肝心脾肺肾]脉|寸关尺|[太少厥][阴阳]病|[无有][汗痰]|[\u3400-\u9fff]{1,2}疸", question)
    protected = [m.group() for m in NEGATIVE_SWEAT.finditer(question)]
    # Enumeration tails must not become artificial tokens such as 恐等.
    emotions = [char for char in question if char in DOMAIN_SINGLE_TERMS] if re.search(r"情志|气机|[怒喜忧思悲恐惊][、，,]", question) else []
    tokens = [term.removesuffix("等") if term.removesuffix("等") in DOMAIN_SINGLE_TERMS else term for term in tokens]
    return unique([term for term in [*quoted, *protected, *compounds, *emotions, *tokens]
                   if term not in STOPWORDS and (len(term) >= 2 or term in emotions or term in CLINICAL_SINGLE_TERMS)
                   and re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9·-]+", term)
                   and not re.fullmatch(r"[0-9]+", term)])[:32]


def make_plan(keyword_query, semantic_query, terms, *, primary=None, expansions=None,
              unmapped=None, attempt=1):
    terms = unique(terms)
    return {"keyword_query": keyword_query, "semantic_query": semantic_query,
            "primary": unique(primary if primary is not None else terms),
            "lexical_terms": terms, "expansions": expansions or [], "unmapped": unmapped or [],
            "attempt": attempt, "catalog_version": 1, "query_version": 4, "term_version": TERM_VERSION}


def knowledge_plan(question: str) -> dict:
    terms = lexical_terms(question)
    # BM25 can tokenize the full question for recall; literal admission uses words.
    plan = make_plan(" ".join(terms), normalize_search(question), terms)
    plan["source_hints"] = re.findall(r"《([^》]{2,30})》", question)
    normalized = normalize_search(question)
    negative = NEGATIVE_SWEAT.search(normalized)
    positive = POSITIVE_SWEAT.search(NEGATIVE_SWEAT.sub("", normalized))
    if negative and positive:
        plan["action_polarity"] = "contrast_sweat"
    elif negative:
        plan["action_polarity"] = "negative_sweat"
        plan["lexical_terms"] = unique([*plan["lexical_terms"], "不可发其汗", "不可汗", "勿汗"])
    elif positive:
        plan["action_polarity"] = "positive_sweat"
    if re.search(r"何方|什么方|哪[个种些]?方|用.*方|何.*汤|什么汤|何.*丸|什么丸", question):
        plan["requested_property"] = "formula"
        # Preserve explicitly paired signs; neither one alone answers the query.
        if "无汗" in normalized and "喘" in normalized:
            plan["required_term_groups"] = [["无汗"], ["喘"]]
    if "关系" in question:
        groups = [[term] for term in CONCEPTS if term in normalized]
        if "天日" in normalized:
            groups = [["天日", "天与日", "天與日"] if group == ["天日"] else group for group in groups]
        if len(groups) >= 2:
            plan["required_term_groups"] = groups
    if re.search(r"提纲|何谓|为何称|所谓", question):
        explicit_concepts = [[term] for term in CONCEPTS if term in normalized]
        if explicit_concepts:
            plan["required_term_groups"] = explicit_concepts
    if "提纲" in question:
        diseases = re.findall(r"太阳病|阳明病|少阳病|太阴病|少阴病|厥阴病", normalized)
        if diseases:
            plan["requested_property"] = "disease_definition"
            plan["required_term_groups"] = [*plan.get("required_term_groups", []),
                *[[disease, disease.removesuffix("病") + "之为病"] for disease in diseases]]
    if re.search(r"主治|功效|性味", question):
        seed = json.loads((Path(__file__).resolve().parents[1] / "resources" / "knowledge_seed.json").read_text(encoding="utf-8"))
        names = [entry["name"] for entry in seed["herbs"] if normalize_search(entry["name"]) in normalized]
        if names:
            plan["required_entities"] = names
    # A focused lexical channel preserves relation/action phrases without labels.
    focused = unique([*(term for group in plan.get("required_term_groups", []) for term in group),
                      *plan.get("required_entities", []),
                      *(["不可发汗", "不可发其汗", "勿汗"] if plan.get("action_polarity") == "negative_sweat" else [])])
    if focused:
        plan["recall_queries"] = [{"kind": "original", "keyword_query": plan["keyword_query"], "semantic_query": plan["semantic_query"]},
                                  {"kind": "explicit_content", "keyword_query": " ".join(focused), "semantic_query": plan["semantic_query"]}]
    plan["evidence_kind"] = ("named_classics" if plan["source_hints"] else
                             "concept_definition" if re.search(r"什么|含义|意思|定义|概念|解释|区别", question)
                             else "general_classics")
    return plan
