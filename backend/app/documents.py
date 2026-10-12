"""Keep accepted source text intact; bound only the model's supporting excerpt."""
from __future__ import annotations

import re

MAX_FILE_TEXT_CHARS = 20_000


def model_excerpt(text: str, limit: int = 1600) -> str:
    if len(text) <= limit:
        return text
    # Select matching clauses across the entire attachment, including its tail.
    from .safety_context import extract_facts
    from .symptoms import extract
    facts = extract_facts(text, kind="attachment") + extract(text, kind="attachment")
    quotes = list(dict.fromkeys(fact["source"]["quote"] for fact in facts))
    selected = "\n".join(quotes) or text
    return selected[:limit] + "\n[附件模型摘录已缩短；完整文本另行参与事实提取与安全预审，摘录不是完整病历。]"


def normalize_document(text: str) -> str:
    # Newlines delimit patient statements and must not be flattened into one clause.
    return re.sub(r"[^\S\n]+", " ", text).strip()
