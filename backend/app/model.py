from __future__ import annotations

import json

import httpx

from .config import API_BASE, API_KEY, MODEL_MODE, MODEL_NAME


def complete(system: str, user: str) -> str:
    if MODEL_MODE != "openai_compatible":
        raise RuntimeError("当前未启用真实模型")
    if not API_BASE or not API_KEY or not MODEL_NAME:
        raise RuntimeError("真实模型配置不完整")
    url = API_BASE.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL_NAME,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def complete_json(system: str, user: str) -> dict:
    raw = complete(system + " 只输出 JSON 对象，不要 Markdown。", user)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("模型未返回可解析的结构化结果") from exc

