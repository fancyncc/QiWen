from __future__ import annotations

import json
import re
from pathlib import Path

# Traditional compatibility rules. Conservative flags, not clinical judgments.
# Shenzhen municipal standard DB4403/T 142-2021, Appendix C:
# https://wjw.sz.gov.cn/attachment/0/758/758943/8708423.pdf
GROUPS = {
    "十八反·甘草": (["甘草"], ["甘遂", "大戟", "海藻", "芫花"]),
    "十八反·乌头": (["乌头", "川乌", "草乌", "附子"], ["半夏", "瓜蒌", "贝母", "白蔹", "白及"]),
    "十八反·藜芦": (["藜芦"], ["人参", "沙参", "丹参", "玄参", "苦参", "细辛", "芍药"]),
}
PAIRS = [
    ("硫黄", "朴硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"),
    ("巴豆", "牵牛"), ("丁香", "郁金"), ("川乌", "犀角"),
    ("草乌", "犀角"), ("牙硝", "三棱"), ("官桂", "石脂"),
    ("人参", "五灵脂"),
]
ALIASES = {"炙甘草": "甘草", "生甘草": "甘草", "法半夏": "半夏", "姜半夏": "半夏",
           "清半夏": "半夏", "制附子": "附子", "生川乌": "川乌", "生草乌": "草乌",
           "浙贝母": "贝母", "川贝母": "贝母", "瓜蒌皮": "瓜蒌", "全瓜蒌": "瓜蒌",
           "平贝母": "贝母", "伊贝母": "贝母", "湖北贝母": "贝母",
           "瓜蒌子": "瓜蒌", "天花粉": "瓜蒌",
           "白芍": "芍药", "赤芍": "芍药", "赤石脂": "石脂", "党参": "党参"}
PREFIXES = ("生", "制", "炒", "炙", "酒", "蜜", "盐", "醋")
KNOWN = set(ALIASES.values()) | {
    "酸枣仁", "茯苓", "知母", "川芎", "桂枝", "生姜", "大枣",
    "黄芪", "白术", "当归", "熟地黄", "柴胡", "黄芩", "陈皮",
    "枳实", "桔梗", "麦冬", "五味子", "泽泻", "牡丹皮", "山药",
}
for left, right in GROUPS.values():
    KNOWN.update(left)
    KNOWN.update(right)
for left, right in PAIRS:
    KNOWN.update((left, right))

# The bundled curated seed is the only additional vocabulary accepted by the
# deterministic checker. A model cannot add an arbitrary name to this set.
_seed_path = Path(__file__).resolve().parents[1] / "resources" / "knowledge_seed.json"
for _herb in json.loads(_seed_path.read_text(encoding="utf-8"))["herbs"]:
    KNOWN.add(_herb["name"])
    for _alias in _herb["aliases"]:
        ALIASES.setdefault(_alias, _herb["name"])


def normalize(name: str) -> str:
    name = re.sub(r"[\s\d克gG]+", "", name).strip()
    if name in ALIASES:
        return ALIASES[name]
    for prefix in PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix) + 1:
            name = name[len(prefix):]
            break
    return ALIASES.get(name, name)


def check_herbs(herbs: list[str] | None) -> list[str]:
    if not herbs or any(not isinstance(item, str) or not item.strip() for item in herbs):
        return ["无法解析完整药材清单，不能判定配伍安全"]
    names = {normalize(item) for item in herbs}
    errors = []
    unknown = names - KNOWN
    if unknown:
        errors.append("含未收录药材：" + "、".join(sorted(unknown)) + "；需人工复核")
    for label, left, right in ((label, *pair) for label, pair in GROUPS.items()):
        for a in left:
            for b in right:
                if a in names and b in names:
                    errors.append(f"{label}：{a}与{b}")
    for a, b in PAIRS:
        if a in names and b in names:
            errors.append(f"十九畏：{a}与{b}")
    return errors
