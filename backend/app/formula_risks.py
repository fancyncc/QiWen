"""Source-derived restrictions for provisional medication references."""
from .db import one
from .prescription import names
from .safety import normalize


def toxic_herbs(plan: dict) -> list[str]:
    result = []
    for name in names(plan):
        row = one("SELECT properties FROM herbs WHERE name IN (?,?)", (name, normalize(name)))
        if row and any(term in (row["properties"] or "") for term in ("有毒", "大毒", "小毒")):
            result.append(name)
    return result
