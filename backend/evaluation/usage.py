"""Count billable attempts without interpreting missing usage as zero."""
from __future__ import annotations
import math


def usage_summary(calls: list[dict], prices: dict | None = None) -> dict:
    attempts, known, tokens, cost, priced = 0, 0, 0, 0.0, 0
    for call in calls:
        # Local ranking has no provider token usage; report it separately.
        if call.get("role") == "cross_encoder":
            continue
        rows = call.get("attempts") or [call]
        for row in rows:
            attempts += 1
            pair = (row.get("input_tokens"), row.get("output_tokens"))
            valid = all(type(v) is int and v >= 0 for v in pair)
            if not valid:
                continue
            known += 1
            tokens += sum(pair)
            price = (prices or {}).get(call.get("model"))
            if isinstance(price, dict) and all(type(price.get(k)) in (int, float) and math.isfinite(price[k]) and price[k] >= 0 for k in ("input_per_million", "output_per_million")):
                cost += (pair[0] * price["input_per_million"] + pair[1] * price["output_per_million"]) / 1_000_000
                priced += 1
    return {"attempts": attempts, "known_attempts": known, "usage_coverage": known / attempts if attempts else None,
            "known_tokens": tokens, "total_tokens": tokens if attempts and known == attempts else None,
            "known_provider_cost": cost if priced else None,
            "provider_cost": cost if attempts and priced == attempts else None,
            "cost_scope": "provider_attempts_including_failed_retries_excludes_judge_and_infrastructure"}
