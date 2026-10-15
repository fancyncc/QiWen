import asyncio
import copy
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.db import one
from app.formula_selection import matches_source, repair_source_choice
from app.main import app


@pytest.mark.parametrize("selected", ["酸枣仁汤", ""])
def test_repair_requires_a_real_model_choice_allows_none_and_preserves_sources(selected):
    with TestClient(app):
        row = one("SELECT name,structured_herbs,indication,source FROM formulas WHERE name='酸枣仁汤'")
        row["herbs"] = json.loads(row.pop("structured_herbs"))
        before = copy.deepcopy(row)
        calls = []
        async def generate(system, user, **kwargs):
            calls.append(kwargs["schema"])
            assert set(kwargs["schema"].model_fields["selected_name"].annotation.__args__) == {"", "酸枣仁汤"}
            assert "实际检索来源" in user and "不适合或不足以支持" in system
            return {"selected_name": selected, "information_gap": "optional", "rationale": "实际依据和未确认条件。",
                    "safety_notes": ["个人用药需核对，剂量留空。"]}
        result = asyncio.run(repair_source_choice({"message": "失眠三天"}, {"name": "自拟方案", "herbs": []}, [row], generate))
        assert len(calls) == 1 and row == before
        if selected:
            assert result["name"] == selected and matches_source(result, [row])
            assert all(item["dosage"]["min"] is None and not item["preparation"] for item in result["herbs"])
        else:
            assert not result["herbs"] and result["information_gap"] == "evidence"


def test_no_source_or_unanswered_clinical_question_does_not_force_a_choice():
    async def forbidden(*args, **kwargs):
        raise AssertionError("No source or necessary unanswered question")
    plan = {"name": "无候选", "herbs": []}
    assert asyncio.run(repair_source_choice({}, plan, [], forbidden)) is plan
    plan["information_gap"] = "patient"
    assert asyncio.run(repair_source_choice({}, plan, [{"name": "任意条目"}], forbidden)) is plan


def test_fabricated_choice_is_rejected_by_enum():
    async def fabricated(*args, **kwargs):
        return {"selected_name": "不存在的自拟方", "information_gap": "optional", "rationale": "无来源"}
    with pytest.raises(ValidationError):
        asyncio.run(repair_source_choice({}, {"name": "自拟方", "herbs": []},
                    [{"name": "实际条目", "herbs": []}], fabricated))


def test_toxic_source_is_not_a_provisional_choice_even_when_composition_matches():
    with TestClient(app):
        rows = []
        for name in ("四逆汤", "理中丸"):
            row = one("SELECT name,structured_herbs,indication,source,contraindications FROM formulas WHERE name=?", (name,))
            row["herbs"] = json.loads(row.pop("structured_herbs"))
            rows.append(row)
        async def choose(system, user, **kwargs):
            assert "四逆汤" not in kwargs["schema"].model_fields["selected_name"].annotation.__args__
            assert "理中丸" in kwargs["schema"].model_fields["selected_name"].annotation.__args__
            assert "contraindications" in user
            return {"selected_name": "理中丸", "information_gap": "optional", "rationale": "依实际依据重新比较，未确认条件仍保留。"}
        result = asyncio.run(repair_source_choice({}, rows[0], rows, choose))
        assert result["name"] == "理中丸" and matches_source(result, rows)
        async def forbidden(*args, **kwargs):
            raise AssertionError("No eligible reference source")
        limited = asyncio.run(repair_source_choice({}, rows[0], rows[:1], forbidden))
        assert not limited["herbs"] and "有毒药材" in limited["rationale"]
