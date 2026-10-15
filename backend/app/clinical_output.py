"""Clinical-facing explanations shared by both consultation modes.

These helpers select stored clinical prose; they never infer a syndrome or
promote a prescription. Numerical references live in a separate source layer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from .prescription_reference import dose_free


class ClinicalAssessmentFields(BaseModel):
    clinical_basis: list[str] = Field(default_factory=list, max_length=6,
        description="逐项解释已确认症状与病机的联系，不只罗列症状，不把未报告当作否认")
    differentials: list[str] = Field(default_factory=list, max_length=3,
        description="主要鉴别证候、支持或欠缺的实际依据，未问到的资料保持未知")
    treatment_principle: str = Field(default="", max_length=300, description="与初步辨证对应的治法，不含药物剂量或服法")


def assessment_text(artifact: dict) -> str:
    parts = [artifact.get("conclusion", "")]
    for title, values in (("辨证依据与病机", artifact.get("clinical_basis", [])),
                          ("鉴别与判断条件", artifact.get("differentials", []))):
        values = [dose_free(value) for value in values]
        if any(values):
            parts.extend(["### " + title, "\n".join("- " + value for value in values if value)])
    principle = dose_free(artifact.get("treatment_principle", ""))
    if principle:
        parts.extend(["### 治法", principle])
    return "\n\n".join(part for part in parts if part)


ASSESSMENT_STYLE = (
    "conclusion用一段话概括初步证候、主次倾向和病机。已能初步分析时必须填写clinical_basis、differentials、treatment_principle："
    "clinical_basis逐项解释实际症状为何支持病机（资料允许时至少两组），不能只说这些症状支持某证；differentials解释一至两个鉴别方向及区分条件；"
    "treatment_principle只写治法原则，不写方名和药物，选方由后续方剂专家处理；具体局限写uncertainties。未询问的热象等不能声称未见或不存在，只能写尚未确认。"
    "不将未报告的舌质、脉象、体质或过敏史当作事实，不以缺少脉象为由抹去已有辨证。"
    "已有舌脉描述必须承接；仅缺细项时具体指出缺什么，不能笼统声称缺乏舌象脉象。"
    "不把检索命中、程序状态、专家运行情况作为辨证结论，不输出内部思维链。"
    "古籍原句仅在实际取得并核对的证据支持时引用，否则说明临床依据即可。"
)
FORMULA_STYLE = (
    "选方说明须写治法、该方与已确认症状的对应、与其他候选的区别和适用条件；"
    "加减需说明触发症状、药物及理由，未经来源和安全审核的加减只能作为待讨论方向，不能写成已完成替换。"
    "组成与剂量由独立处方卡承载，comparison/rationale 不含数值剂量或服法。"
    "避免只写候选生成完成、审查通过、信息不足或需复核。"
    "若该方只覆盖辨证的一部分，明确不能覆盖的环节，不得声称全面对证。"
    "含有毒药材的方不能通过本轮未经临床批准的处方参考提供，不得因为含有怕冷等单个共有表现就选用急危重症方。"
)


def formula_explanation(state: dict) -> str:
    return dose_free(state.get("formula_reasoning", "")) or dose_free(
        state.get("agent_artifacts", {}).get("formula", {}).get("comparison", ""))


def herb_explanation(state: dict) -> str:
    artifact = state.get("agent_artifacts", {}).get("herb", {}) or state.get("herb_review", {})
    analysis = dose_free(artifact.get("clinical_review", ""))
    actions = [dose_free(value) for value in artifact.get("required_actions", [])]
    problems = [dose_free(value) for value in artifact.get("problems", [])]
    parts = [analysis, *("需处理：" + value for value in dict.fromkeys(problems) if value),
             *("建议：" + value for value in dict.fromkeys(actions) if value)]
    return "\n\n".join(value for value in parts if value)


def selection_advice(state: dict, reference: dict | None) -> str:
    explanation = formula_explanation(state)
    if reference:
        return ("**参考方：" + reference["name"] + "**\n\n"
                + (explanation or "方剂条目主治：" + reference.get("indication", "")
                   + "。仍需结合本次辨证确认适用条件。")
                + "\n\n具体组成与用量依据见下方独立处方参考。")
    unresolved = [value for value in state.get("safety_errors", []) if value not in state.get("manual_review_reasons", [])]
    hidden = "候选涉及尚未解决的风险，药材清单已隐藏。" if unresolved else ""
    return (explanation + "\n\n" if explanation else "") + hidden + (
        "当前尚不能形成可执行的个人处方。具体药材或剂量涉及尚未解决的安全或来源问题，需医师核对；"
        "上述辨证和治法可供面诊时参考。")
