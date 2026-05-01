import pytest
from tradingagents.agents.schemas import (
    OptionLeg,
    TargetOption,
    ParameterTweak,
    StrategyAlternative,
    OptionEvaluationReport,
    render_option_evaluation,
)


def test_option_leg_model():
    leg = OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16")
    assert leg.strike == 130.0
    assert leg.option_type == "call"


def test_target_option_single_leg():
    opt = TargetOption(
        ticker="NVDA",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16")],
    )
    assert len(opt.legs) == 1
    assert opt.user_notes is None


def test_target_option_with_notes():
    opt = TargetOption(
        ticker="NVDA",
        strategy="call_debit_spread",
        legs=[
            OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16"),
            OptionLeg(action="sell", option_type="call", strike=140.0, expiration="2026-05-16"),
        ],
        user_notes="Max $200 per strategy.",
    )
    assert opt.user_notes == "Max $200 per strategy."
    assert len(opt.legs) == 2


def test_render_option_evaluation_contains_verdict():
    report = OptionEvaluationReport(
        verdict="Buy",
        thesis_alignment="Aligns with bullish PM verdict.",
        contract_analysis="IV is reasonable at 30%.",
        risk_assessment="Max loss is premium paid.",
        parameter_tweaks=[
            ParameterTweak(
                legs=[OptionLeg(action="buy", option_type="call", strike=128.0, expiration="2026-05-16")],
                rationale="Lower strike gives more delta.",
                estimated_cost_change="~$40 more expensive",
            )
        ],
        strategy_alternatives=[
            StrategyAlternative(
                strategy="Bull Call Spread",
                legs=[
                    OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16"),
                    OptionLeg(action="sell", option_type="call", strike=145.0, expiration="2026-05-16"),
                ],
                rationale="Reduces premium outlay.",
                tradeoff="Caps upside above $145.",
            )
        ],
        constraints_acknowledged="No constraints provided.",
        summary="This is a reasonable bullish trade.",
    )
    rendered = render_option_evaluation(report)
    assert "**Verdict**: Buy" in rendered
    assert "Thesis Alignment" in rendered
    assert "Parameter Tweaks" in rendered
    assert "Strategy Alternatives" in rendered
    assert "Summary" in rendered
