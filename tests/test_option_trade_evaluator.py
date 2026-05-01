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


from unittest.mock import patch, MagicMock
import pandas as pd
from tradingagents.dataflows.y_finance import get_full_options_chain_for_target


def _make_chain_df():
    """Minimal options chain DataFrame matching yfinance structure."""
    return pd.DataFrame({
        "contractSymbol": ["NVDA260516C00130000"],
        "lastTradeDate": [pd.Timestamp("2026-04-28")],
        "strike": [130.0],
        "lastPrice": [5.50],
        "bid": [5.40],
        "ask": [5.60],
        "change": [0.10],
        "percentChange": [1.85],
        "volume": [500],
        "openInterest": [1200],
        "impliedVolatility": [0.35],
        "inTheMoney": [False],
        "contractSize": ["REGULAR"],
        "currency": ["USD"],
    })


def test_full_options_chain_returns_string():
    """get_full_options_chain_for_target returns a non-empty markdown string."""
    mock_ticker = MagicMock()
    mock_ticker.options = ["2026-05-09", "2026-05-16", "2026-05-23"]
    mock_ticker.fast_info.last_price = 128.50
    chain = MagicMock()
    chain.calls = _make_chain_df()
    chain.puts = _make_chain_df()
    mock_ticker.option_chain.return_value = chain

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker), \
         patch("tradingagents.dataflows.y_finance.get_risk_free_rate", return_value=0.05):
        result = get_full_options_chain_for_target("NVDA", "2026-05-16", num_neighbors=1)

    assert isinstance(result, str)
    assert len(result) > 100
    assert "2026-05-16" in result


def test_full_options_chain_highlights_target_expiry():
    """The function labels the target expiry with TARGET EXPIRY."""
    mock_ticker = MagicMock()
    mock_ticker.options = ["2026-05-16", "2026-05-23"]
    mock_ticker.fast_info.last_price = 128.50
    chain = MagicMock()
    chain.calls = _make_chain_df()
    chain.puts = _make_chain_df()
    mock_ticker.option_chain.return_value = chain

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker), \
         patch("tradingagents.dataflows.y_finance.get_risk_free_rate", return_value=0.05):
        result = get_full_options_chain_for_target("NVDA", "2026-05-16")

    assert "TARGET EXPIRY" in result


def test_full_options_chain_no_options_returns_message():
    """Returns a clear message when no options are listed for the ticker."""
    mock_ticker = MagicMock()
    mock_ticker.options = []

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
        result = get_full_options_chain_for_target("FAKE", "2026-05-16")

    assert "No options data available" in result
