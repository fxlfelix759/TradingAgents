# tests/test_position_reviewers.py
import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.schemas import (
    ExistingStockPosition,
    ExistingOptionPosition,
    OptionLeg,
    StockPositionReviewReport,
    OptionPositionReviewReport,
)


def _make_base_state(extra=None):
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-03",
        "final_trade_decision": "**Rating**: Buy\n**Summary**: Bullish on MSFT.",
        "investment_plan": "Buy with conviction.",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "market_report": "Bullish technicals.",
        "fundamentals_report": "Strong earnings growth.",
        "sentiment_report": "Positive social sentiment.",
        "news_report": "No adverse news.",
        "options_report": "Moderate IV at 25%.",
        "messages": [],
        "existing_stock_position": None,
        "existing_option_position": None,
    }
    if extra:
        state.update(extra)
    return state


# --- StockPositionReviewer ---

def test_stock_reviewer_noop_when_no_position(mock_llm_client):
    from tradingagents.agents.analysts.stock_position_reviewer import create_stock_position_reviewer
    node = create_stock_position_reviewer(mock_llm_client.get_llm())
    result = node(_make_base_state())
    assert result == {}


def test_stock_reviewer_calls_llm_when_position_set(mock_llm_client):
    from tradingagents.agents.analysts.stock_position_reviewer import create_stock_position_reviewer

    mock_report = StockPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$800 (-16%)",
        thesis_status="Thesis intact.",
        action_plan="Hold with stop at $390.",
        exit_triggers="Close below $390.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingStockPosition(entry_price=500.0, shares=10.0)
    state = _make_base_state({"existing_stock_position": pos})

    with patch(
        "tradingagents.agents.analysts.stock_position_reviewer.yf.Ticker"
    ) as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 420.0
        mock_ticker_cls.return_value = mock_ticker

        node = create_stock_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    mock_ticker_cls.assert_called_once_with("MSFT")
    assert "stock_position_review" in result
    assert "## Stock Position Review: Hold" in result["stock_position_review"]
    assert "-$800" in result["stock_position_review"]


# --- OptionPositionReviewer ---

def test_option_reviewer_noop_when_no_position(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer
    node = create_option_position_reviewer(mock_llm_client.get_llm())
    result = node(_make_base_state())
    assert result == {}


def test_option_reviewer_calls_llm_and_chain_when_position_set(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer

    mock_report = OptionPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$300 (-35%)",
        thesis_status="Bullish thesis valid.",
        time_risk="42 DTE, theta $12/day, breakeven $428.50.",
        roll_suggestion=None,
        exit_triggers="Close if stock drops below $400.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingOptionPosition(
        ticker="MSFT",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-06-20")],
        net_premium=8.50,
        contracts=2,
    )
    state = _make_base_state({"existing_option_position": pos})

    with patch(
        "tradingagents.agents.analysts.option_position_reviewer.get_full_options_chain_for_target",
        return_value="# Mocked chain data",
    ):
        node = create_option_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    assert "option_position_review" in result
    assert "## Option Position Review: Hold" in result["option_position_review"]


@pytest.mark.smoke
def test_option_reviewer_with_roll_recommendation(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer

    mock_report = OptionPositionReviewReport(
        recommendation="Roll",
        pnl_summary="-$200 (-25%)",
        thesis_status="Bullish but needs more time.",
        time_risk="14 DTE, theta $22/day.",
        roll_suggestion="Roll to $420 call expiring 2026-07-18 for $3.00 net debit.",
        exit_triggers="Close if stock breaks $410.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingOptionPosition(
        ticker="MSFT",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-05-17")],
        net_premium=8.50,
        contracts=1,
    )
    state = _make_base_state({"existing_option_position": pos})

    with patch(
        "tradingagents.agents.analysts.option_position_reviewer.get_full_options_chain_for_target",
        return_value="# Mocked chain data",
    ):
        node = create_option_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    assert "Roll Suggestion" in result["option_position_review"]
    assert "2026-07-18" in result["option_position_review"]
