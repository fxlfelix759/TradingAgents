import pytest
from tradingagents.agents.schemas import (
    OptionLeg,
    ExistingStockPosition,
    ExistingOptionPosition,
    StockPositionReviewReport,
    OptionPositionReviewReport,
    render_stock_position_review,
    render_option_position_review,
)


def test_existing_stock_position_fields():
    pos = ExistingStockPosition(entry_price=500.0, shares=10.0)
    assert pos.entry_price == 500.0
    assert pos.shares == 10.0


def test_existing_option_position_fields():
    pos = ExistingOptionPosition(
        ticker="MSFT",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-06-20")],
        net_premium=8.50,
        contracts=2,
    )
    assert pos.net_premium == 8.50
    assert pos.contracts == 2
    assert len(pos.legs) == 1


def test_stock_review_report_fields():
    report = StockPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$800 (-16%)",
        thesis_status="Bullish thesis intact.",
        action_plan="Hold with stop at $390.",
        exit_triggers="Close if price drops below $390 or earnings miss.",
    )
    assert report.recommendation == "Hold"


def test_option_review_report_fields_no_roll():
    report = OptionPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$300 (-35%)",
        thesis_status="Bullish thesis still valid.",
        time_risk="42 DTE, theta $12/day, breakeven $428.50.",
        roll_suggestion=None,
        exit_triggers="Close if stock drops below $400.",
    )
    assert report.roll_suggestion is None


def test_option_review_report_fields_with_roll():
    report = OptionPositionReviewReport(
        recommendation="Roll",
        pnl_summary="-$200 (-25%)",
        thesis_status="Bullish but needs more time.",
        time_risk="14 DTE, theta $22/day, breakeven $428.50.",
        roll_suggestion="Roll to $420 call expiring 2026-07-18 for $3.00 net debit.",
        exit_triggers="Close immediately if stock breaks below $410.",
    )
    assert "2026-07-18" in report.roll_suggestion


def test_render_stock_position_review_sections():
    report = StockPositionReviewReport(
        recommendation="Reduce",
        pnl_summary="-$800 (-16%)",
        thesis_status="Thesis weakened by macro headwinds.",
        action_plan="Sell 50% of position at market.",
        exit_triggers="Close remainder if price falls below $400.",
    )
    rendered = render_stock_position_review(report)
    assert "## Stock Position Review: Reduce" in rendered
    assert "P&L Summary" in rendered
    assert "Thesis Status" in rendered
    assert "Action Plan" in rendered
    assert "Exit Triggers" in rendered


def test_render_option_position_review_no_roll():
    report = OptionPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$300 (-35%)",
        thesis_status="Bullish thesis still valid.",
        time_risk="42 DTE, theta $12/day.",
        roll_suggestion=None,
        exit_triggers="Close if stock drops below $400.",
    )
    rendered = render_option_position_review(report)
    assert "## Option Position Review: Hold" in rendered
    assert "Roll Suggestion" not in rendered
    assert "Time Risk" in rendered


def test_render_option_position_review_with_roll():
    report = OptionPositionReviewReport(
        recommendation="Roll",
        pnl_summary="-$200 (-25%)",
        thesis_status="Needs more time.",
        time_risk="14 DTE, theta $22/day.",
        roll_suggestion="Roll to $420 call 2026-07-18.",
        exit_triggers="Close if stock breaks $410.",
    )
    rendered = render_option_position_review(report)
    assert "Roll Suggestion" in rendered
    assert "2026-07-18" in rendered


def test_render_option_position_review_empty_string_roll_included():
    """Empty string roll_suggestion should still render the section (not None = include)."""
    report = OptionPositionReviewReport(
        recommendation="Roll",
        pnl_summary="-$200 (-25%)",
        thesis_status="Needs more time.",
        time_risk="14 DTE, theta $22/day.",
        roll_suggestion="",
        exit_triggers="Close if stock breaks $410.",
    )
    rendered = render_option_position_review(report)
    assert "Roll Suggestion" in rendered
