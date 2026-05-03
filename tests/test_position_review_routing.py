# tests/test_position_review_routing.py
import pytest
from langgraph.graph import END
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.agents.schemas import (
    ExistingStockPosition,
    ExistingOptionPosition,
    OptionLeg,
    TargetOption,
)


def _cl():
    return ConditionalLogic()


def test_routes_to_option_trade_evaluator_when_target_option():
    state = {
        "target_option": TargetOption(
            ticker="MSFT",
            strategy="long_call",
            legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-06-20")],
        ),
        "existing_stock_position": None,
        "existing_option_position": None,
    }
    assert _cl().route_post_pipeline(state) == "Option Trade Evaluator"


def test_routes_to_stock_reviewer_when_stock_position():
    state = {
        "target_option": None,
        "existing_stock_position": ExistingStockPosition(entry_price=500.0, shares=10.0),
        "existing_option_position": None,
    }
    assert _cl().route_post_pipeline(state) == "Stock Position Reviewer"


def test_routes_to_option_reviewer_when_option_position():
    state = {
        "target_option": None,
        "existing_stock_position": None,
        "existing_option_position": ExistingOptionPosition(
            ticker="MSFT",
            strategy="long_call",
            legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-06-20")],
            net_premium=8.50,
            contracts=2,
        ),
    }
    assert _cl().route_post_pipeline(state) == "Option Position Reviewer"


def test_routes_to_end_when_all_none():
    state = {
        "target_option": None,
        "existing_stock_position": None,
        "existing_option_position": None,
    }
    assert _cl().route_post_pipeline(state) == END
