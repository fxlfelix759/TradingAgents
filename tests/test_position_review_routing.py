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


def test_propagate_signature_accepts_position_args():
    """propagate() accepts the new position keyword arguments without TypeError."""
    import inspect
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    sig = inspect.signature(TradingAgentsGraph.propagate)
    params = sig.parameters
    assert "existing_stock_position" in params
    assert "existing_option_position" in params


def test_graph_compiles_with_position_reviewer_nodes():
    """Graph setup compiles without error when position reviewer nodes are wired."""
    from unittest.mock import MagicMock
    from tradingagents.graph.setup import GraphSetup
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    quick_llm = MagicMock()
    quick_llm.with_structured_output.return_value = MagicMock()
    deep_llm = MagicMock()
    deep_llm.with_structured_output.return_value = MagicMock()

    @tool
    def dummy_tool(query: str) -> str:
        """dummy"""
        return query

    tool_nodes = {k: ToolNode([dummy_tool]) for k in ["market", "social", "news", "fundamentals"]}
    cl = ConditionalLogic()
    gs = GraphSetup(quick_llm, deep_llm, tool_nodes, cl)
    workflow = gs.setup_graph(["market"])
    graph = workflow.compile()
    assert graph is not None
