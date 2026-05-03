"""StockPositionReviewer: reviews an existing stock position against the pipeline thesis."""

from __future__ import annotations

import logging
from typing import Any, Dict

import yfinance as yf

from tradingagents.agents.schemas import StockPositionReviewReport, render_stock_position_review
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext

logger = logging.getLogger(__name__)


def create_stock_position_reviewer(llm: Any):
    """Factory for the StockPositionReviewer node.

    Short-circuits with an empty dict when state["existing_stock_position"] is None.
    """
    structured_llm = bind_structured(llm, StockPositionReviewReport, "Stock Position Reviewer")

    def stock_position_reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        position = state.get("existing_stock_position")
        if not position:
            return {}

        ticker = state.get("company_of_interest", "")
        instrument_context = build_instrument_context(ticker)

        try:
            ticker_obj = yf.Ticker(ticker)
            current_price = ticker_obj.fast_info.last_price
        except Exception:
            current_price = None

        price_str = f"${current_price:.2f}" if current_price is not None else "unavailable"
        if current_price is not None:
            pnl_abs = (current_price - position.entry_price) * position.shares
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            pnl_context = f"Unrealized P&L: ${pnl_abs:+.2f} ({pnl_pct:+.1f}%)"
        else:
            pnl_context = "Current price unavailable — estimate P&L from available data."

        position_block = (
            f"Entry price: ${position.entry_price:.2f} per share\n"
            f"Shares held: {position.shares}\n"
            f"Current price: {price_str}\n"
            f"{pnl_context}"
        )

        prompt = f"""You are an expert portfolio advisor reviewing an existing stock position.

{instrument_context}

---

## Existing Position

{position_block}

---

## Portfolio Manager's Directional Verdict (fresh analysis)

{state.get("final_trade_decision", "")}

---

## Analyst Reports (Context)

**Market Report:**
{state.get("market_report", "")}

**Fundamentals Report:**
{state.get("fundamentals_report", "")}

**News Report:**
{state.get("news_report", "")}

**Sentiment Report:**
{state.get("sentiment_report", "")}

**Options Overview:**
{state.get("options_report", "")}

---

## Your Task

The user bought {ticker} at ${position.entry_price:.2f} and now holds {position.shares} shares.
The current price is {price_str}. The fresh analysis is shown above.

Given the EXISTING POSITION (not a new entry), provide:

1. **Recommendation**: Hold / Add / Reduce / Close — consider that the user is already committed
2. **P&L Summary**: Current unrealized P&L in $ and %
3. **Thesis Status**: Is the original thesis still intact? Reference specific evidence
4. **Action Plan**: Concrete steps with specific price levels and sizing
5. **Exit Triggers**: Specific price or event conditions that would change your recommendation

Focus on WHAT TO DO WITH THE EXISTING POSITION, not whether the initial entry was correct."""

        report_md = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_stock_position_review,
            "Stock Position Reviewer",
        )

        return {"stock_position_review": report_md}

    return stock_position_reviewer_node
