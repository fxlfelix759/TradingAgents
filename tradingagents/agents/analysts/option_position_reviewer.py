"""OptionPositionReviewer: reviews an existing option position against the pipeline thesis."""

from __future__ import annotations

import logging
from typing import Any, Dict

from tradingagents.agents.schemas import OptionPositionReviewReport, render_option_position_review
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.dataflows.interface import get_full_options_chain_for_target

logger = logging.getLogger(__name__)


def create_option_position_reviewer(llm: Any):
    """Factory for the OptionPositionReviewer node.

    Short-circuits with an empty dict when state["existing_option_position"] is None.
    """
    structured_llm = bind_structured(llm, OptionPositionReviewReport, "Option Position Reviewer")

    def option_position_reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        position = state.get("existing_option_position")
        if position is None:
            return {}

        instrument_context = build_instrument_context(position.ticker)

        expiries = sorted({leg.expiration for leg in position.legs})
        chain_sections = []
        for expiry in expiries:
            chain_sections.append(
                get_full_options_chain_for_target(position.ticker, expiry, num_neighbors=2)
            )
        full_chain = "\n\n".join(chain_sections)

        legs_desc = "\n".join(
            f"  - Leg {i+1}: {leg.action.upper()} {leg.option_type.upper()} "
            f"${leg.strike} exp {leg.expiration}"
            for i, leg in enumerate(position.legs)
        )

        cost_direction = "net debit paid" if position.net_premium >= 0 else "net credit received"
        position_block = (
            f"Strategy: {position.strategy}\n"
            f"Legs:\n{legs_desc}\n"
            f"Cost basis: ${abs(position.net_premium):.2f}/contract ({cost_direction})\n"
            f"Contracts held: {position.contracts}\n"
            f"Total cost basis: ${abs(position.net_premium) * position.contracts:.2f}"
        )

        prompt = f"""You are an expert options strategist reviewing an existing option position.

{instrument_context}

---

## Existing Option Position

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

**Options Overview (summary):**
{state.get("options_report", "")}

---

## Full Options Chain Data (with Greeks)

{full_chain}

---

## Your Task

The user ALREADY OWNS this option position. Review it given the current market conditions above.

**IMPORTANT — Data freshness:** The chain data above includes a `lastTradeDate` column.
- If `lastTradeDate` for the target strike is today or very recent: use the `bid`/`ask` midpoint as the current mark.
- If `lastTradeDate` is stale (more than 1 trading day old): the bid/ask may not reflect the current market. Use `lastPrice` combined with the current stock price and IV to estimate fair value, and flag the staleness explicitly in your P&L Summary.
- Never report a mark that is inconsistent with the current underlying price shown at the top of the chain data.

1. **Recommendation**: Hold / Close Now / Roll / Partial Close / Hedge
2. **P&L Summary**: Current mark vs. cost basis in $ and % — note if chain data appears stale
3. **Thesis Status**: Is the original directional thesis still intact? Has IV changed significantly?
4. **Time Risk**: DTE remaining, daily theta burn ($), breakeven at expiry, upcoming event risk
5. **Roll Suggestion**: If rolling, specify target strike/expiry and estimated roll cost (or null)
6. **Exit Triggers**: Specific price, DTE, or P&L% conditions that would change your recommendation

Be specific — reference actual strikes, IVs, and Greeks from the chain data."""

        report_md = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_option_position_review,
            "Option Position Reviewer",
        )

        return {"option_position_review": report_md}

    return option_position_reviewer_node
