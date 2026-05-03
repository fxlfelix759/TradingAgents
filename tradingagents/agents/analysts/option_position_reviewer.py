"""OptionPositionReviewer: reviews an existing option position against the pipeline thesis."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import yfinance as yf

from tradingagents.agents.schemas import OptionPositionReviewReport, render_option_position_review
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.dataflows.interface import get_full_options_chain_for_target

logger = logging.getLogger(__name__)


def _fetch_leg_price(ticker: str, expiry: str, strike: float, option_type: str) -> Optional[Dict]:
    """Fetch current bid/ask/lastPrice/lastTradeDate for a specific contract.

    Returns a dict with keys: bid, ask, mid, last, last_trade_date, iv
    or None if not found.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        chain = ticker_obj.option_chain(expiry)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == float(strike)]
        if row.empty:
            return None
        r = row.iloc[0]
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        last = float(r.get("lastPrice", 0) or 0)
        iv = float(r.get("impliedVolatility", 0) or 0)
        trade_date = r.get("lastTradeDate", None)
        if hasattr(trade_date, "date"):
            trade_date_str = str(trade_date.date())
        elif trade_date is not None:
            trade_date_str = str(trade_date)[:10]
        else:
            trade_date_str = "unknown"
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last
        return {
            "bid": bid, "ask": ask, "mid": mid, "last": last,
            "last_trade_date": trade_date_str, "iv": round(iv * 100, 1),
        }
    except Exception as exc:
        logger.warning("Could not fetch leg price for %s %s %s %s: %s",
                       ticker, expiry, strike, option_type, exc)
        return None


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

        # Fetch current stock price
        try:
            ticker_obj = yf.Ticker(position.ticker.upper())
            current_price = ticker_obj.fast_info.last_price
        except Exception:
            current_price = None

        # Fetch each leg's current market price directly — do NOT rely on LLM
        # to find the correct row in a large chain table.
        today_str = datetime.today().strftime("%Y-%m-%d")
        leg_price_lines = []
        for i, leg in enumerate(position.legs):
            data = _fetch_leg_price(position.ticker, leg.expiration, leg.strike, leg.option_type)
            if data:
                freshness = "(fresh)" if data["last_trade_date"] >= today_str[:8] else f"(last traded {data['last_trade_date']} — may be stale)"
                leg_price_lines.append(
                    f"  Leg {i+1} ({leg.action.upper()} {leg.option_type.upper()} "
                    f"${leg.strike} exp {leg.expiration}): "
                    f"bid={data['bid']} ask={data['ask']} mid={data['mid']} "
                    f"last={data['last']} IV={data['iv']}% {freshness}"
                )
            else:
                leg_price_lines.append(
                    f"  Leg {i+1} ({leg.action.upper()} {leg.option_type.upper()} "
                    f"${leg.strike} exp {leg.expiration}): price data unavailable — estimate from chain below"
                )

        current_prices_block = "\n".join(leg_price_lines)

        # Full chain for Greeks / context (LLM uses this for theta, IV skew, roll targets)
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

        stock_price_str = f"${current_price:.2f}" if current_price is not None else "unavailable"

        prompt = f"""You are an expert options strategist reviewing an existing option position.

{instrument_context}

---

## Existing Option Position

{position_block}

---

## Current Market Prices — USE THESE FOR P&L (authoritative, pre-extracted)

Underlying ({position.ticker}): {stock_price_str}

{current_prices_block}

Use the `mid` price as the current mark for each leg. If a leg shows "may be stale",
use the `last` price and note the uncertainty. Do NOT look up these prices in the chain
table below — these values are already extracted for you.

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

## Full Options Chain (use for Greeks, IV skew, and roll target identification only)

{full_chain}

---

## Your Task

The user ALREADY OWNS this option position. Review it given the current market conditions.

1. **Recommendation**: Hold / Close Now / Roll / Partial Close / Hedge
2. **P&L Summary**: Use the pre-extracted mid prices above to compute current value vs. cost basis in $ and %
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
