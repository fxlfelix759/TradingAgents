"""OptionTradeEvaluator: evaluates a user-specified option strategy against the pipeline thesis."""

from __future__ import annotations

import logging
from typing import Any, Dict

from tradingagents.agents.schemas import OptionEvaluationReport, render_option_evaluation
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.dataflows.interface import get_full_options_chain_for_target

logger = logging.getLogger(__name__)


def create_option_trade_evaluator(llm: Any):
    """Factory for the OptionTradeEvaluator node.

    Short-circuits with an empty dict when state["target_option"] is None,
    so the node is safe to wire unconditionally after Portfolio Manager.
    """
    structured_llm = bind_structured(llm, OptionEvaluationReport, "Option Trade Evaluator")

    def option_trade_evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        target_option = state.get("target_option")
        if not target_option:
            return {}

        # Collect all unique expiries across legs
        expiries = sorted({leg.expiration for leg in target_option.legs})
        chain_sections = []
        for expiry in expiries:
            chain_sections.append(
                get_full_options_chain_for_target(target_option.ticker, expiry, num_neighbors=2)
            )
        full_chain = "\n\n".join(chain_sections)

        instrument_context = build_instrument_context(target_option.ticker)

        legs_desc = "\n".join(
            f"  - Leg {i+1}: {leg.action.upper()} {leg.option_type.upper()} "
            f"${leg.strike} exp {leg.expiration}"
            for i, leg in enumerate(target_option.legs)
        )
        strategy_block = f"Strategy: {target_option.strategy}\nLegs:\n{legs_desc}"

        constraints_block = (
            f"\n\nUSER CONSTRAINTS / NOTES (treat as hard constraints):\n{target_option.user_notes}"
            if target_option.user_notes
            else ""
        )

        prompt = f"""You are an expert options strategist evaluating a specific option trade.

{instrument_context}

---

## User's Proposed Option Strategy

{strategy_block}{constraints_block}

---

## Portfolio Manager's Directional Verdict

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

Evaluate the proposed option strategy against the directional thesis and options chain data above.

1. **Verdict**: Is this a Strong Buy, Buy, Neutral, Avoid, or Strong Avoid?
2. **Thesis Alignment**: Does the strategy's direction and time horizon match the PM's verdict?
3. **Contract Analysis**: Assess IV level, Greeks (delta/theta/vega/gamma), bid-ask spread, and liquidity.
4. **Risk Assessment**: Daily theta decay ($), breakeven price, max loss, IV crush risk.
5. **Parameter Tweaks**: Up to 3 same-structure tweaks that improve risk/reward. Respect user constraints.
6. **Strategy Alternatives**: Up to 3 alternative structures. Suppress any that violate user constraints.
7. **Constraints**: Acknowledge each user constraint and how it shaped your suggestions.
8. **Summary**: One paragraph covering verdict, key reason, primary risk, and best modification.

Be specific — reference actual strikes, IVs, and Greeks from the chain data."""

        report_md = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_option_evaluation,
            "Option Trade Evaluator",
        )

        return {"option_evaluation_report": report_md}

    return option_trade_evaluator_node
