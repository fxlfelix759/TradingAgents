"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        options_report = state.get("options_report", "")
        options_line = (
            f"- Options chain analysis (OI concentration, IV, Max Pain, P/C ratio):\n{options_report}\n"
            if options_report
            else ""
        )

        change_report = state.get("change_report", "")
        change_line = (
            f"- Change since last analysis (from the Change Analyst):\n{change_report}\n"
            if change_report
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{options_line}{change_line}{lessons_line}
**Risk Analysts Debate History:**
{history}

---

**Required output sections:**

1. **Overall rating** — one of the five tiers above.
2. **Executive summary** — 2-4 sentences covering entry strategy, sizing, and key risk levels.
3. **Investment thesis** — detailed reasoning anchored in specific evidence.
4. **Time-horizon recommendations** — provide a separate Buy/Hold/Sell call, rationale, optional price target, and key catalysts for each of:
   - **Short term (1 week)**: focus on near-term price action, upcoming events, and technical levels.
   - **Medium term (1 month)**: focus on earnings, macro data, and intermediate trend.
   - **Long term (6 months)**: focus on fundamental valuation, secular trends, and strategic positioning.
5. **Options strategy** — use the options chain data (OI walls, Max Pain, IV level) to recommend a concrete strategy. Align strike selection with the OI concentration levels and set expiry consistent with the relevant time horizon. Specify strategy name, rationale, suggested expiry, strike guidance, and plain-language risk/reward.

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
