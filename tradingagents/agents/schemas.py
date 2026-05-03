"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager — time-horizon and options sub-models
# ---------------------------------------------------------------------------


class TimeHorizonRecommendation(BaseModel):
    """Directional recommendation for a specific time horizon."""

    action: TraderAction = Field(
        description="Directional action for this horizon: Buy, Hold, or Sell.",
    )
    rationale: str = Field(
        description=(
            "Two to three sentences explaining the key drivers and risks "
            "for this specific time frame."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional price target for this horizon in the instrument's quote currency.",
    )
    key_catalysts: str = Field(
        description=(
            "Comma-separated list of the most important upcoming catalysts "
            "(events, data releases, technical levels) relevant to this horizon."
        ),
    )


class OptionsRecommendation(BaseModel):
    """Options strategy recommendation tied to the near/medium-term outlook."""

    strategy: str = Field(
        description=(
            "Name of the recommended options strategy, e.g. 'Long Call', "
            "'Bull Put Spread', 'Covered Call', 'Long Straddle', 'Cash-Secured Put'."
        ),
    )
    rationale: str = Field(
        description=(
            "Why this options strategy fits the current outlook, implied-volatility "
            "environment, and risk/reward preference. Two to four sentences."
        ),
    )
    suggested_expiry: str = Field(
        description=(
            "Recommended expiry window, e.g. '7-14 DTE for the short-term play, "
            "30-45 DTE for the medium-term play'."
        ),
    )
    strike_guidance: str = Field(
        description=(
            "Strike selection guidance relative to the current price, "
            "e.g. 'ATM call' or '5% OTM put spread with the short leg at support'."
        ),
    )
    risk_reward: str = Field(
        description=(
            "Plain-language risk/reward summary, e.g. "
            "'Max loss: premium paid (~$150); target: 2–3× if thesis plays out within 2 weeks'."
        ),
    )


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional overall target price in the instrument's quote currency.",
    )
    short_term: TimeHorizonRecommendation = Field(
        description=(
            "1-week outlook: a concrete Buy/Hold/Sell call with rationale, "
            "an optional price target, and the key catalysts for the next 7 days."
        ),
    )
    medium_term: TimeHorizonRecommendation = Field(
        description=(
            "1-month outlook: a concrete Buy/Hold/Sell call with rationale, "
            "an optional price target, and the key catalysts for the next 30 days."
        ),
    )
    long_term: TimeHorizonRecommendation = Field(
        description=(
            "6-month outlook: a concrete Buy/Hold/Sell call with rationale, "
            "an optional price target, and the key catalysts for the next 6 months."
        ),
    )
    options_analysis: OptionsRecommendation = Field(
        description=(
            "An options strategy recommendation consistent with the overall "
            "directional view, including strategy name, rationale, expiry window, "
            "strike guidance, and risk/reward."
        ),
    )


def _render_horizon(label: str, h: TimeHorizonRecommendation) -> list[str]:
    parts = [f"### {label}", "", f"**Action**: {h.action.value}", "", f"**Rationale**: {h.rationale}"]
    if h.price_target is not None:
        parts.extend(["", f"**Price Target**: {h.price_target}"])
    parts.extend(["", f"**Key Catalysts**: {h.key_catalysts}"])
    return parts


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])

    # Time-horizon recommendations
    parts.extend(["", "---", "", "## Trading Recommendations by Time Horizon", ""])
    parts.extend(_render_horizon("Short Term (1 Week)", decision.short_term))
    parts.extend([""])
    parts.extend(_render_horizon("Medium Term (1 Month)", decision.medium_term))
    parts.extend([""])
    parts.extend(_render_horizon("Long Term (6 Months)", decision.long_term))

    # Options analysis
    opt = decision.options_analysis
    parts.extend([
        "",
        "---",
        "",
        "## Options Analysis & Recommendation",
        "",
        f"**Strategy**: {opt.strategy}",
        "",
        f"**Rationale**: {opt.rationale}",
        "",
        f"**Suggested Expiry**: {opt.suggested_expiry}",
        "",
        f"**Strike Guidance**: {opt.strike_guidance}",
        "",
        f"**Risk / Reward**: {opt.risk_reward}",
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Option Trade Evaluator — input models and output schema
# ---------------------------------------------------------------------------


class OptionLeg(BaseModel):
    """A single leg in an option strategy."""

    action: Literal["buy", "sell"] = Field(
        description="Whether this leg is a long (buy) or short (sell) position."
    )
    option_type: Literal["call", "put"] = Field(
        description="Call or put."
    )
    strike: float = Field(
        description="Strike price of this leg."
    )
    expiration: str = Field(
        description="Expiration date in YYYY-MM-DD format."
    )


class TargetOption(BaseModel):
    """User-specified option strategy to evaluate."""

    ticker: str = Field(description="Underlying ticker symbol.")
    strategy: str = Field(
        description=(
            "Strategy identifier, e.g. 'long_call', 'call_debit_spread', 'iron_condor'."
        )
    )
    legs: List[OptionLeg] = Field(description="All legs of the strategy in order.")
    user_notes: Optional[str] = Field(
        default=None,
        description="User-supplied constraints or rationale, e.g. budget limits.",
    )


class ParameterTweak(BaseModel):
    """Same strategy structure, better parameters."""

    legs: List[OptionLeg] = Field(description="Full revised leg set.")
    rationale: str = Field(description="Why these parameters are better.")
    estimated_cost_change: str = Field(
        description="Rough cost impact, e.g. '~$30 cheaper per contract'."
    )


class StrategyAlternative(BaseModel):
    """A fundamentally different strategy that better fits the current conditions."""

    strategy: str = Field(description="Strategy name, e.g. 'Bull Call Spread'.")
    legs: List[OptionLeg] = Field(description="All legs of the alternative strategy.")
    rationale: str = Field(description="Why this alternative fits the thesis better.")
    tradeoff: str = Field(description="What the trader gives up vs the original strategy.")


class OptionEvaluationReport(BaseModel):
    """Structured evaluation of a user-specified option strategy."""

    verdict: Literal["Strong Buy", "Buy", "Neutral", "Avoid", "Strong Avoid"] = Field(
        description=(
            "Overall assessment: 'Strong Buy' = well-aligned and attractively priced; "
            "'Strong Avoid' = conflicts with thesis or mispriced; 'Neutral' = acceptable but not optimal."
        )
    )
    thesis_alignment: str = Field(
        description=(
            "How well the strategy's direction and time horizon align with the Portfolio "
            "Manager's verdict and the analyst reports. Note any mismatches."
        )
    )
    contract_analysis: str = Field(
        description=(
            "Analysis of the specific contract(s): IV level vs historical context, "
            "Greeks (delta/theta/vega), bid-ask spread, open interest liquidity."
        )
    )
    risk_assessment: str = Field(
        description=(
            "Key risks: daily theta decay in dollar terms, breakeven price at expiry, "
            "max loss, IV crush risk (e.g. post-earnings), and probability of profit."
        )
    )
    parameter_tweaks: List[ParameterTweak] = Field(
        description=(
            "Up to 3 tweaks to the same strategy structure that improve risk/reward. "
            "Each must comply with user_notes constraints."
        )
    )
    strategy_alternatives: List[StrategyAlternative] = Field(
        description=(
            "Up to 3 alternative strategy structures that better fit the current conditions. "
            "Suppress any that violate user_notes constraints."
        )
    )
    constraints_acknowledged: str = Field(
        description=(
            "Explicit acknowledgement of each constraint in user_notes and how it "
            "shaped the suggestions. If user_notes is empty, write 'No constraints provided.'"
        )
    )
    summary: str = Field(
        description=(
            "One-paragraph executive summary covering verdict, key reason, primary "
            "risk, and the single best modification or alternative."
        )
    )


def render_option_evaluation(report: OptionEvaluationReport) -> str:
    """Render an OptionEvaluationReport to markdown."""

    def fmt_leg(leg: OptionLeg) -> str:
        return f"{leg.action.capitalize()} {leg.option_type.upper()} ${leg.strike} exp {leg.expiration}"

    parts = [
        f"**Verdict**: {report.verdict}",
        "",
        "### Thesis Alignment",
        report.thesis_alignment,
        "",
        "### Contract Analysis",
        report.contract_analysis,
        "",
        "### Risk Assessment",
        report.risk_assessment,
    ]

    if report.parameter_tweaks:
        parts += ["", "### Parameter Tweaks"]
        for i, tweak in enumerate(report.parameter_tweaks, 1):
            legs_str = " / ".join(fmt_leg(leg) for leg in tweak.legs)
            parts += [
                "",
                f"**Tweak {i}**: {legs_str}",
                f"- Rationale: {tweak.rationale}",
                f"- Cost change: {tweak.estimated_cost_change}",
            ]

    if report.strategy_alternatives:
        parts += ["", "### Strategy Alternatives"]
        for i, alt in enumerate(report.strategy_alternatives, 1):
            legs_str = " / ".join(fmt_leg(leg) for leg in alt.legs)
            parts += [
                "",
                f"**Alternative {i} — {alt.strategy}**: {legs_str}",
                f"- Rationale: {alt.rationale}",
                f"- Tradeoff: {alt.tradeoff}",
            ]

    parts += [
        "",
        "### Constraints Acknowledged",
        report.constraints_acknowledged,
        "",
        "### Summary",
        report.summary,
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Existing Position Review — input models and output schemas
# ---------------------------------------------------------------------------


class ExistingStockPosition(BaseModel):
    """A stock position the user already holds."""

    entry_price: float = Field(description="Cost basis per share in USD.")
    shares: float = Field(description="Number of shares held.")


class ExistingOptionPosition(BaseModel):
    """An option position the user already holds."""

    ticker: str = Field(description="Underlying ticker symbol.")
    strategy: str = Field(
        description="Strategy identifier, e.g. 'long_call', 'iron_condor'."
    )
    legs: List[OptionLeg] = Field(description="All legs of the strategy in order.")
    net_premium: float = Field(
        description=(
            "Total net debit (positive) or net credit (negative) paid/received "
            "for the whole strategy per contract set, in USD."
        )
    )
    contracts: int = Field(description="Number of contract sets held.")


class StockPositionReviewReport(BaseModel):
    """Structured review of an existing stock position."""

    recommendation: Literal["Hold", "Add", "Reduce", "Close"] = Field(
        description=(
            "'Hold' = keep current size; 'Add' = increase position; "
            "'Reduce' = trim partial position; 'Close' = exit entirely."
        )
    )
    pnl_summary: str = Field(
        description="Current unrealized P&L in absolute $ and percentage terms."
    )
    thesis_status: str = Field(
        description=(
            "Whether the original bull/bear thesis behind the entry is still intact, "
            "partially intact, or broken — with specific evidence from the reports."
        )
    )
    action_plan: str = Field(
        description=(
            "Specific, actionable steps: price levels to act at, sizing guidance, "
            "and timeline. Reference the current price and entry price."
        )
    )
    exit_triggers: str = Field(
        description=(
            "Concrete conditions (price levels, events, time) that would change "
            "this recommendation."
        )
    )


class OptionPositionReviewReport(BaseModel):
    """Structured review of an existing option position."""

    recommendation: Literal["Hold", "Close Now", "Roll", "Partial Close", "Hedge"] = Field(
        description=(
            "'Hold' = keep as-is; 'Close Now' = exit immediately; "
            "'Roll' = close and reopen in a later expiry/different strike; "
            "'Partial Close' = close a subset of contracts; "
            "'Hedge' = add a protective position."
        )
    )
    pnl_summary: str = Field(
        description=(
            "Current estimated P&L vs. the net_premium paid/received, "
            "in absolute $ and percentage terms."
        )
    )
    thesis_status: str = Field(
        description=(
            "Whether the original directional thesis is still intact based on "
            "current analyst reports. Note any changes in IV or macro context."
        )
    )
    time_risk: str = Field(
        description=(
            "DTE remaining, estimated daily theta burn in $, distance to breakeven "
            "at expiry, and any upcoming event risk (earnings, FOMC, etc.)."
        )
    )
    roll_suggestion: Optional[str] = Field(
        default=None,
        description=(
            "If recommendation is 'Roll', specify the target strike, expiry, and "
            "estimated net cost/credit of the roll. None for all other recommendations."
        ),
    )
    exit_triggers: str = Field(
        description=(
            "Concrete conditions (price levels, DTE threshold, P&L % limit) that "
            "would change this recommendation."
        )
    )


def render_stock_position_review(report: StockPositionReviewReport) -> str:
    """Render a StockPositionReviewReport to markdown."""
    parts = [
        f"## Stock Position Review: {report.recommendation}",
        "",
        "### P&L Summary",
        report.pnl_summary,
        "",
        "### Thesis Status",
        report.thesis_status,
        "",
        "### Action Plan",
        report.action_plan,
        "",
        "### Exit Triggers",
        report.exit_triggers,
    ]
    return "\n".join(parts)


def render_option_position_review(report: OptionPositionReviewReport) -> str:
    """Render an OptionPositionReviewReport to markdown."""
    parts = [
        f"## Option Position Review: {report.recommendation}",
        "",
        "### P&L Summary",
        report.pnl_summary,
        "",
        "### Thesis Status",
        report.thesis_status,
        "",
        "### Time Risk",
        report.time_risk,
    ]
    if report.roll_suggestion:
        parts += ["", "### Roll Suggestion", report.roll_suggestion]
    parts += ["", "### Exit Triggers", report.exit_triggers]
    return "\n".join(parts)
