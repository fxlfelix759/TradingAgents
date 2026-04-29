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
from typing import Optional

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
