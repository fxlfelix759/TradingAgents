# Existing Position Review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "review existing position" flow to `tradingagents analyze` so users can get a dedicated hold/add/reduce/close recommendation for a stock or option position they already own.

**Architecture:** Two new post-pipeline agent nodes (`StockPositionReviewer`, `OptionPositionReviewer`) run after `Portfolio Manager` via a new conditional routing function. The CLI Step 3 becomes a 4-way branch. Cached main reports are reused unchanged; position review output is stored in separate state fields and saved to `6_position_review/`.

**Tech Stack:** LangGraph, Pydantic, yfinance, questionary (Rich CLI), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `tradingagents/agents/schemas.py` | 4 new Pydantic models + 2 render helpers |
| Modify | `tradingagents/agents/utils/agent_states.py` | 4 new AgentState fields |
| Modify | `tradingagents/graph/propagation.py` | Pass new fields in `create_initial_state()` |
| Create | `tradingagents/agents/analysts/stock_position_reviewer.py` | StockPositionReviewer node factory |
| Create | `tradingagents/agents/analysts/option_position_reviewer.py` | OptionPositionReviewer node factory |
| Modify | `tradingagents/agents/__init__.py` | Export the two new factories |
| Modify | `tradingagents/graph/conditional_logic.py` | `route_post_pipeline()` routing function |
| Modify | `tradingagents/graph/setup.py` | Wire new nodes + conditional edge from Portfolio Manager |
| Modify | `tradingagents/graph/trading_graph.py` | Extend `propagate()`, add 2 cache-bypass methods, update `_log_state` and `_run_graph` |
| Modify | `cli/utils.py` | `ask_existing_position()` |
| Modify | `cli/main.py` | Step 3 branch, cache bypass, streaming init state, display, save |
| Create | `tests/test_position_review_schemas.py` | Schema + render unit tests |
| Create | `tests/test_position_reviewers.py` | Node no-op + LLM-call unit tests |
| Modify | `tests/test_position_review_routing.py` | Routing + graph compile smoke tests |

---

### Task 1: Schemas and render helpers

**Files:**
- Modify: `tradingagents/agents/schemas.py` (append after `OptionEvaluationReport`)
- Create: `tests/test_position_review_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_position_review_schemas.py
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_position_review_schemas.py -v
```
Expected: ImportError — `ExistingStockPosition` not yet defined

- [ ] **Step 3: Add the 4 models and 2 render helpers to `tradingagents/agents/schemas.py`**

Append after the closing of `OptionEvaluationReport` (after line 458 in the current file):

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_position_review_schemas.py -v
```
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/schemas.py tests/test_position_review_schemas.py
git commit -m "feat: add ExistingPosition schemas and PositionReviewReport models with render helpers"
```

---

### Task 2: AgentState — add 4 new fields

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_position_review_schemas.py`:

```python
def test_agent_state_has_position_fields():
    """AgentState TypedDict accepts new position fields without error."""
    from tradingagents.agents.utils.agent_states import AgentState
    # TypedDicts can be instantiated as dicts — just check the annotations exist
    hints = AgentState.__annotations__
    assert "existing_stock_position" in hints
    assert "existing_option_position" in hints
    assert "stock_position_review" in hints
    assert "option_position_review" in hints
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_position_review_schemas.py::test_agent_state_has_position_fields -v
```
Expected: FAIL — `AssertionError` (fields not in annotations)

- [ ] **Step 3: Add the 4 fields to `tradingagents/agents/utils/agent_states.py`**

Add these imports at the top of the file alongside `TargetOption`:

```python
from tradingagents.agents.schemas import (
    TargetOption,
    ExistingStockPosition,
    ExistingOptionPosition,
)
```

Add to the `AgentState` class after the `option_evaluation_report` field (current last field):

```python
    existing_stock_position: Annotated[
        Optional[ExistingStockPosition],
        "Existing stock position to review (None = skip reviewer)",
    ]
    existing_option_position: Annotated[
        Optional[ExistingOptionPosition],
        "Existing option position to review (None = skip reviewer)",
    ]
    stock_position_review: Annotated[
        Optional[str],
        "Markdown review report produced by StockPositionReviewer",
    ]
    option_position_review: Annotated[
        Optional[str],
        "Markdown review report produced by OptionPositionReviewer",
    ]
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_position_review_schemas.py::test_agent_state_has_position_fields -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tests/test_position_review_schemas.py
git commit -m "feat: add existing_position and position_review fields to AgentState"
```

---

### Task 3: Propagation — pass position fields in initial state

**Files:**
- Modify: `tradingagents/graph/propagation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_position_review_schemas.py`:

```python
def test_propagator_initial_state_has_position_fields():
    from tradingagents.graph.propagation import Propagator
    from tradingagents.agents.schemas import ExistingStockPosition

    p = Propagator()
    pos = ExistingStockPosition(entry_price=500.0, shares=10.0)
    state = p.create_initial_state(
        "MSFT", "2026-05-03",
        existing_stock_position=pos,
    )
    assert state["existing_stock_position"] is pos
    assert state["existing_option_position"] is None
    assert state["stock_position_review"] is None
    assert state["option_position_review"] is None
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_position_review_schemas.py::test_propagator_initial_state_has_position_fields -v
```
Expected: FAIL — `TypeError` (unexpected keyword argument)

- [ ] **Step 3: Update `tradingagents/graph/propagation.py`**

Change the `create_initial_state` signature and body:

```python
    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        past_context: str = "",
        target_option=None,
        existing_stock_position=None,
        existing_option_position=None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "change_report": "",
            "target_option": target_option,
            "option_evaluation_report": None,
            "existing_stock_position": existing_stock_position,
            "existing_option_position": existing_option_position,
            "stock_position_review": None,
            "option_position_review": None,
        }
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_position_review_schemas.py::test_propagator_initial_state_has_position_fields -v
```
Expected: PASS

- [ ] **Step 5: Run the full schema test file**

```bash
pytest tests/test_position_review_schemas.py -v
```
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tradingagents/graph/propagation.py tests/test_position_review_schemas.py
git commit -m "feat: propagate existing_position fields through initial graph state"
```

---

### Task 4: StockPositionReviewer node

**Files:**
- Create: `tradingagents/agents/analysts/stock_position_reviewer.py`
- Create: `tests/test_position_reviewers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_position_reviewers.py
import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.schemas import (
    ExistingStockPosition,
    ExistingOptionPosition,
    OptionLeg,
    StockPositionReviewReport,
    OptionPositionReviewReport,
)


def _make_base_state(extra=None):
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-03",
        "final_trade_decision": "**Rating**: Buy\n**Summary**: Bullish on MSFT.",
        "investment_plan": "Buy with conviction.",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "market_report": "Bullish technicals.",
        "fundamentals_report": "Strong earnings growth.",
        "sentiment_report": "Positive social sentiment.",
        "news_report": "No adverse news.",
        "options_report": "Moderate IV at 25%.",
        "messages": [],
        "existing_stock_position": None,
        "existing_option_position": None,
    }
    if extra:
        state.update(extra)
    return state


# --- StockPositionReviewer ---

def test_stock_reviewer_noop_when_no_position(mock_llm_client):
    from tradingagents.agents.analysts.stock_position_reviewer import create_stock_position_reviewer
    node = create_stock_position_reviewer(mock_llm_client.get_llm())
    result = node(_make_base_state())
    assert result == {}


def test_stock_reviewer_calls_llm_when_position_set(mock_llm_client):
    from tradingagents.agents.analysts.stock_position_reviewer import create_stock_position_reviewer
    from tradingagents.agents.schemas import render_stock_position_review

    mock_report = StockPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$800 (-16%)",
        thesis_status="Thesis intact.",
        action_plan="Hold with stop at $390.",
        exit_triggers="Close below $390.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingStockPosition(entry_price=500.0, shares=10.0)
    state = _make_base_state({"existing_stock_position": pos})

    with patch(
        "tradingagents.agents.analysts.stock_position_reviewer.yf.Ticker"
    ) as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 420.0
        mock_ticker_cls.return_value = mock_ticker

        node = create_stock_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    assert "stock_position_review" in result
    assert "## Stock Position Review: Hold" in result["stock_position_review"]
    assert "-$800" in result["stock_position_review"]
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_position_reviewers.py::test_stock_reviewer_noop_when_no_position tests/test_position_reviewers.py::test_stock_reviewer_calls_llm_when_position_set -v
```
Expected: ImportError — `stock_position_reviewer` module does not exist

- [ ] **Step 3: Create `tradingagents/agents/analysts/stock_position_reviewer.py`**

```python
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

        price_str = f"${current_price:.2f}" if current_price else "unavailable"
        if current_price:
            pnl_abs = (current_price - position.entry_price) * position.shares
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            pnl_context = (
                f"Unrealized P&L: ${pnl_abs:+.2f} ({pnl_pct:+.1f}%)"
            )
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
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
pytest tests/test_position_reviewers.py::test_stock_reviewer_noop_when_no_position tests/test_position_reviewers.py::test_stock_reviewer_calls_llm_when_position_set -v
```
Expected: Both PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/stock_position_reviewer.py tests/test_position_reviewers.py
git commit -m "feat: add StockPositionReviewer agent node"
```

---

### Task 5: OptionPositionReviewer node

**Files:**
- Create: `tradingagents/agents/analysts/option_position_reviewer.py`
- Modify: `tests/test_position_reviewers.py`

- [ ] **Step 1: Add the failing tests** (append to `tests/test_position_reviewers.py`)

```python
# --- OptionPositionReviewer ---

def test_option_reviewer_noop_when_no_position(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer
    node = create_option_position_reviewer(mock_llm_client.get_llm())
    result = node(_make_base_state())
    assert result == {}


def test_option_reviewer_calls_llm_and_chain_when_position_set(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer

    mock_report = OptionPositionReviewReport(
        recommendation="Hold",
        pnl_summary="-$300 (-35%)",
        thesis_status="Bullish thesis valid.",
        time_risk="42 DTE, theta $12/day, breakeven $428.50.",
        roll_suggestion=None,
        exit_triggers="Close if stock drops below $400.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingOptionPosition(
        ticker="MSFT",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-06-20")],
        net_premium=8.50,
        contracts=2,
    )
    state = _make_base_state({"existing_option_position": pos})

    with patch(
        "tradingagents.agents.analysts.option_position_reviewer.get_full_options_chain_for_target",
        return_value="# Mocked chain data",
    ):
        node = create_option_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    assert "option_position_review" in result
    assert "## Option Position Review: Hold" in result["option_position_review"]


@pytest.mark.smoke
def test_option_reviewer_with_roll_recommendation(mock_llm_client):
    from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer

    mock_report = OptionPositionReviewReport(
        recommendation="Roll",
        pnl_summary="-$200 (-25%)",
        thesis_status="Bullish but needs more time.",
        time_risk="14 DTE, theta $22/day.",
        roll_suggestion="Roll to $420 call expiring 2026-07-18 for $3.00 net debit.",
        exit_triggers="Close if stock breaks $410.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    pos = ExistingOptionPosition(
        ticker="MSFT",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=420.0, expiration="2026-05-17")],
        net_premium=8.50,
        contracts=1,
    )
    state = _make_base_state({"existing_option_position": pos})

    with patch(
        "tradingagents.agents.analysts.option_position_reviewer.get_full_options_chain_for_target",
        return_value="# Mocked chain data",
    ):
        node = create_option_position_reviewer(mock_llm_client.get_llm())
        result = node(state)

    assert "Roll Suggestion" in result["option_position_review"]
    assert "2026-07-18" in result["option_position_review"]
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_position_reviewers.py -k "option_reviewer" -v
```
Expected: ImportError — `option_position_reviewer` module does not exist

- [ ] **Step 3: Create `tradingagents/agents/analysts/option_position_reviewer.py`**

```python
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
        if not position:
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

1. **Recommendation**: Hold / Close Now / Roll / Partial Close / Hedge
2. **P&L Summary**: Estimate current value from chain data vs. cost basis in $ and %
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
```

- [ ] **Step 4: Run to confirm all reviewer tests pass**

```bash
pytest tests/test_position_reviewers.py -v
```
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/option_position_reviewer.py tests/test_position_reviewers.py
git commit -m "feat: add OptionPositionReviewer agent node"
```

---

### Task 6: Export new factories from `tradingagents/agents/__init__.py`

**Files:**
- Modify: `tradingagents/agents/__init__.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_position_reviewers.py`)

```python
def test_position_reviewer_factories_importable_from_agents():
    from tradingagents.agents import (
        create_stock_position_reviewer,
        create_option_position_reviewer,
    )
    assert callable(create_stock_position_reviewer)
    assert callable(create_option_position_reviewer)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_position_reviewers.py::test_position_reviewer_factories_importable_from_agents -v
```
Expected: ImportError

- [ ] **Step 3: Add exports to `tradingagents/agents/__init__.py`**

Add after the `create_option_trade_evaluator` import line:
```python
from .analysts.stock_position_reviewer import create_stock_position_reviewer
from .analysts.option_position_reviewer import create_option_position_reviewer
```

Add to `__all__`:
```python
    "create_stock_position_reviewer",
    "create_option_position_reviewer",
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_position_reviewers.py::test_position_reviewer_factories_importable_from_agents -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/__init__.py tests/test_position_reviewers.py
git commit -m "feat: export StockPositionReviewer and OptionPositionReviewer factories"
```

---

### Task 7: ConditionalLogic — add `route_post_pipeline`

**Files:**
- Modify: `tradingagents/graph/conditional_logic.py`
- Create: `tests/test_position_review_routing.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_position_review_routing.py -v
```
Expected: AttributeError — `ConditionalLogic` has no `route_post_pipeline`

- [ ] **Step 3: Add `route_post_pipeline` to `tradingagents/graph/conditional_logic.py`**

Add import at top of the file:
```python
from langgraph.graph import END
```

Add method to the `ConditionalLogic` class after `should_continue_risk_analysis`:

```python
    def route_post_pipeline(self, state: AgentState) -> str:
        """Route from Portfolio Manager to the appropriate post-pipeline node."""
        if state.get("target_option") is not None:
            return "Option Trade Evaluator"
        if state.get("existing_stock_position") is not None:
            return "Stock Position Reviewer"
        if state.get("existing_option_position") is not None:
            return "Option Position Reviewer"
        return END
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
pytest tests/test_position_review_routing.py -v
```
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/conditional_logic.py tests/test_position_review_routing.py
git commit -m "feat: add route_post_pipeline conditional routing function"
```

---

### Task 8: GraphSetup — wire new nodes and conditional edge

**Files:**
- Modify: `tradingagents/graph/setup.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_position_review_routing.py`)

```python
def test_graph_compiles_with_position_reviewer_nodes():
    """Graph setup compiles without error when position reviewer nodes are wired."""
    from unittest.mock import MagicMock
    from tradingagents.graph.setup import GraphSetup
    from tradingagents.graph.conditional_logic import ConditionalLogic

    quick_llm = MagicMock()
    quick_llm.with_structured_output.return_value = MagicMock()
    deep_llm = MagicMock()
    deep_llm.with_structured_output.return_value = MagicMock()

    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def dummy_tool(query: str) -> str:
        """dummy"""
        return query

    tool_nodes = {
        k: ToolNode([dummy_tool])
        for k in ["market", "social", "news", "fundamentals"]
    }
    cl = ConditionalLogic()
    gs = GraphSetup(quick_llm, deep_llm, tool_nodes, cl)
    workflow = gs.setup_graph(["market"])
    graph = workflow.compile()
    assert graph is not None
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_position_review_routing.py::test_graph_compiles_with_position_reviewer_nodes -v
```
Expected: PASS currently (graph still compiles with old wiring) — this test will fail AFTER we change setup.py if we break something. Run it now to establish a green baseline.

- [ ] **Step 3: Update `tradingagents/graph/setup.py`**

Replace the current OptionTradeEvaluator wiring at the bottom of `setup_graph()`:

```python
        # Option Trade Evaluator (no-op when target_option is absent)
        from tradingagents.agents import create_option_trade_evaluator
        option_evaluator_node = create_option_trade_evaluator(self.deep_thinking_llm)
        workflow.add_node("Option Trade Evaluator", option_evaluator_node)
        workflow.add_edge("Portfolio Manager", "Option Trade Evaluator")
        workflow.add_edge("Option Trade Evaluator", END)
```

With:

```python
        # Post-pipeline nodes: route from Portfolio Manager based on what the user requested
        from tradingagents.agents import (
            create_option_trade_evaluator,
            create_stock_position_reviewer,
            create_option_position_reviewer,
        )
        workflow.add_node("Option Trade Evaluator", create_option_trade_evaluator(self.deep_thinking_llm))
        workflow.add_node("Stock Position Reviewer", create_stock_position_reviewer(self.deep_thinking_llm))
        workflow.add_node("Option Position Reviewer", create_option_position_reviewer(self.deep_thinking_llm))

        workflow.add_conditional_edges(
            "Portfolio Manager",
            self.conditional_logic.route_post_pipeline,
            {
                "Option Trade Evaluator": "Option Trade Evaluator",
                "Stock Position Reviewer": "Stock Position Reviewer",
                "Option Position Reviewer": "Option Position Reviewer",
                END: END,
            },
        )
        workflow.add_edge("Option Trade Evaluator", END)
        workflow.add_edge("Stock Position Reviewer", END)
        workflow.add_edge("Option Position Reviewer", END)
```

- [ ] **Step 4: Run to confirm graph still compiles**

```bash
pytest tests/test_position_review_routing.py -v
```
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/setup.py tests/test_position_review_routing.py
git commit -m "feat: wire StockPositionReviewer and OptionPositionReviewer into graph with conditional routing"
```

---

### Task 9: TradingAgentsGraph — extend propagate, add cache bypass methods, update log

**Files:**
- Modify: `tradingagents/graph/trading_graph.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_position_review_routing.py`)

```python
def test_propagate_signature_accepts_position_args():
    """propagate() accepts the new position keyword arguments without TypeError."""
    import inspect
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    sig = inspect.signature(TradingAgentsGraph.propagate)
    params = sig.parameters
    assert "existing_stock_position" in params
    assert "existing_option_position" in params
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_position_review_routing.py::test_propagate_signature_accepts_position_args -v
```
Expected: FAIL — `AssertionError` (parameters not present)

- [ ] **Step 3: Update `tradingagents/graph/trading_graph.py`**

**3a.** Change `propagate()` signature and its cache-bypass block. Find the current `propagate` method and replace it:

```python
    def propagate(
        self,
        company_name,
        trade_date,
        target_option=None,
        existing_stock_position=None,
        existing_option_position=None,
    ):
        """Run the trading agents graph for a company on a specific date."""
        self.ticker = company_name

        cached_path = (
            Path(self.config["results_dir"])
            / company_name
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{trade_date}.json"
        )

        if target_option is not None and cached_path.exists():
            logger.info(
                "Cache hit for %s on %s — running option evaluator only.",
                company_name, trade_date,
            )
            report = self._evaluate_option_only(company_name, trade_date, target_option, cached_path)
            return None, report

        if existing_stock_position is not None and cached_path.exists():
            logger.info(
                "Cache hit for %s on %s — running stock position reviewer only.",
                company_name, trade_date,
            )
            report = self._review_stock_position_only(company_name, trade_date, existing_stock_position, cached_path)
            return None, report

        if existing_option_position is not None and cached_path.exists():
            logger.info(
                "Cache hit for %s on %s — running option position reviewer only.",
                company_name, trade_date,
            )
            report = self._review_option_position_only(company_name, trade_date, existing_option_position, cached_path)
            return None, report

        self._target_option = target_option
        self._existing_stock_position = existing_stock_position
        self._existing_option_position = existing_option_position

        self._resolve_pending_entries(company_name)

        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()
```

**3b.** Add the two new cache-bypass methods after `_evaluate_option_only`:

```python
    def _review_stock_position_only(self, company_name, trade_date, position, cached_path):
        """Run only the StockPositionReviewer against a cached analysis JSON."""
        with open(cached_path, encoding="utf-8") as f:
            cached = json.load(f)

        state = {
            "company_of_interest": cached.get("company_of_interest", company_name),
            "trade_date": cached.get("trade_date", str(trade_date)),
            "existing_stock_position": position,
            "final_trade_decision": cached.get("final_trade_decision", ""),
            "investment_plan": cached.get("investment_plan", ""),
            "trader_investment_plan": cached.get("trader_investment_decision", ""),
            "market_report": cached.get("market_report", ""),
            "fundamentals_report": cached.get("fundamentals_report", ""),
            "sentiment_report": cached.get("sentiment_report", ""),
            "news_report": cached.get("news_report", ""),
            "options_report": cached.get("options_report", ""),
            "messages": [],
        }

        from tradingagents.agents.analysts.stock_position_reviewer import create_stock_position_reviewer
        reviewer = create_stock_position_reviewer(self.deep_thinking_llm)
        result = reviewer(state)
        report = result.get("stock_position_review", "")

        cached["stock_position_review"] = report
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=4, ensure_ascii=False)

        return report

    def _review_option_position_only(self, company_name, trade_date, position, cached_path):
        """Run only the OptionPositionReviewer against a cached analysis JSON."""
        with open(cached_path, encoding="utf-8") as f:
            cached = json.load(f)

        state = {
            "company_of_interest": cached.get("company_of_interest", company_name),
            "trade_date": cached.get("trade_date", str(trade_date)),
            "existing_option_position": position,
            "final_trade_decision": cached.get("final_trade_decision", ""),
            "investment_plan": cached.get("investment_plan", ""),
            "trader_investment_plan": cached.get("trader_investment_decision", ""),
            "market_report": cached.get("market_report", ""),
            "fundamentals_report": cached.get("fundamentals_report", ""),
            "sentiment_report": cached.get("sentiment_report", ""),
            "news_report": cached.get("news_report", ""),
            "options_report": cached.get("options_report", ""),
            "messages": [],
        }

        from tradingagents.agents.analysts.option_position_reviewer import create_option_position_reviewer
        reviewer = create_option_position_reviewer(self.deep_thinking_llm)
        result = reviewer(state)
        report = result.get("option_position_review", "")

        cached["option_position_review"] = report
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=4, ensure_ascii=False)

        return report
```

**3c.** Update `_run_graph` to pass position fields to `create_initial_state`. Find the line:

```python
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context,
            target_option=getattr(self, "_target_option", None),
        )
```

Replace with:

```python
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context,
            target_option=getattr(self, "_target_option", None),
            existing_stock_position=getattr(self, "_existing_stock_position", None),
            existing_option_position=getattr(self, "_existing_option_position", None),
        )
```

**3d.** Update `_log_state` to persist the two new report fields. Find the line:

```python
            "option_evaluation_report": final_state.get("option_evaluation_report", ""),
```

Add after it:

```python
            "stock_position_review": final_state.get("stock_position_review", ""),
            "option_position_review": final_state.get("option_position_review", ""),
```

- [ ] **Step 4: Run to confirm the signature test passes**

```bash
pytest tests/test_position_review_routing.py -v
```
Expected: All 6 tests PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest -m unit -v
```
Expected: All existing unit tests still pass

- [ ] **Step 6: Commit**

```bash
git add tradingagents/graph/trading_graph.py tests/test_position_review_routing.py
git commit -m "feat: extend TradingAgentsGraph.propagate with position review cache bypass paths"
```

---

### Task 10: CLI utils — `ask_existing_position`

**Files:**
- Modify: `cli/utils.py`

- [ ] **Step 1: Add the imports needed at the top of `cli/utils.py`**

The file already imports `OptionLeg`, `TargetOption`. Add:

```python
from tradingagents.agents.schemas import (
    OptionLeg,
    TargetOption,
    ExistingStockPosition,
    ExistingOptionPosition,
)
```

(Replace the existing `OptionLeg`/`TargetOption` imports with this combined import.)

- [ ] **Step 2: Add `ask_existing_position` to `cli/utils.py`** (append after `ask_option_strategy`)

```python
_MODE_NEW_OPTION = "Evaluate a new option strategy"
_MODE_STOCK = "Review an existing stock position"
_MODE_OPTION = "Review an existing option position"
_MODE_SKIP = "Skip"


def ask_existing_position(ticker: str) -> dict:
    """Step 3 prompt: 4-way branch for analysis mode.

    Returns a dict with exactly one non-None value among:
        target_option, existing_stock_position, existing_option_position
    All three are None when the user skips.
    """
    result = {
        "target_option": None,
        "existing_stock_position": None,
        "existing_option_position": None,
    }

    mode = questionary.select(
        "What would you like to do?",
        choices=[_MODE_NEW_OPTION, _MODE_STOCK, _MODE_OPTION, _MODE_SKIP],
        style=_OPTION_STYLE,
    ).ask()

    if mode is None or mode == _MODE_SKIP:
        return result

    if mode == _MODE_NEW_OPTION:
        result["target_option"] = ask_option_strategy(ticker)
        return result

    if mode == _MODE_STOCK:
        entry_price_str = questionary.text(
            "Entry price per share ($):",
            validate=lambda x: x.replace(".", "", 1).isdigit() or "Enter a numeric price.",
        ).ask()
        if entry_price_str is None:
            return result

        shares_str = questionary.text(
            "Number of shares held:",
            validate=lambda x: x.replace(".", "", 1).isdigit() or "Enter a numeric quantity.",
        ).ask()
        if shares_str is None:
            return result

        result["existing_stock_position"] = ExistingStockPosition(
            entry_price=float(entry_price_str),
            shares=float(shares_str),
        )
        return result

    if mode == _MODE_OPTION:
        console.print("\n[cyan]Enter the details of your existing option position.[/cyan]")
        target = ask_option_strategy(ticker)
        if target is None:
            return result

        net_premium_str = questionary.text(
            "Net premium paid/received per contract ($, positive = debit, negative = credit):",
            validate=lambda x: (
                x.lstrip("-").replace(".", "", 1).isdigit()
            ) or "Enter a numeric premium (e.g. 8.50 or -1.20).",
        ).ask()
        if net_premium_str is None:
            return result

        contracts_str = questionary.text(
            "Number of contracts held:",
            validate=lambda x: x.isdigit() or "Enter a whole number.",
        ).ask()
        if contracts_str is None:
            return result

        result["existing_option_position"] = ExistingOptionPosition(
            ticker=target.ticker,
            strategy=target.strategy,
            legs=target.legs,
            net_premium=float(net_premium_str),
            contracts=int(contracts_str),
        )
        return result

    return result
```

- [ ] **Step 3: Verify the existing imports at the top of `cli/utils.py` and update the schema import**

Find the current import of `OptionLeg` and `TargetOption` in `cli/utils.py`. It likely reads:
```python
from tradingagents.agents.schemas import OptionLeg, TargetOption
```
Replace it with:
```python
from tradingagents.agents.schemas import (
    OptionLeg,
    TargetOption,
    ExistingStockPosition,
    ExistingOptionPosition,
)
```

- [ ] **Step 4: Smoke-test the import**

```bash
python -c "from cli.utils import ask_existing_position; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add cli/utils.py
git commit -m "feat: add ask_existing_position 4-way CLI branch for Step 3"
```

---

### Task 11: CLI main — wire Step 3, cache bypass, streaming, display, save

**Files:**
- Modify: `cli/main.py`

- [ ] **Step 1: Update `get_user_selections()` — replace `ask_option_strategy` with `ask_existing_position`**

Find the Step 3 block (around line 545–552):

```python
    # Step 3: Option Strategy (optional)
    console.print(
        create_question_box(
            "Step 3: Option Strategy (Optional)",
            "Optionally evaluate a specific option strategy against this analysis",
        )
    )
    target_option = ask_option_strategy(selected_ticker)
```

Replace with:

```python
    # Step 3: Analysis mode
    console.print(
        create_question_box(
            "Step 3: Analysis Mode",
            "Evaluate a new option strategy, review an existing position, or skip",
        )
    )
    from cli.utils import ask_existing_position
    step3 = ask_existing_position(selected_ticker)
    target_option = step3["target_option"]
    existing_stock_position = step3["existing_stock_position"]
    existing_option_position = step3["existing_option_position"]
```

- [ ] **Step 2: Update the `return` dict of `get_user_selections()` to include new fields**

Find the return statement (around line 629) and add the two new keys:

```python
    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "model": selected_model,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
        "target_option": target_option,
        "existing_stock_position": existing_stock_position,
        "existing_option_position": existing_option_position,
        "cache_exists": _cache_exists,
    }
```

- [ ] **Step 3: Update the cache notice message to cover position review**

Find (around line 539):
```python
    if _cache_exists:
        console.print(
            f"\n[green]Found existing report for {selected_ticker} on {analysis_date}. "
            "Cached analysis will be used for option evaluation.[/green]\n"
        )
```

Replace with:
```python
    if _cache_exists:
        console.print(
            f"\n[green]Found existing report for {selected_ticker} on {analysis_date}. "
            "Cached analysis will be used if you request a position review or option evaluation.[/green]\n"
        )
```

- [ ] **Step 4: Update the cache-bypass block in `run_analysis()`**

Find (around line 1062):
```python
        target_option = selections.get("target_option")
        cache_exists = selections.get("cache_exists", False)

        if target_option and cache_exists:
            # Cache hit: run only the evaluator directly
            _, option_report = graph.propagate(
                selections["ticker"], selections["analysis_date"], target_option=target_option
            )
            if option_report:
                console.print(Rule("Option Trade Evaluation", style="bold cyan"))
                console.print(Markdown(option_report))
            return
```

Replace with:
```python
        target_option = selections.get("target_option")
        existing_stock_position = selections.get("existing_stock_position")
        existing_option_position = selections.get("existing_option_position")
        cache_exists = selections.get("cache_exists", False)

        if cache_exists:
            if target_option:
                _, report = graph.propagate(
                    selections["ticker"], selections["analysis_date"],
                    target_option=target_option,
                )
                if report:
                    console.print(Rule("Option Trade Evaluation", style="bold cyan"))
                    console.print(Markdown(report))
                return
            if existing_stock_position:
                _, report = graph.propagate(
                    selections["ticker"], selections["analysis_date"],
                    existing_stock_position=existing_stock_position,
                )
                if report:
                    console.print(Rule("Stock Position Review", style="bold green"))
                    console.print(Markdown(report))
                return
            if existing_option_position:
                _, report = graph.propagate(
                    selections["ticker"], selections["analysis_date"],
                    existing_option_position=existing_option_position,
                )
                if report:
                    console.print(Rule("Option Position Review", style="bold cyan"))
                    console.print(Markdown(report))
                return
```

- [ ] **Step 5: Update the direct streaming path — pass position fields to `create_initial_state`**

Find (around line 1077):
```python
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"],
            target_option=target_option,
        )
```

Replace with:
```python
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"],
            target_option=target_option,
            existing_stock_position=existing_stock_position,
            existing_option_position=existing_option_position,
        )
```

- [ ] **Step 6: Update `display_complete_report()` to show position reviews**

Find (around line 801):
```python
    if final_state.get("option_evaluation_report"):
        console.print(Rule("Option Trade Evaluation", style="bold cyan"))
        console.print(Markdown(final_state["option_evaluation_report"]))
```

Add after it:
```python
    if final_state.get("stock_position_review"):
        console.print(Rule("Stock Position Review", style="bold green"))
        console.print(Markdown(final_state["stock_position_review"]))

    if final_state.get("option_position_review"):
        console.print(Rule("Option Position Review", style="bold cyan"))
        console.print(Markdown(final_state["option_position_review"]))
```

- [ ] **Step 7: Update `save_report_to_disk()` to save position review reports**

Find (around line 735):
```python
    # Write consolidated report
    header = ...
```

Add before the `# Write consolidated report` comment:
```python
    # 6. Position Review
    if final_state.get("stock_position_review"):
        position_dir = save_path / "6_position_review"
        position_dir.mkdir(exist_ok=True)
        (position_dir / "stock_position_review.md").write_text(
            final_state["stock_position_review"], encoding="utf-8"
        )
        sections.append(f"## VI. Stock Position Review\n\n{final_state['stock_position_review']}")

    if final_state.get("option_position_review"):
        position_dir = save_path / "6_position_review"
        position_dir.mkdir(exist_ok=True)
        (position_dir / "option_position_review.md").write_text(
            final_state["option_position_review"], encoding="utf-8"
        )
        sections.append(f"## VI. Option Position Review\n\n{final_state['option_position_review']}")
```

- [ ] **Step 8: Smoke-test the CLI import**

```bash
python -c "from cli.main import run_analysis; print('OK')"
```
Expected: `OK`

- [ ] **Step 9: Run full test suite**

```bash
pytest -v
```
Expected: All tests pass, no regressions

- [ ] **Step 10: Commit**

```bash
git add cli/main.py
git commit -m "feat: wire existing position review into CLI Step 3, cache bypass, display, and save"
```

---

## Self-Review

**Spec coverage check:**
- [x] ExistingStockPosition schema (Task 1)
- [x] ExistingOptionPosition schema (Task 1)
- [x] StockPositionReviewReport schema + render (Task 1)
- [x] OptionPositionReviewReport schema + render (Task 1)
- [x] AgentState new fields (Task 2)
- [x] Propagation updated (Task 3)
- [x] StockPositionReviewer node (Task 4)
- [x] OptionPositionReviewer node (Task 5)
- [x] __init__.py exports (Task 6)
- [x] route_post_pipeline (Task 7)
- [x] Graph rewiring (Task 8)
- [x] propagate() extended (Task 9)
- [x] _review_stock_position_only cache bypass (Task 9)
- [x] _review_option_position_only cache bypass (Task 9)
- [x] _log_state updated (Task 9)
- [x] _run_graph updated (Task 9)
- [x] ask_existing_position (Task 10)
- [x] CLI Step 3 branch (Task 11)
- [x] CLI cache bypass (Task 11)
- [x] CLI streaming init state (Task 11)
- [x] display_complete_report (Task 11)
- [x] save_report_to_disk (Task 11)

**Notes:**
- AgentState stores position reviews as `Optional[str]` (rendered markdown), consistent with `option_evaluation_report`. The spec stated `Optional[StockPositionReviewReport]` but the correct pattern for this codebase is rendered strings in state.
- The `ask_existing_position` import inside `get_user_selections()` avoids a circular import since `cli/utils.py` already imports from the agent layer.
