# Existing Position Review — Design Spec

**Date:** 2026-05-03  
**Branch:** feat/option-trade-evaluator  
**Status:** Approved

---

## Overview

Add a "review existing position" flow to the `tradingagents analyze` command. Users can input a position they already hold (stock or option) and receive a dedicated review report alongside the main analysis. The main analysis report stays clean and reusable; position data never pollutes it.

If a cached analysis exists for the requested ticker+date, only the reviewer node runs (no full pipeline). If no cache exists, the full pipeline runs first, then the reviewer node runs at the end.

---

## UX Flow

Step 3 of the CLI prompt becomes a 4-way branch (mutually exclusive):

```
Step 3: Analysis Mode
  A) Evaluate a new option strategy    → existing ask_option_strategy() → target_option
  B) Review existing stock position    → ask entry_price + shares       → existing_stock_position
  C) Review existing option position   → ask legs + net_premium + contracts → existing_option_position
  D) Skip                              → all None, full pipeline only
```

### Stock position data collected (option B)
- Entry price per share (float)
- Number of shares (float)

### Option position data collected (option C)
- Strategy category + specific strategy (same leg-selection flow as current ask_option_strategy)
- Per-leg: strike + expiration date
- Net premium paid/received for the whole strategy (float, single number)
- Number of contracts (int)

---

## Schemas

Add to `tradingagents/agents/schemas.py`:

```python
class ExistingStockPosition(BaseModel):
    entry_price: float        # cost basis per share
    shares: float             # number of shares held

class ExistingOptionPosition(BaseModel):
    ticker: str
    strategy: str             # e.g. "long_call", "iron_condor"
    legs: List[OptionLeg]     # reuses existing OptionLeg
    net_premium: float        # total net debit (positive) or credit (negative)
    contracts: int            # number of contract sets

class StockPositionReviewReport(BaseModel):
    recommendation: Literal["Hold", "Add", "Reduce", "Close"]
    pnl_summary: str          # current unrealized P&L in $ and %
    thesis_status: str        # is the original bull/bear thesis still intact?
    action_plan: str          # specific steps to take now
    exit_triggers: str        # conditions that would change this recommendation

class OptionPositionReviewReport(BaseModel):
    recommendation: Literal["Hold", "Close Now", "Roll", "Partial Close", "Hedge"]
    pnl_summary: str          # current unrealized P&L vs. net_premium paid
    thesis_status: str        # is the underlying thesis still intact?
    time_risk: str            # DTE remaining, theta burn rate, breakeven distance
    roll_suggestion: Optional[str]  # None if not rolling; else target strike/expiry
    exit_triggers: str        # conditions that would change this recommendation
```

---

## AgentState additions

In `tradingagents/agents/utils/agent_states.py`:

```python
# inputs
existing_stock_position:  Optional[ExistingStockPosition]   = None
existing_option_position: Optional[ExistingOptionPosition]  = None

# outputs
stock_position_review:    Optional[StockPositionReviewReport]  = None
option_position_review:   Optional[OptionPositionReviewReport] = None
```

---

## New Agent Nodes

### `tradingagents/agents/analysts/stock_position_reviewer.py`

- Factory: `create_stock_position_reviewer(llm)`
- Short-circuits (no-op) if `state["existing_stock_position"] is None`
- Fetches current stock price via existing yfinance tools
- Builds prompt with: existing_stock_position, current price, final_trade_decision, investment_plan, all analyst reports
- Computes P&L context in the prompt (entry vs. current, % change)
- Structured output bound to `StockPositionReviewReport`
- Writes result to `state["stock_position_review"]`

### `tradingagents/agents/analysts/option_position_reviewer.py`

- Factory: `create_option_position_reviewer(llm)`
- Short-circuits if `state["existing_option_position"] is None`
- Fetches live options chain with Greeks for all unique expiries (same approach as OptionTradeEvaluator)
- Builds prompt with: existing_option_position, current chain data, final_trade_decision, all analyst reports
- Structured output bound to `OptionPositionReviewReport`
- Writes result to `state["option_position_review"]`

---

## Graph Changes

### `tradingagents/graph/conditional_logic.py`

Add routing function:

```python
def route_post_pipeline(state: AgentState) -> str:
    if state.get("target_option") is not None:
        return "Option Trade Evaluator"
    if state.get("existing_stock_position") is not None:
        return "Stock Position Reviewer"
    if state.get("existing_option_position") is not None:
        return "Option Position Reviewer"
    return END
```

### `tradingagents/graph/setup.py`

Replace:
```python
workflow.add_edge("Portfolio Manager", "Option Trade Evaluator")
workflow.add_edge("Option Trade Evaluator", END)
```

With:
```python
workflow.add_node("Stock Position Reviewer", stock_reviewer_node)
workflow.add_node("Option Position Reviewer", option_reviewer_node)

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

---

## TradingAgentsGraph Changes (`trading_graph.py`)

### `propagate()` signature extension

```python
def propagate(
    self,
    company_name,
    trade_date,
    target_option: "TargetOption | None" = None,
    existing_stock_position: "ExistingStockPosition | None" = None,
    existing_option_position: "ExistingOptionPosition | None" = None,
):
```

### Cache bypass paths

Extend the existing cache-bypass block to cover all three reviewer types:

```python
if target_option is not None and cached_path.exists():
    return None, self._evaluate_option_only(...)

if existing_stock_position is not None and cached_path.exists():
    return None, self._review_stock_position_only(...)

if existing_option_position is not None and cached_path.exists():
    return None, self._review_option_position_only(...)
```

Add two new private methods mirroring `_evaluate_option_only()`:
- `_review_stock_position_only(company_name, trade_date, position, cached_path)`
- `_review_option_position_only(company_name, trade_date, position, cached_path)`

Both load the cached JSON, inject the position into state, run the node, persist the review back to the JSON, and return the report.

---

## CLI Changes

### `cli/utils.py`

Add `ask_existing_position(ticker)`:
- Prompts for mode: new option strategy / existing stock / existing option / skip
- Always returns a dict with exactly one non-None value:
  ```python
  {"target_option": TargetOption | None,
   "existing_stock_position": ExistingStockPosition | None,
   "existing_option_position": ExistingOptionPosition | None}
  ```
- If new option strategy: delegates to existing `ask_option_strategy()` → sets `target_option`
- If existing stock: prompts entry_price and shares → sets `existing_stock_position`
- If existing option: reuses leg-collection flow from `ask_option_strategy()`, then asks net_premium (net debit positive / net credit negative) and contracts → sets `existing_option_position`
- If skip: all three fields are None

### `cli/main.py`

**`get_user_selections()`**: Step 3 calls `ask_existing_position(ticker)` instead of `ask_option_strategy(ticker)` directly. Returns dict now includes `existing_stock_position` and `existing_option_position` alongside `target_option`.

**`MessageBuffer`**: Add `stock_position_review` and `option_position_review` report slots.

**`run_analysis()`**:
- Pass new position fields to `TradingAgentsGraph.propagate()`
- After analysis completes, display and save position review reports in a separate section from the main report
- Report files saved under a new subdirectory: `6_position_review/`

**Display**: Position review panel shown after the main 5-team report, clearly labeled "Position Review" so it's visually distinct.

---

## Propagation Changes

In `tradingagents/graph/propagation.py`, `create_initial_state()` must initialise the four new fields so the graph state is always well-typed:

```python
"existing_stock_position":  existing_stock_position,   # passed in from propagate()
"existing_option_position": existing_option_position,  # passed in from propagate()
"stock_position_review":    None,
"option_position_review":   None,
```

---

## Report Persistence

Main cache JSON (`full_states_log_{date}.json`) gets two new optional keys:
- `"stock_position_review"`: serialized StockPositionReviewReport
- `"option_position_review"`: serialized OptionPositionReviewReport

These are written by the cache-bypass methods and the graph's `_run_graph()` state flush. The main analysis fields are never modified by position review logic.

---

## What Is Not In Scope

- Multiple positions in one run
- Tax lot tracking or realized P&L
- Per-leg premium tracking for multi-leg option positions
- Portfolio-level position aggregation
- Legging out of individual option legs
