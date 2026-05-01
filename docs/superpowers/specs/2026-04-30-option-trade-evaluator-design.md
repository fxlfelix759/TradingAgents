# Option Trade Evaluator — Design Spec

**Date:** 2026-04-30
**Status:** Approved

---

## Overview

Add an optional option trade evaluation step to the existing TradingAgents pipeline. A user can input a specific option strategy (single-leg or multi-leg) alongside a ticker and date. After the full analyst pipeline runs, a new `OptionTradeEvaluator` node evaluates the contract against the established directional thesis, produces a structured report with verdict, risk analysis, parameter tweaks, and strategy alternatives. If an analysis report already exists for the same ticker and date, the pipeline is skipped and the evaluator runs directly against the cached report.

---

## Architecture

### Graph change

```
... → Portfolio Manager → OptionTradeEvaluator → END
                          (no-op if no target_option)
```

`OptionTradeEvaluator` is a new LangGraph node added at the end of the graph in `tradingagents/graph/setup.py`. It is wired unconditionally after Portfolio Manager; the node itself short-circuits immediately when `target_option` is absent from state.

### Cache reuse path

`TradingAgentsGraph.propagate()` gains an optional `target_option: TargetOption` parameter. At call time:

1. Check for an existing log file at `~/.tradingagents/logs/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`.
2. If found and `target_option` is set: load the JSON, reconstruct the minimal state needed, and call a new `evaluate_option_only()` method that runs only the `OptionTradeEvaluator` node. No LLM calls are made for the full analyst pipeline.
3. If not found: run the full pipeline with `target_option` set in the initial state so the evaluator fires at the end.

This makes repeated option evaluations against the same underlying analysis instant and cost-free.

---

## Data Layer

### New function: `get_full_options_chain_for_target`

Location: `tradingagents/dataflows/y_finance.py`

```python
def get_full_options_chain_for_target(
    ticker: str,
    target_expiry: str,       # YYYY-MM-DD
    num_neighbors: int = 2,   # number of expiries after target to include (plus 1 prior if available)
) -> str:
```

Returns a markdown report containing, for each of the target expiry plus up to 2 neighboring expiries:

- Full calls and puts tables: `contractSymbol`, `strike`, `bid`, `ask`, `lastPrice`, `volume`, `openInterest`, `impliedVolatility`, `inTheMoney`, `lastTradeDate`
- Computed Greeks per row via Black-Scholes (see below)
- Max Pain strike
- Put/Call OI ratio
- The target contract row(s) called out explicitly at the top of each expiry section

For multi-leg strategies with different expiries (e.g. calendars), all unique expiry dates are fetched with their own neighbors.

### Greek computation utility

Location: `tradingagents/dataflows/options_greeks.py`

```python
def compute_greeks(S, K, T, r, sigma, option_type) -> dict:
    # Black-Scholes delta, gamma, theta, vega
    # S: current stock price
    # K: strike
    # T: time to expiry in years
    # r: risk-free rate (fetched from ^IRX via yfinance)
    # sigma: implied volatility (from chain)
    # option_type: "call" or "put"
```

Uses `scipy.stats.norm` (already a transitive dependency). Risk-free rate is fetched once per session from `yf.Ticker("^IRX")` and cached.

### New LangChain tool: `get_full_options_chain`

Location: `tradingagents/agents/utils/options_tools.py`

Wraps `get_full_options_chain_for_target`. Registered exclusively on the `OptionTradeEvaluator`'s tool node — not added to any existing tool node.

---

## Input Model

Location: `tradingagents/agents/utils/agent_states.py`

```python
class OptionLeg(BaseModel):
    action: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    strike: float
    expiration: str          # YYYY-MM-DD

class TargetOption(BaseModel):
    ticker: str
    strategy: str            # e.g. "long_call", "call_debit_spread", "iron_condor"
    legs: list[OptionLeg]
    user_notes: Optional[str] = None  # budget constraints, personal rationale
```

`AgentState` gains two new optional fields:

```python
target_option: Optional[TargetOption]
option_evaluation_report: Optional[str]   # markdown output from evaluator
```

`propagation.py` initialises both to `None` in `create_initial_state()`.

---

## Strategy Taxonomy

The CLI and evaluator recognise the following strategies (value of `TargetOption.strategy`):

| Category | Strategy | Legs |
|----------|----------|------|
| Single Leg | `long_call` | 1 buy call |
| Single Leg | `long_put` | 1 buy put |
| Single Leg | `short_call` | 1 sell call |
| Single Leg | `short_put` | 1 sell put |
| Vertical Spread | `call_debit_spread` | buy lower call / sell higher call |
| Vertical Spread | `call_credit_spread` | sell lower call / buy higher call |
| Vertical Spread | `put_debit_spread` | buy higher put / sell lower put |
| Vertical Spread | `put_credit_spread` | sell higher put / buy lower put |
| Calendar | `call_calendar` | buy far call / sell near call (same strike) |
| Calendar | `put_calendar` | buy far put / sell near put (same strike) |
| Volatility | `straddle` | buy call + buy put (same strike, same expiry) |
| Volatility | `strangle` | buy OTM call + buy OTM put (same expiry) |
| Multi-Leg | `iron_condor` | sell OTM call spread + sell OTM put spread |
| Multi-Leg | `iron_butterfly` | sell ATM straddle + buy OTM wings |

---

## OptionTradeEvaluator Agent

Location: `tradingagents/agents/analysts/option_trade_evaluator.py`

Uses `deep_thinking_llm` with structured output. Calls `get_full_options_chain` as a tool to fetch chain data before reasoning.

### Prompt context injected

- `target_option` — the full strategy with all legs and user notes
- `final_trade_decision` + `investment_plan` — PM's directional verdict
- All analyst reports (market, fundamentals, news, sentiment, options overview)

### Output schema

```python
class ParameterTweak(BaseModel):
    legs: list[OptionLeg]       # full revised leg set
    rationale: str              # why this is better
    estimated_cost_change: str  # e.g. "~$30 cheaper per contract"

class StrategyAlternative(BaseModel):
    strategy: str               # e.g. "Bull Call Spread"
    legs: list[OptionLeg]
    rationale: str
    tradeoff: str               # what you give up vs the original

class OptionEvaluationReport(BaseModel):
    verdict: Literal["Strong Buy", "Buy", "Neutral", "Avoid", "Strong Avoid"]
    thesis_alignment: str           # alignment with PM's directional verdict
    contract_analysis: str          # IV context, Greeks, bid-ask spread, liquidity
    risk_assessment: str            # theta decay, breakeven, max loss, IV crush risk
    parameter_tweaks: list[ParameterTweak]
    strategy_alternatives: list[StrategyAlternative]
    constraints_acknowledged: str   # how user_notes were interpreted and respected
    summary: str                    # one-paragraph executive summary
```

The serialised report is stored as markdown in `state["option_evaluation_report"]` and written into the existing JSON log file alongside the other reports.

### Constraint handling

`user_notes` are injected into the system prompt as hard constraints. The evaluator:
- Suppresses suggestions that violate stated constraints (e.g. no wider spreads if budget is stated)
- Ranks alternatives by cost when a budget is given
- Explicitly acknowledges each constraint in `constraints_acknowledged`

---

## CLI Changes

Location: `cli/main.py`

The interactive prompt flow (using `questionary`) adds the following after the existing ticker and date questions:

```
1. Ticker?                         → NVDA
2. Analysis date?                  → 2026-04-30
   [cache check happens here]
   "Found existing report for NVDA on 2026-04-30. Using cached analysis."

3. Evaluate a specific option strategy? → Yes / No
   [if No: proceed as today]

4. Category?
   → Single Leg / Vertical Spread / Calendar / Volatility / Multi-Leg

5. Strategy?                       → [options filtered by category]

   [leg prompts, repeated per leg with leg label shown]
6. [Leg 1 - Buy] Strike?           → 130
7. [Leg 1 - Buy] Expiration?       → 2026-05-16
8. [Leg 2 - Sell] Strike?          → 140
9. [Leg 2 - Sell] Expiration?      → 2026-05-16

10. Any constraints or context? (optional, Enter to skip)
    e.g. "Max $200/strategy. Don't widen the spread."
```

Single-leg strategies skip the leg label prefix. Calendar strategies prompt for two different expiration dates. Straddles/strangles prompt for two strikes. Iron condors prompt for 4 strikes.

---

## Files Changed / Created

| File | Change |
|------|--------|
| `tradingagents/agents/utils/agent_states.py` | Add `OptionLeg`, `TargetOption`, `OptionEvaluationReport` models; add `target_option` and `option_evaluation_report` to `AgentState` |
| `tradingagents/dataflows/options_greeks.py` | **New** — Black-Scholes Greek computation utility |
| `tradingagents/dataflows/y_finance.py` | Add `get_full_options_chain_for_target` |
| `tradingagents/dataflows/interface.py` | Route `get_full_options_chain` tool to yfinance |
| `tradingagents/agents/utils/options_tools.py` | Add `get_full_options_chain` LangChain tool |
| `tradingagents/agents/analysts/option_trade_evaluator.py` | **New** — `OptionTradeEvaluator` agent node |
| `tradingagents/agents/__init__.py` | Export `create_option_trade_evaluator` |
| `tradingagents/graph/setup.py` | Add evaluator node; wire `Portfolio Manager → OptionTradeEvaluator → END` |
| `tradingagents/graph/trading_graph.py` | Add `target_option` param to `propagate()`; add `evaluate_option_only()` cache path; log `option_evaluation_report` |
| `tradingagents/graph/propagation.py` | Initialise `target_option=None`, `option_evaluation_report=None` in state |
| `cli/main.py` | Add multi-level strategy selection and per-leg prompts |
| `tests/test_option_trade_evaluator.py` | **New** — unit tests |

---

## Out of Scope

- Real-time Greeks from a broker API (Black-Scholes approximation is sufficient)
- Position sizing or portfolio margin calculations
- Backtesting option strategies
- Support for exotic options (barriers, binary, etc.)
