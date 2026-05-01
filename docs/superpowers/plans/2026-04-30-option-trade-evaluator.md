# Option Trade Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional post-PM `OptionTradeEvaluator` node that evaluates a user-specified option strategy against the established directional thesis, suggesting parameter tweaks and strategy alternatives.

**Architecture:** A new `OptionTradeEvaluator` node runs after Portfolio Manager; it short-circuits when no `target_option` is in state. The evaluator calls `get_full_options_chain_for_target()` directly (not via tool-call loop) to fetch full chain data with computed Black-Scholes Greeks, then invokes `deep_thinking_llm` with structured output. If a cached JSON report already exists for the same ticker+date, `trading_graph.py` detects it and calls `evaluate_option_only()` directly, skipping the full pipeline. The CLI adds tiered strategy selection with per-leg prompts after the date input, branching on cache detection.

**Tech Stack:** Python, LangGraph, LangChain, Pydantic v2, yfinance, scipy (transitive dep), questionary, rich, typer

---

## File Map

| File | Action |
|------|--------|
| `tradingagents/dataflows/options_greeks.py` | **New** — Black-Scholes Greeks + risk-free rate fetch |
| `tradingagents/agents/schemas.py` | Add `OptionLeg`, `TargetOption`, `ParameterTweak`, `StrategyAlternative`, `OptionEvaluationReport`, `render_option_evaluation` |
| `tradingagents/agents/utils/agent_states.py` | Add `target_option` and `option_evaluation_report` fields to `AgentState` |
| `tradingagents/graph/propagation.py` | Initialise new state fields to `None` |
| `tradingagents/dataflows/y_finance.py` | Add `get_full_options_chain_for_target` |
| `tradingagents/dataflows/interface.py` | Add `get_full_options_chain_for_target` to `VENDOR_METHODS` routing |
| `tradingagents/agents/analysts/option_trade_evaluator.py` | **New** — `create_option_trade_evaluator` agent factory |
| `tradingagents/agents/__init__.py` | Export `create_option_trade_evaluator` |
| `tradingagents/graph/setup.py` | Wire evaluator node after Portfolio Manager |
| `tradingagents/graph/trading_graph.py` | Add `target_option` param to `propagate()`; add `evaluate_option_only()`; log evaluation report |
| `cli/utils.py` | Add `ask_option_strategy()` helper (tiered questionary flow) |
| `cli/main.py` | Add cache check + option strategy prompts after date step |
| `tests/test_options_greeks.py` | **New** — unit tests for Black-Scholes |
| `tests/test_option_trade_evaluator.py` | **New** — unit tests for evaluator, schema, and data function |

---

## Task 1: Black-Scholes Greeks Utility

**Files:**
- Create: `tradingagents/dataflows/options_greeks.py`
- Create: `tests/test_options_greeks.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_options_greeks.py`:

```python
import math
import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.options_greeks import compute_greeks, get_risk_free_rate


def test_call_delta_atm():
    """ATM call delta should be close to 0.5."""
    greeks = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert 0.55 < greeks["delta"] < 0.65  # ATM call slightly above 0.5


def test_put_delta_atm():
    """ATM put delta should be negative and close to -0.5."""
    greeks = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert -0.5 < greeks["delta"] < -0.35


def test_call_put_delta_sum():
    """Call delta + |put delta| should be approximately 1 (put-call parity)."""
    call = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    put = compute_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert abs(call["delta"] + put["delta"] - 1.0) < 0.01


def test_gamma_positive():
    """Gamma is always positive for both calls and puts."""
    for otype in ("call", "put"):
        g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type=otype)
        assert g["gamma"] > 0


def test_theta_negative():
    """Theta should be negative (time decay costs the holder)."""
    for otype in ("call", "put"):
        g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type=otype)
        assert g["theta"] < 0


def test_vega_positive():
    """Vega is always positive for both calls and puts."""
    for otype in ("call", "put"):
        g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type=otype)
        assert g["vega"] > 0


def test_deep_itm_call_delta_near_one():
    """Deep ITM call delta should be close to 1."""
    greeks = compute_greeks(S=150, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
    assert greeks["delta"] > 0.95


def test_deep_otm_call_delta_near_zero():
    """Deep OTM call delta should be close to 0."""
    greeks = compute_greeks(S=50, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
    assert greeks["delta"] < 0.05


def test_near_expiry_raises_no_error():
    """Very short time to expiry (T=0.001) should not raise."""
    greeks = compute_greeks(S=100, K=100, T=0.001, r=0.05, sigma=0.2, option_type="call")
    assert "delta" in greeks


def test_get_risk_free_rate_mocked():
    """get_risk_free_rate returns a float between 0 and 1."""
    mock_ticker = MagicMock()
    mock_ticker.fast_info.last_price = 5.25  # 5.25% annualized
    with patch("tradingagents.dataflows.options_greeks.yf.Ticker", return_value=mock_ticker):
        r = get_risk_free_rate()
    assert abs(r - 0.0525) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_options_greeks.py -v
```
Expected: `ModuleNotFoundError: No module named 'tradingagents.dataflows.options_greeks'`

- [ ] **Step 3: Implement `options_greeks.py`**

Create `tradingagents/dataflows/options_greeks.py`:

```python
"""Black-Scholes Greeks computation for options evaluation."""

from __future__ import annotations

import math
import logging
from functools import lru_cache

import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)

_FALLBACK_RATE = 0.05  # fallback when ^IRX is unavailable


def get_risk_free_rate() -> float:
    """Fetch the 3-month T-bill rate from ^IRX; fall back to 5% on error."""
    try:
        ticker = yf.Ticker("^IRX")
        rate_pct = ticker.fast_info.last_price
        if rate_pct and rate_pct > 0:
            return float(rate_pct) / 100.0
    except Exception as exc:
        logger.warning("Could not fetch ^IRX risk-free rate (%s); using %.0f%%", exc, _FALLBACK_RATE * 100)
    return _FALLBACK_RATE


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict:
    """Compute Black-Scholes Greeks for a single option contract.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiry in years (use at least 0.001 to avoid division by zero)
        r: Risk-free rate as a decimal (e.g. 0.05 for 5%)
        sigma: Implied volatility as a decimal (e.g. 0.30 for 30%)
        option_type: "call" or "put"

    Returns:
        dict with keys: delta, gamma, theta (daily), vega (per 1% IV move)
    """
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-6)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% change in IV

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_options_greeks.py -v
```
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/dataflows/options_greeks.py tests/test_options_greeks.py
git commit -m "feat: add Black-Scholes Greeks computation utility"
```

---

## Task 2: Pydantic Models — Input and Output Schemas

**Files:**
- Modify: `tradingagents/agents/schemas.py` (append to existing file)
- Test: `tests/test_option_trade_evaluator.py` (partial — schema tests only)

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_option_trade_evaluator.py` (schema section only for now):

```python
import pytest
from tradingagents.agents.schemas import (
    OptionLeg,
    TargetOption,
    ParameterTweak,
    StrategyAlternative,
    OptionEvaluationReport,
    render_option_evaluation,
)


def test_option_leg_model():
    leg = OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16")
    assert leg.strike == 130.0
    assert leg.option_type == "call"


def test_target_option_single_leg():
    opt = TargetOption(
        ticker="NVDA",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16")],
    )
    assert len(opt.legs) == 1
    assert opt.user_notes is None


def test_target_option_with_notes():
    opt = TargetOption(
        ticker="NVDA",
        strategy="call_debit_spread",
        legs=[
            OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16"),
            OptionLeg(action="sell", option_type="call", strike=140.0, expiration="2026-05-16"),
        ],
        user_notes="Max $200 per strategy.",
    )
    assert opt.user_notes == "Max $200 per strategy."
    assert len(opt.legs) == 2


def test_render_option_evaluation_contains_verdict():
    report = OptionEvaluationReport(
        verdict="Buy",
        thesis_alignment="Aligns with bullish PM verdict.",
        contract_analysis="IV is reasonable at 30%.",
        risk_assessment="Max loss is premium paid.",
        parameter_tweaks=[
            ParameterTweak(
                legs=[OptionLeg(action="buy", option_type="call", strike=128.0, expiration="2026-05-16")],
                rationale="Lower strike gives more delta.",
                estimated_cost_change="~$40 more expensive",
            )
        ],
        strategy_alternatives=[
            StrategyAlternative(
                strategy="Bull Call Spread",
                legs=[
                    OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16"),
                    OptionLeg(action="sell", option_type="call", strike=145.0, expiration="2026-05-16"),
                ],
                rationale="Reduces premium outlay.",
                tradeoff="Caps upside above $145.",
            )
        ],
        constraints_acknowledged="No constraints provided.",
        summary="This is a reasonable bullish trade.",
    )
    rendered = render_option_evaluation(report)
    assert "**Verdict**: Buy" in rendered
    assert "Thesis Alignment" in rendered
    assert "Parameter Tweaks" in rendered
    assert "Strategy Alternatives" in rendered
    assert "Summary" in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py -v
```
Expected: `ImportError` — models not yet defined.

- [ ] **Step 3: Add models and render function to `schemas.py`**

Append to `tradingagents/agents/schemas.py` (after the existing `render_pm_decision` function):

```python
# ---------------------------------------------------------------------------
# Option Trade Evaluator — input models and output schema
# ---------------------------------------------------------------------------

from typing import List, Literal


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
            "Strategy identifier, e.g. 'long_call', 'call_debit_spread', 'iron_condor'. "
            "See strategy taxonomy in the design spec."
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
            "Overall assessment of the trade. 'Strong Buy' means the strategy is well-aligned "
            "and attractively priced; 'Strong Avoid' means it conflicts with the thesis or is "
            "mispriced; 'Neutral' means it is acceptable but not optimal."
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
            "Greeks (delta/theta/vega), bid-ask spread, open interest liquidity, "
            "and whether the pricing is fair given the current IV environment."
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
            "Up to 3 alternative strategy structures that better fit the current "
            "conditions. Suppress any that violate user_notes constraints."
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
            legs_str = " / ".join(fmt_leg(l) for l in tweak.legs)
            parts += [
                f"",
                f"**Tweak {i}**: {legs_str}",
                f"- Rationale: {tweak.rationale}",
                f"- Cost change: {tweak.estimated_cost_change}",
            ]

    if report.strategy_alternatives:
        parts += ["", "### Strategy Alternatives"]
        for i, alt in enumerate(report.strategy_alternatives, 1):
            legs_str = " / ".join(fmt_leg(l) for l in alt.legs)
            parts += [
                f"",
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py -v
```
Expected: all 5 schema tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/agents/schemas.py tests/test_option_trade_evaluator.py
git commit -m "feat: add OptionLeg, TargetOption, and OptionEvaluationReport schemas"
```

---

## Task 3: AgentState Fields + Propagation Init

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Modify: `tradingagents/graph/propagation.py`

- [ ] **Step 1: Add fields to `agent_states.py`**

Add the following imports at the top of `tradingagents/agents/utils/agent_states.py`:

```python
from typing import Optional
from tradingagents.agents.schemas import TargetOption
```

Then append these two fields at the end of the `AgentState` class (after `change_report`):

```python
    target_option: Annotated[
        Optional[TargetOption],
        "User-specified option strategy to evaluate (None = skip evaluator)",
    ]
    option_evaluation_report: Annotated[
        Optional[str],
        "Markdown evaluation report produced by OptionTradeEvaluator",
    ]
```

- [ ] **Step 2: Initialise new fields in `propagation.py`**

In `tradingagents/graph/propagation.py`, inside `create_initial_state()`, add after `"change_report": ""`:

```python
            "target_option": None,
            "option_evaluation_report": None,
```

Also add `target_option: Optional["TargetOption"] = None` as a parameter to `create_initial_state()`:

```python
    def create_initial_state(
        self, company_name: str, trade_date: str, past_context: str = "", target_option=None
    ) -> Dict[str, Any]:
```

And set it in the returned dict:

```python
            "target_option": target_option,
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/ -m unit -v
```
Expected: all previously passing unit tests still PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/agents/utils/agent_states.py tradingagents/graph/propagation.py
git commit -m "feat: add target_option and option_evaluation_report to AgentState"
```

---

## Task 4: Full Options Chain Data Function

**Files:**
- Modify: `tradingagents/dataflows/y_finance.py`
- Modify: `tradingagents/dataflows/interface.py`
- Modify: `tests/test_option_trade_evaluator.py` (add data tests)

- [ ] **Step 1: Add data function tests**

Append to `tests/test_option_trade_evaluator.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
from tradingagents.dataflows.y_finance import get_full_options_chain_for_target


def _make_chain_df():
    """Minimal options chain DataFrame matching yfinance structure."""
    return pd.DataFrame({
        "contractSymbol": ["NVDA260516C00130000"],
        "lastTradeDate": [pd.Timestamp("2026-04-28")],
        "strike": [130.0],
        "lastPrice": [5.50],
        "bid": [5.40],
        "ask": [5.60],
        "change": [0.10],
        "percentChange": [1.85],
        "volume": [500],
        "openInterest": [1200],
        "impliedVolatility": [0.35],
        "inTheMoney": [False],
        "contractSize": ["REGULAR"],
        "currency": ["USD"],
    })


def test_full_options_chain_returns_string():
    """get_full_options_chain_for_target returns a non-empty markdown string."""
    mock_ticker = MagicMock()
    mock_ticker.options = ["2026-05-09", "2026-05-16", "2026-05-23"]
    mock_ticker.fast_info.last_price = 128.50
    chain = MagicMock()
    chain.calls = _make_chain_df()
    chain.puts = _make_chain_df()
    mock_ticker.option_chain.return_value = chain

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker), \
         patch("tradingagents.dataflows.y_finance.get_risk_free_rate", return_value=0.05):
        result = get_full_options_chain_for_target("NVDA", "2026-05-16", num_neighbors=1)

    assert isinstance(result, str)
    assert len(result) > 100
    assert "2026-05-16" in result


def test_full_options_chain_highlights_target_strike():
    """The function explicitly calls out the target expiry in the output."""
    mock_ticker = MagicMock()
    mock_ticker.options = ["2026-05-16", "2026-05-23"]
    mock_ticker.fast_info.last_price = 128.50
    chain = MagicMock()
    chain.calls = _make_chain_df()
    chain.puts = _make_chain_df()
    mock_ticker.option_chain.return_value = chain

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker), \
         patch("tradingagents.dataflows.y_finance.get_risk_free_rate", return_value=0.05):
        result = get_full_options_chain_for_target("NVDA", "2026-05-16")

    assert "TARGET EXPIRY" in result


def test_full_options_chain_no_options_returns_message():
    """Returns a clear message when no options are listed for the ticker."""
    mock_ticker = MagicMock()
    mock_ticker.options = []

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
        result = get_full_options_chain_for_target("FAKE", "2026-05-16")

    assert "No options data available" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py::test_full_options_chain_returns_string -v
```
Expected: `ImportError` — function not yet defined.

- [ ] **Step 3: Implement `get_full_options_chain_for_target` in `y_finance.py`**

Add the following import at the top of `tradingagents/dataflows/y_finance.py` (if not present):

```python
from tradingagents.dataflows.options_greeks import compute_greeks, get_risk_free_rate
```

Then append this function to `tradingagents/dataflows/y_finance.py`:

```python
def get_full_options_chain_for_target(
    ticker: str,
    target_expiry: str,
    num_neighbors: int = 2,
) -> str:
    """Fetch the full options chain for target_expiry plus neighbors with computed Greeks.

    Returns all strikes (not top-5) for target_expiry, 1 prior expiry if available,
    and up to num_neighbors expiries after target. Greeks are computed via Black-Scholes.
    The target expiry section is labelled '(TARGET EXPIRY)' for easy LLM identification.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        expirations = yf_retry(lambda: ticker_obj.options)

        if not expirations:
            return f"No options data available for {ticker.upper()}."

        # Identify target expiry index (exact match or nearest future)
        sorted_exps = sorted(expirations)
        target_idx = None
        for i, exp in enumerate(sorted_exps):
            if exp >= target_expiry:
                target_idx = i
                break
        if target_idx is None:
            target_idx = len(sorted_exps) - 1

        # Collect: 1 prior + target + num_neighbors after
        start = max(0, target_idx - 1)
        end = min(len(sorted_exps), target_idx + num_neighbors + 1)
        selected = sorted_exps[start:end]

        current_price_info = yf_retry(lambda: ticker_obj.fast_info)
        try:
            S = round(float(current_price_info.last_price), 2)
        except Exception:
            S = None

        r = get_risk_free_rate()
        today = datetime.today().date()

        lines = [
            f"# Full Options Chain for {ticker.upper()}",
            f"# Target expiry: {target_expiry}  |  Current price: {S if S else 'N/A'}",
            f"# Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for exp in selected:
            label = "(TARGET EXPIRY)" if exp == sorted_exps[target_idx] else ""
            chain = yf_retry(lambda e=exp: ticker_obj.option_chain(e))
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Compute time to expiry
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365, 0.001)
            except Exception:
                T = 0.25

            # Compute Greeks for each row
            def add_greeks(df, option_type):
                if S is None:
                    for col in ("delta", "gamma", "theta", "vega"):
                        df[col] = float("nan")
                    return df
                rows = []
                for _, row in df.iterrows():
                    sigma = max(float(row.get("impliedVolatility", 0.3) or 0.3), 0.001)
                    K = float(row["strike"])
                    greeks = compute_greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
                    rows.append(greeks)
                for col in ("delta", "gamma", "theta", "vega"):
                    df[col] = [g[col] for g in rows]
                return df

            calls = add_greeks(calls, "call")
            puts = add_greeks(puts, "put")

            total_call_oi = int(calls["openInterest"].fillna(0).sum())
            total_put_oi = int(puts["openInterest"].fillna(0).sum())
            pc_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else float("nan")
            max_pain = _compute_max_pain(
                calls[["strike", "openInterest"]].copy(),
                puts[["strike", "openInterest"]].copy(),
            )

            display_cols = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest",
                            "impliedVolatility", "inTheMoney", "delta", "gamma", "theta", "vega"]
            calls_display = calls[[c for c in display_cols if c in calls.columns]]
            puts_display = puts[[c for c in display_cols if c in puts.columns]]

            lines += [
                f"## Expiry: {exp} {label}",
                f"- Max Pain: {max_pain}  |  P/C OI Ratio: {pc_ratio}",
                f"- Total Call OI: {total_call_oi:,}  |  Total Put OI: {total_put_oi:,}",
                "",
                "### Calls (all strikes)",
                calls_display.to_string(index=False),
                "",
                "### Puts (all strikes)",
                puts_display.to_string(index=False),
                "",
            ]

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving full options chain for {ticker}: {e}"
```

- [ ] **Step 4: Add routing in `interface.py`**

In `tradingagents/dataflows/interface.py`, add the import at the top:

```python
from .y_finance import (
    ...
    get_full_options_chain_yfinance,
    get_full_options_chain_for_target as get_full_options_chain_for_target_yfinance,
)
```

Add to `VENDOR_METHODS` dict (after the existing `"get_options_chain"` entry):

```python
    "get_full_options_chain_for_target": {
        "yfinance": get_full_options_chain_for_target_yfinance,
    },
```

Add a direct routing function at the bottom of `interface.py` (the evaluator calls this directly, not via `route_to_vendor`):

```python
def get_full_options_chain_for_target(
    ticker: str,
    target_expiry: str,
    num_neighbors: int = 2,
) -> str:
    """Route full chain fetch to yfinance (only vendor that supports this)."""
    return get_full_options_chain_for_target_yfinance(ticker, target_expiry, num_neighbors)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py -k "chain" -v
```
Expected: all 3 chain tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/dataflows/y_finance.py tradingagents/dataflows/interface.py tests/test_option_trade_evaluator.py
git commit -m "feat: add get_full_options_chain_for_target with Greeks"
```

---

## Task 5: OptionTradeEvaluator Agent

**Files:**
- Create: `tradingagents/agents/analysts/option_trade_evaluator.py`
- Modify: `tradingagents/agents/__init__.py`
- Modify: `tests/test_option_trade_evaluator.py` (add evaluator node test)

- [ ] **Step 1: Add evaluator node test**

Append to `tests/test_option_trade_evaluator.py`:

```python
from tradingagents.agents.analysts.option_trade_evaluator import create_option_trade_evaluator
from tradingagents.agents.schemas import OptionLeg, TargetOption


def _make_target_option():
    return TargetOption(
        ticker="NVDA",
        strategy="long_call",
        legs=[OptionLeg(action="buy", option_type="call", strike=130.0, expiration="2026-05-16")],
        user_notes="Max $300 per contract.",
    )


def _make_state(target_option=None):
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-30",
        "target_option": target_option,
        "final_trade_decision": "**Rating**: Buy\n**Executive Summary**: Bullish on NVDA.",
        "investment_plan": "Buy with strong conviction.",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "market_report": "Bullish technicals.",
        "fundamentals_report": "Strong earnings.",
        "sentiment_report": "Positive sentiment.",
        "news_report": "No adverse news.",
        "options_report": "Moderate IV.",
        "messages": [],
    }


def test_evaluator_noop_when_no_target_option(mock_llm_client):
    """OptionTradeEvaluator returns empty dict when target_option is None."""
    node = create_option_trade_evaluator(mock_llm_client.get_llm())
    result = node(_make_state(target_option=None))
    assert result == {}


def test_evaluator_calls_llm_when_target_option_set(mock_llm_client):
    """OptionTradeEvaluator calls the structured LLM when target_option is present."""
    from tradingagents.agents.schemas import OptionEvaluationReport, ParameterTweak, StrategyAlternative

    mock_report = OptionEvaluationReport(
        verdict="Buy",
        thesis_alignment="Aligned.",
        contract_analysis="IV is reasonable.",
        risk_assessment="Max loss is premium.",
        parameter_tweaks=[],
        strategy_alternatives=[],
        constraints_acknowledged="Budget $300 noted.",
        summary="Good trade.",
    )

    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    with patch("tradingagents.agents.analysts.option_trade_evaluator.get_full_options_chain_for_target",
               return_value="# Full chain data..."):
        node = create_option_trade_evaluator(mock_llm_client.get_llm())
        result = node(_make_state(target_option=_make_target_option()))

    assert "option_evaluation_report" in result
    assert "**Verdict**: Buy" in result["option_evaluation_report"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py -k "evaluator" -v
```
Expected: `ImportError` — module not yet defined.

- [ ] **Step 3: Implement `option_trade_evaluator.py`**

Create `tradingagents/agents/analysts/option_trade_evaluator.py`:

```python
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

        # Collect all unique expiries across legs for chain fetch
        expiries = sorted({leg.expiration for leg in target_option.legs})
        chain_sections = []
        for expiry in expiries:
            chain_sections.append(
                get_full_options_chain_for_target(target_option.ticker, expiry, num_neighbors=2)
            )
        full_chain = "\n\n".join(chain_sections)

        instrument_context = build_instrument_context(target_option.ticker)

        # Describe the strategy for the prompt
        legs_desc = "\n".join(
            f"  - Leg {i+1}: {leg.action.upper()} {leg.option_type.upper()} "
            f"${leg.strike} exp {leg.expiration}"
            for i, leg in enumerate(target_option.legs)
        )
        strategy_block = (
            f"Strategy: {target_option.strategy}\n"
            f"Legs:\n{legs_desc}"
        )

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
5. **Parameter Tweaks**: Up to 3 same-structure tweaks (strike/expiry adjustments) that improve risk/reward. Respect user constraints.
6. **Strategy Alternatives**: Up to 3 alternative structures better suited to conditions. Suppress any that violate user constraints.
7. **Constraints**: Acknowledge each user constraint and explain how it shaped your suggestions.
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
```

- [ ] **Step 4: Export from `__init__.py`**

In `tradingagents/agents/__init__.py`, add:

```python
from .analysts.option_trade_evaluator import create_option_trade_evaluator
```

And add `"create_option_trade_evaluator"` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/test_option_trade_evaluator.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/agents/analysts/option_trade_evaluator.py tradingagents/agents/__init__.py tests/test_option_trade_evaluator.py
git commit -m "feat: add OptionTradeEvaluator agent node"
```

---

## Task 6: Graph Wiring + Cache Path

**Files:**
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/graph/trading_graph.py`

- [ ] **Step 1: Wire evaluator node in `setup.py`**

In `tradingagents/graph/setup.py`, inside `setup_graph()`, add before `workflow.add_edge("Portfolio Manager", END)`:

```python
        # Option Trade Evaluator (no-op when target_option is absent)
        option_evaluator_node = create_option_trade_evaluator(self.deep_thinking_llm)
        workflow.add_node("Option Trade Evaluator", option_evaluator_node)
```

Replace `workflow.add_edge("Portfolio Manager", END)` with:

```python
        workflow.add_edge("Portfolio Manager", "Option Trade Evaluator")
        workflow.add_edge("Option Trade Evaluator", END)
```

- [ ] **Step 2: Add `target_option` to `propagate()` and cache path in `trading_graph.py`**

In `tradingagents/graph/trading_graph.py`:

**2a.** Add import at top (if not present):
```python
from tradingagents.agents.schemas import TargetOption
```

**2b.** Update `propagate()` signature:
```python
    def propagate(self, company_name, trade_date, target_option: "TargetOption | None" = None):
```

**2c.** At the start of `propagate()`, add cache detection before the checkpoint logic:
```python
        self.ticker = company_name

        # If a cached report exists and the user only wants option evaluation, skip the pipeline.
        if target_option is not None:
            cached_path = (
                Path(self.config["results_dir"])
                / company_name
                / "TradingAgentsStrategy_logs"
                / f"full_states_log_{trade_date}.json"
            )
            if cached_path.exists():
                logger.info("Cache hit for %s on %s — running option evaluator only.", company_name, trade_date)
                report = self._evaluate_option_only(company_name, trade_date, target_option, cached_path)
                return None, report

        self._target_option = target_option  # stored for _run_graph
```

**2d.** Add `_evaluate_option_only()` method:
```python
    def _evaluate_option_only(self, company_name, trade_date, target_option, cached_path):
        """Run only the OptionTradeEvaluator using a cached analysis report."""
        with open(cached_path, encoding="utf-8") as f:
            cached = json.load(f)

        state = {
            "company_of_interest": cached.get("company_of_interest", company_name),
            "trade_date": cached.get("trade_date", str(trade_date)),
            "target_option": target_option,
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

        from tradingagents.agents.analysts.option_trade_evaluator import create_option_trade_evaluator
        evaluator = create_option_trade_evaluator(self.deep_thinking_llm)
        result = evaluator(state)
        report = result.get("option_evaluation_report", "")

        # Persist the evaluation back into the cached JSON
        cached["option_evaluation_report"] = report
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=4, ensure_ascii=False)

        return report
```

**2e.** In `_run_graph()`, pass `target_option` into initial state. Replace the `create_initial_state` call:
```python
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context,
            target_option=getattr(self, "_target_option", None),
        )
```

**2f.** In `_log_state()`, add `option_evaluation_report` to the logged dict (after `"change_report"`):
```python
            "option_evaluation_report": final_state.get("option_evaluation_report", ""),
```

- [ ] **Step 3: Verify full test suite still passes**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/ -m unit -v
```
Expected: all unit tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add tradingagents/graph/setup.py tradingagents/graph/trading_graph.py
git commit -m "feat: wire OptionTradeEvaluator into graph + cache bypass path"
```

---

## Task 7: CLI — Option Strategy Prompts

**Files:**
- Modify: `cli/utils.py`
- Modify: `cli/main.py`

- [ ] **Step 1: Add strategy helpers to `cli/utils.py`**

Append to `cli/utils.py`:

```python
from tradingagents.agents.schemas import OptionLeg, TargetOption

# Strategy taxonomy: category → [(display_name, strategy_key, legs_template)]
# Each leg_template: {"action": "buy"|"sell", "option_type": "call"|"put", "label": display}
STRATEGY_TAXONOMY = {
    "Single Leg": [
        ("Long Call", "long_call", [{"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call)"}]),
        ("Long Put", "long_put", [{"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put)"}]),
        ("Short Call", "short_call", [{"action": "sell", "option_type": "call", "label": "Leg 1 (Sell Call)"}]),
        ("Short Put", "short_put", [{"action": "sell", "option_type": "put", "label": "Leg 1 (Sell Put)"}]),
    ],
    "Vertical Spread": [
        ("Call Debit Spread", "call_debit_spread", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call — lower strike)"},
            {"action": "sell", "option_type": "call", "label": "Leg 2 (Sell Call — higher strike)"},
        ]),
        ("Call Credit Spread", "call_credit_spread", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell Call — lower strike)"},
            {"action": "buy", "option_type": "call", "label": "Leg 2 (Buy Call — higher strike)"},
        ]),
        ("Put Debit Spread", "put_debit_spread", [
            {"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put — higher strike)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell Put — lower strike)"},
        ]),
        ("Put Credit Spread", "put_credit_spread", [
            {"action": "sell", "option_type": "put", "label": "Leg 1 (Sell Put — higher strike)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy Put — lower strike)"},
        ]),
    ],
    "Calendar": [
        ("Call Calendar", "call_calendar", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call — far expiry)"},
            {"action": "sell", "option_type": "call", "label": "Leg 2 (Sell Call — near expiry, same strike)"},
        ]),
        ("Put Calendar", "put_calendar", [
            {"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put — far expiry)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell Put — near expiry, same strike)"},
        ]),
    ],
    "Volatility": [
        ("Straddle", "straddle", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy Put — same strike)"},
        ]),
        ("Strangle", "strangle", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy OTM Call)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy OTM Put)"},
        ]),
    ],
    "Multi-Leg": [
        ("Iron Condor", "iron_condor", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell OTM Call)"},
            {"action": "buy", "option_type": "call", "label": "Leg 2 (Buy further OTM Call — wing)"},
            {"action": "sell", "option_type": "put", "label": "Leg 3 (Sell OTM Put)"},
            {"action": "buy", "option_type": "put", "label": "Leg 4 (Buy further OTM Put — wing)"},
        ]),
        ("Iron Butterfly", "iron_butterfly", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell ATM Call)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell ATM Put — same strike)"},
            {"action": "buy", "option_type": "call", "label": "Leg 3 (Buy OTM Call — wing)"},
            {"action": "buy", "option_type": "put", "label": "Leg 4 (Buy OTM Put — wing)"},
        ]),
    ],
}


def ask_option_strategy(ticker: str) -> Optional[TargetOption]:
    """Interactive multi-step prompt to collect an option strategy. Returns None if skipped."""
    wants_eval = questionary.confirm(
        "Evaluate a specific option strategy?",
        default=False,
        style=questionary.Style([("highlighted", "noinherit")]),
    ).ask()

    if not wants_eval:
        return None

    # Category selection
    category = questionary.select(
        "Select strategy category:",
        choices=list(STRATEGY_TAXONOMY.keys()),
        style=questionary.Style([("selected", "fg:cyan noinherit"), ("highlighted", "noinherit")]),
    ).ask()
    if not category:
        return None

    # Strategy selection within category
    strategies = STRATEGY_TAXONOMY[category]
    strategy_display, strategy_key, legs_template = questionary.select(
        "Select strategy:",
        choices=[questionary.Choice(display, value=(display, key, template))
                 for display, key, template in strategies],
        style=questionary.Style([("selected", "fg:cyan noinherit"), ("highlighted", "noinherit")]),
    ).ask()
    if not strategy_key:
        return None

    # Collect per-leg parameters
    legs = []
    for leg_def in legs_template:
        console.print(f"\n[cyan]{leg_def['label']}[/cyan]")
        strike = questionary.text(
            "Strike price:",
            validate=lambda x: x.replace(".", "", 1).isdigit() or "Enter a numeric strike price.",
        ).ask()
        if strike is None:
            return None

        expiration = questionary.text(
            "Expiration date (YYYY-MM-DD):",
            validate=lambda x: len(x) == 10 and x[4] == "-" and x[7] == "-"
                               or "Use YYYY-MM-DD format.",
        ).ask()
        if expiration is None:
            return None

        legs.append(OptionLeg(
            action=leg_def["action"],
            option_type=leg_def["option_type"],
            strike=float(strike),
            expiration=expiration,
        ))

    # Optional user notes
    user_notes = questionary.text(
        "Any constraints or context? (optional — press Enter to skip)\n"
        "  e.g. 'Max $200/strategy. Don't widen the spread.'",
        default="",
    ).ask()

    return TargetOption(
        ticker=ticker,
        strategy=strategy_key,
        legs=legs,
        user_notes=user_notes.strip() if user_notes and user_notes.strip() else None,
    )
```

- [ ] **Step 2: Add cache detection + option prompts to `cli/main.py`**

In `cli/main.py`, add this import at the top (with other imports):

```python
from cli.utils import ask_option_strategy
```

In the `collect_analysis_parameters()` function (or wherever ticker + date are collected), after Step 2 (analysis date) and before Step 3 (output language), add a cache check and option strategy prompt block:

```python
    # Cache check: notify user if a report already exists for this ticker+date
    _cache_log_path = (
        Path(DEFAULT_CONFIG["results_dir"])
        / selected_ticker
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{analysis_date}.json"
    )
    _cache_exists = _cache_log_path.exists()
    if _cache_exists:
        console.print(
            f"[green]Found existing report for {selected_ticker} on {analysis_date}. "
            "Using cached analysis for option evaluation.[/green]"
        )

    # Step 3: Option strategy (optional)
    console.print(
        create_question_box(
            "Step 3: Option Strategy (Optional)",
            "Optionally evaluate a specific option strategy against this analysis",
        )
    )
    target_option = ask_option_strategy(selected_ticker)

    # If cache exists and no option requested, still need to run the full pipeline
    # but note the cache will be used if target_option is set.
```

Renumber the remaining steps (old Step 3 → Step 4, etc.) in the UI display strings only (the `create_question_box` titles).

Finally, add `target_option` to the returned dict:

```python
    return {
        ...
        "target_option": target_option,
        "cache_exists": _cache_exists,
    }
```

In the `analyze` command (where `graph.propagate()` or `graph.graph.stream()` is called), pass `target_option` when calling `propagate()`:

Find the line that starts the stream:
```python
        for chunk in graph.graph.stream(init_agent_state, **args):
```

Replace with a branch:

```python
        target_option = selections.get("target_option")
        cache_exists = selections.get("cache_exists", False)

        if target_option and cache_exists:
            # Cache hit path: skip streaming, run evaluator directly
            _, option_report = graph.propagate(
                selections["ticker"], selections["analysis_date"], target_option=target_option
            )
            if option_report:
                console.print(Rule("Option Trade Evaluation", style="bold cyan"))
                console.print(Markdown(option_report))
        else:
            # Normal path: stream full graph (evaluator fires at end if target_option set)
            init_agent_state = graph.propagator.create_initial_state(
                selections["ticker"], selections["analysis_date"],
                target_option=target_option,
            )
            args = graph.propagator.get_graph_args(callbacks=[stats_handler])

            trace = []
            for chunk in graph.graph.stream(init_agent_state, **args):
                # ... existing streaming handling unchanged ...
```

Also add display of `option_evaluation_report` in `display_complete_report()` and `save_report_to_disk()` by checking `final_state.get("option_evaluation_report")` and rendering it as a new section.

- [ ] **Step 3: Verify CLI imports cleanly**

```bash
cd /Users/felix/workspace/TradingAgent && python -c "from cli.main import app; print('CLI import OK')"
```
Expected: `CLI import OK`

- [ ] **Step 4: Run full test suite**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/ -m unit -v
```
Expected: all unit tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/felix/workspace/TradingAgent && git add cli/utils.py cli/main.py
git commit -m "feat: add tiered option strategy CLI prompts with cache detection"
```

---

## Task 8: Integration Smoke Test + Push

**Files:**
- Modify: `tests/test_option_trade_evaluator.py` (add smoke test)

- [ ] **Step 1: Add integration smoke test**

Append to `tests/test_option_trade_evaluator.py`:

```python
import pytest

@pytest.mark.smoke
def test_full_evaluator_pipeline_smoke(mock_llm_client):
    """End-to-end smoke: evaluator node wires correctly into a minimal graph."""
    from tradingagents.agents.analysts.option_trade_evaluator import create_option_trade_evaluator
    from tradingagents.agents.schemas import OptionEvaluationReport, ParameterTweak, StrategyAlternative

    mock_report = OptionEvaluationReport(
        verdict="Neutral",
        thesis_alignment="Partially aligned.",
        contract_analysis="IV is elevated.",
        risk_assessment="Theta decay is $8/day.",
        parameter_tweaks=[],
        strategy_alternatives=[],
        constraints_acknowledged="No constraints provided.",
        summary="Acceptable trade but not optimal.",
    )
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = mock_report
    mock_llm_client.get_llm.return_value.with_structured_output.return_value = structured_mock

    target = TargetOption(
        ticker="AAPL",
        strategy="call_debit_spread",
        legs=[
            OptionLeg(action="buy", option_type="call", strike=200.0, expiration="2026-05-30"),
            OptionLeg(action="sell", option_type="call", strike=210.0, expiration="2026-05-30"),
        ],
    )

    with patch("tradingagents.agents.analysts.option_trade_evaluator.get_full_options_chain_for_target",
               return_value="# Mocked chain"):
        node = create_option_trade_evaluator(mock_llm_client.get_llm())
        result = node(_make_state(target_option=target))

    assert result["option_evaluation_report"] is not None
    assert "Neutral" in result["option_evaluation_report"]
```

- [ ] **Step 2: Run all tests**

```bash
cd /Users/felix/workspace/TradingAgent && pytest tests/ -v
```
Expected: all tests PASS (unit + smoke).

- [ ] **Step 3: Push**

```bash
cd /Users/felix/workspace/TradingAgent && git push
```

---

## Self-Review Checklist

- [x] **Spec coverage**: All spec sections covered — Greeks (Task 1), models (Task 2), state (Task 3), data function (Task 4), evaluator agent (Task 5), graph wiring + cache path (Task 6), CLI (Task 7).
- [x] **No placeholders**: All code blocks are complete and runnable.
- [x] **Type consistency**: `OptionLeg` and `TargetOption` defined in Task 2 (schemas.py) and referenced by name in Tasks 3, 4, 5, 6, 7. `render_option_evaluation` defined in Task 2, used in Task 5. `get_full_options_chain_for_target` defined in Task 4, imported in Task 5.
- [x] **Task ordering**: Each task depends only on prior tasks. Tasks 1–2 are leaf utilities with no upstream dependencies. Tasks 3–5 depend on 1–2. Tasks 6–7 depend on 3–5.
