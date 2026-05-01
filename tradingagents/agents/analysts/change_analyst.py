"""Change Analyst: diffs the current run against the most recent prior report.

The Change Analyst runs after the Research Manager has produced an investment
plan (so a current rating + bull/bear debate exists) and before the Trader, so
that downstream agents can take "what changed since last analysis" into account.

It loads the most recent prior ``full_states_log_<date>.json`` for the same
ticker (strictly older than the current ``trade_date``) and asks the quick LLM
to summarise three deltas:

1. **Rating / decision delta** — current investment plan rating vs. prior
   final_trade_decision.
2. **Price + technical-indicator delta** — drawn from the market reports of
   both runs.
3. **Bull / bear thesis delta** — drawn from each run's
   ``investment_debate_state``.

When no prior report is on file, the node writes a short stub instead of
calling the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date as _date
from pathlib import Path
from typing import Optional

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_LOG_FILENAME_RE = re.compile(r"^full_states_log_(\d{4}-\d{2}-\d{2})\.json$")


def find_prior_report(
    results_dir: str | Path,
    ticker: str,
    current_date: str,
) -> Optional[dict]:
    """Return the parsed contents of the most recent prior state log.

    Looks under ``<results_dir>/<ticker>/TradingAgentsStrategy_logs/`` for
    files named ``full_states_log_YYYY-MM-DD.json`` whose date is strictly
    older than ``current_date``. Returns the parsed JSON of the most recent
    match (with an injected ``trade_date`` if the JSON itself is missing one),
    or ``None`` if no such file exists or the directory is missing.

    The function is tolerant of unparseable JSON (logs a warning and skips).
    """
    log_dir = Path(results_dir) / ticker / "TradingAgentsStrategy_logs"
    if not log_dir.is_dir():
        return None

    try:
        current = _date.fromisoformat(current_date)
    except ValueError:
        logger.warning("Could not parse current_date %r as ISO date", current_date)
        return None

    candidates: list[tuple[_date, Path]] = []
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        match = _LOG_FILENAME_RE.match(entry.name)
        if not match:
            continue
        try:
            entry_date = _date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if entry_date < current:
            candidates.append((entry_date, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    prior_date, prior_path = candidates[0]
    try:
        with prior_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read prior state log %s: %s", prior_path, e)
        return None

    data.setdefault("trade_date", prior_date.isoformat())
    return data


def _truncate(text: str, limit: int = 4000) -> str:
    """Trim long fields so the prompt stays within sensible limits."""
    if not text:
        return "(none)"
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


def _format_no_prior_stub(ticker: str, trade_date: str) -> str:
    return (
        f"# Change Report — {ticker} on {trade_date}\n\n"
        f"No prior report on file for {ticker}. This is the first analysis "
        f"in the local results directory, so there is no baseline to diff "
        f"against."
    )


def create_change_analyst(llm):
    """Build the Change Analyst node.

    The node is intentionally LLM-driven (rather than a structured-output
    agent) so it can produce free-form prose comparing the two reports.
    """

    def change_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        config = get_config()
        results_dir = config.get("results_dir", "")

        prior = find_prior_report(results_dir, ticker, current_date)

        if prior is None:
            return {"change_report": _format_no_prior_stub(ticker, current_date)}

        prior_date = prior.get("trade_date", "unknown date")

        # Pull current-run fields. Research Manager has already written the
        # latest investment_plan + debate state by the time we run.
        current_market = state.get("market_report", "")
        current_invest_plan = state.get("investment_plan", "")
        current_debate = state.get("investment_debate_state", {}) or {}
        current_bull = current_debate.get("bull_history", "")
        current_bear = current_debate.get("bear_history", "")

        # Pull prior-run fields. Older logs may not have every field, so use
        # .get() defensively. We compare against final_trade_decision since
        # that is the cleanest record of the prior rating.
        prior_decision = prior.get("final_trade_decision", "")
        prior_market = prior.get("market_report", "")
        prior_debate = prior.get("investment_debate_state", {}) or {}
        prior_bull = prior_debate.get("bull_history", "")
        prior_bear = prior_debate.get("bear_history", "")

        instrument_context = build_instrument_context(ticker)

        prompt = f"""You are the Change Analyst. Your job is to compare the prior analysis
of {ticker} (dated {prior_date}) with the current analysis (dated {current_date})
and produce a concise report describing **what has changed**. Do not re-do the
analysis from scratch — focus only on deltas the trader needs to know about.

{instrument_context}

Structure your report with exactly these three sections:

## 1. Rating & Decision Delta
Compare the prior final trade decision with the current Research Manager
investment plan. Call out: prior rating vs. current rating, any change in
direction (e.g. Buy → Hold), and the most important reason driving the change.
If the rating is unchanged, say so explicitly and note whether the conviction
appears stronger, weaker, or the same.

## 2. Price & Technical-Indicator Delta
Using the two market reports below, summarise how price action and the key
technical indicators have moved between {prior_date} and {current_date}.
Quote specific numbers where the reports give them (e.g. RSI levels, moving
averages, support/resistance, MACD direction). If a new indicator regime has
formed (e.g. golden cross, breakout, breakdown), flag it.

## 3. Bull / Bear Thesis Delta
Summarise how the bull and bear arguments have evolved. New arguments? Old
arguments dropped? Has either side gained or lost evidence? Keep this tight —
2-4 bullet points per side.

Finish with a one-sentence "Bottom line" that states the single most important
thing that has changed for the trader.

---

PRIOR ANALYSIS ({prior_date})
============================

Prior final trade decision:
{_truncate(prior_decision)}

Prior market report:
{_truncate(prior_market)}

Prior bull case:
{_truncate(prior_bull, 2000)}

Prior bear case:
{_truncate(prior_bear, 2000)}

CURRENT ANALYSIS ({current_date})
=================================

Current investment plan (from Research Manager):
{_truncate(current_invest_plan)}

Current market report:
{_truncate(current_market)}

Current bull case:
{_truncate(current_bull, 2000)}

Current bear case:
{_truncate(current_bear, 2000)}
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        report = getattr(response, "content", None) or str(response)

        return {"change_report": report}

    return change_analyst_node
