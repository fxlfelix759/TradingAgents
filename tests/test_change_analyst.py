"""Tests for the Change Analyst node and its prior-report loader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts.change_analyst import (
    create_change_analyst,
    find_prior_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_state_log(results_dir: Path, ticker: str, date: str, payload: dict) -> Path:
    """Drop a fake full_states_log_<date>.json under the standard path."""
    log_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"full_states_log_{date}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# find_prior_report
# ---------------------------------------------------------------------------

def test_find_prior_report_returns_none_when_dir_missing(tmp_path):
    assert find_prior_report(tmp_path, "NVDA", "2026-04-28") is None


def test_find_prior_report_returns_none_when_no_logs(tmp_path):
    (tmp_path / "NVDA" / "TradingAgentsStrategy_logs").mkdir(parents=True)
    assert find_prior_report(tmp_path, "NVDA", "2026-04-28") is None


def test_find_prior_report_picks_most_recent_strictly_older(tmp_path):
    _write_state_log(tmp_path, "NVDA", "2026-01-15", {"final_trade_decision": "old"})
    _write_state_log(tmp_path, "NVDA", "2026-03-01", {"final_trade_decision": "newer"})
    # Same-day log must be excluded — diff is "since last", not "vs self".
    _write_state_log(tmp_path, "NVDA", "2026-04-28", {"final_trade_decision": "today"})
    # Future-dated file should be ignored too.
    _write_state_log(tmp_path, "NVDA", "2026-05-10", {"final_trade_decision": "future"})

    prior = find_prior_report(tmp_path, "NVDA", "2026-04-28")

    assert prior is not None
    assert prior["final_trade_decision"] == "newer"
    assert prior["trade_date"] == "2026-03-01"


def test_find_prior_report_skips_unparseable_filenames(tmp_path):
    log_dir = tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True)
    # Wrong filename shape — must be ignored, not crash.
    (log_dir / "garbage.json").write_text("{}", encoding="utf-8")
    (log_dir / "full_states_log_not-a-date.json").write_text("{}", encoding="utf-8")
    _write_state_log(tmp_path, "NVDA", "2026-02-01", {"final_trade_decision": "real"})

    prior = find_prior_report(tmp_path, "NVDA", "2026-04-28")

    assert prior is not None
    assert prior["final_trade_decision"] == "real"


def test_find_prior_report_handles_corrupt_json(tmp_path):
    log_dir = tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "full_states_log_2026-02-01.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert find_prior_report(tmp_path, "NVDA", "2026-04-28") is None


def test_find_prior_report_bad_current_date_returns_none(tmp_path):
    _write_state_log(tmp_path, "NVDA", "2026-02-01", {"final_trade_decision": "ok"})
    assert find_prior_report(tmp_path, "NVDA", "not-a-date") is None


# ---------------------------------------------------------------------------
# create_change_analyst node behaviour
# ---------------------------------------------------------------------------

def test_change_analyst_writes_stub_when_no_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tradingagents.agents.analysts.change_analyst.get_config",
        lambda: {"results_dir": str(tmp_path)},
    )

    llm = MagicMock()
    node = create_change_analyst(llm)

    out = node({
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-28",
        "market_report": "current market",
        "investment_plan": "Rating: Buy",
        "investment_debate_state": {"bull_history": "", "bear_history": ""},
    })

    assert "change_report" in out
    assert "No prior report" in out["change_report"]
    assert "NVDA" in out["change_report"]
    # No prior → must not invoke the LLM.
    llm.invoke.assert_not_called()


def test_change_analyst_invokes_llm_when_prior_exists(tmp_path, monkeypatch):
    _write_state_log(
        tmp_path,
        "NVDA",
        "2026-03-01",
        {
            "final_trade_decision": "Rating: Hold",
            "market_report": "prior market notes; RSI 55",
            "investment_debate_state": {
                "bull_history": "Bull: AI demand strong",
                "bear_history": "Bear: valuation rich",
            },
        },
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.change_analyst.get_config",
        lambda: {"results_dir": str(tmp_path)},
    )

    fake_response = MagicMock()
    fake_response.content = "## 1. Rating & Decision Delta\nHold → Buy."
    llm = MagicMock()
    llm.invoke.return_value = fake_response

    node = create_change_analyst(llm)
    out = node({
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-28",
        "market_report": "current market notes; RSI 68",
        "investment_plan": "Rating: Buy",
        "investment_debate_state": {
            "bull_history": "Bull: new product launch",
            "bear_history": "Bear: rate-cut delay",
        },
    })

    llm.invoke.assert_called_once()
    prompt = llm.invoke.call_args.args[0]
    # Prompt must surface both reports so the diff is grounded.
    assert "2026-03-01" in prompt
    assert "2026-04-28" in prompt
    assert "Rating: Hold" in prompt
    assert "Rating: Buy" in prompt
    assert "RSI 55" in prompt and "RSI 68" in prompt
    assert out["change_report"] == fake_response.content
