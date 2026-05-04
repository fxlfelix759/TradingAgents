# Stocktwits Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `get_stocktwits_posts` tool that fetches real community sentiment from Stocktwits and wire it into `social_media_analyst`, replacing the misleading "social media" prompt with one that accurately describes the two data sources available.

**Architecture:** A thin `dataflows/stocktwits.py` fetches from the public Stocktwits API and returns a formatted string. A `@tool`-decorated wrapper in `agents/utils/social_media_tools.py` exposes it to LangChain. `social_media_analyst.py` gains the new tool and a rewritten prompt.

**Tech Stack:** `requests` (already a dependency), `langchain_core.tools.tool`, Stocktwits public REST API (no auth).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tradingagents/dataflows/stocktwits.py` | HTTP fetch, ticker normalisation, response parsing, formatted output |
| Create | `tradingagents/agents/utils/social_media_tools.py` | `@tool` wrapper for LangChain |
| Modify | `tradingagents/agents/analysts/social_media_analyst.py` | Add tool, rewrite prompt |
| Create | `tests/test_stocktwits.py` | Unit tests (mocked HTTP) |

---

## Task 1: Raw Stocktwits fetcher

**Files:**
- Create: `tradingagents/dataflows/stocktwits.py`
- Create: `tests/test_stocktwits.py`

### Stocktwits API reference

```
GET https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
```

Response shape (relevant fields):
```json
{
  "response": {"status": 200},
  "messages": [
    {
      "id": 123456,
      "body": "Strong earnings beat, holding calls.",
      "created_at": "2026-05-02T14:32:00Z",
      "user": {"username": "trader123"},
      "entities": {
        "sentiment": {"basic": "Bullish"}
      }
    }
  ]
}
```

`entities.sentiment` is absent on unlabelled messages. `messages` may be absent or empty if the ticker has no activity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stocktwits.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestStripExchangeSuffix:
    def test_strips_toronto_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("CNR.TO") == "CNR"

    def test_strips_london_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("SHEL.L") == "SHEL"

    def test_strips_hongkong_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("0700.HK") == "0700"

    def test_strips_tokyo_suffix(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("7203.T") == "7203"

    def test_plain_ticker_unchanged(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("AAPL") == "AAPL"

    def test_uppercase(self):
        from tradingagents.dataflows.stocktwits import _strip_suffix
        assert _strip_suffix("aapl") == "AAPL"


@pytest.mark.unit
class TestGetStocktwitsMessages:
    def _mock_response(self, messages, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.json.return_value = {"response": {"status": status}, "messages": messages}
        return mock

    def test_returns_formatted_string_with_sentiment_summary(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        messages = [
            {
                "id": 1,
                "body": "Earnings beat!",
                "created_at": "2026-05-02T14:32:00Z",
                "user": {"username": "bull_trader"},
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "id": 2,
                "body": "Overvalued imo",
                "created_at": "2026-05-02T13:00:00Z",
                "user": {"username": "bear_trader"},
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
            {
                "id": 3,
                "body": "Watching closely",
                "created_at": "2026-05-02T12:00:00Z",
                "user": {"username": "neutral_user"},
                "entities": {},
            },
        ]
        with patch("requests.get", return_value=self._mock_response(messages)):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "Bullish: 1" in result
        assert "Bearish: 1" in result
        assert "Unlabelled: 1" in result
        assert "bull_trader" in result
        assert "Earnings beat!" in result

    def test_strips_exchange_suffix_before_api_call(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", return_value=self._mock_response([])) as mock_get:
            get_stocktwits_messages("CNR.TO", "2026-05-02")
        called_url = mock_get.call_args[0][0]
        assert "CNR.TO" not in called_url
        assert "CNR" in called_url

    def test_empty_messages_returns_no_activity_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", return_value=self._mock_response([])):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "No Stocktwits messages found" in result

    def test_api_error_returns_error_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        with patch("requests.get", side_effect=Exception("connection refused")):
            result = get_stocktwits_messages("AAPL", "2026-05-02")
        assert "Error fetching Stocktwits" in result

    def test_non_200_status_returns_error_string(self):
        from tradingagents.dataflows.stocktwits import get_stocktwits_messages
        mock = MagicMock()
        mock.status_code = 404
        mock.json.return_value = {"response": {"status": 404}, "error": ["Not found"]}
        with patch("requests.get", return_value=mock):
            result = get_stocktwits_messages("ZZZZ", "2026-05-02")
        assert "Error fetching Stocktwits" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_stocktwits.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (file doesn't exist yet).

- [ ] **Step 3: Implement `tradingagents/dataflows/stocktwits.py`**

```python
"""Stocktwits public API integration — no authentication required."""

import re
import requests

_SUFFIX_RE = re.compile(r"\.[A-Z]{1,3}$")
_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_HEADERS = {"User-Agent": "TradingAgent/1.0"}


def _strip_suffix(ticker: str) -> str:
    """Remove exchange suffix (e.g. .TO, .L, .HK, .T) and uppercase."""
    return _SUFFIX_RE.sub("", ticker.upper())


def get_stocktwits_messages(ticker: str, curr_date: str) -> str:
    """
    Fetch the 30 most recent Stocktwits messages for a ticker.

    Returns a formatted string with per-message details and a sentiment
    summary (Bullish / Bearish / Unlabelled counts).
    """
    symbol = _strip_suffix(ticker)
    url = _API_URL.format(symbol=symbol)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
    except Exception as exc:
        return f"Error fetching Stocktwits data for {ticker}: {exc}"

    if resp.status_code != 200:
        return (
            f"Error fetching Stocktwits data for {ticker}: "
            f"HTTP {resp.status_code}"
        )

    data = resp.json()
    messages = data.get("messages", [])

    if not messages:
        return f"No Stocktwits messages found for ${symbol} as of {curr_date}."

    bullish = bearish = unlabelled = 0
    lines: list[str] = []

    for i, msg in enumerate(messages, start=1):
        body = msg.get("body", "").strip()
        created_at = msg.get("created_at", "")[:16].replace("T", " ")
        username = msg.get("user", {}).get("username", "unknown")
        sentiment_obj = msg.get("entities", {}).get("sentiment")
        sentiment = sentiment_obj.get("basic", "Unlabelled") if sentiment_obj else "Unlabelled"

        if sentiment == "Bullish":
            bullish += 1
        elif sentiment == "Bearish":
            bearish += 1
        else:
            sentiment = "Unlabelled"
            unlabelled += 1

        lines.append(f"### {i}. @{username} [{sentiment}] — {created_at} UTC\n{body}")

    summary = f"Sentiment summary: Bullish: {bullish} | Bearish: {bearish} | Unlabelled: {unlabelled}"
    header = f"## Stocktwits: ${symbol} (most recent messages as of {curr_date})\n\n{summary}"

    return header + "\n\n" + "\n\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_stocktwits.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/stocktwits.py tests/test_stocktwits.py
git commit -m "feat: add Stocktwits fetcher with sentiment parsing"
```

---

## Task 2: `@tool` wrapper

**Files:**
- Create: `tradingagents/agents/utils/social_media_tools.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stocktwits.py`:

```python
@pytest.mark.unit
class TestGetStocktwitsPostsTool:
    def test_tool_calls_fetcher_and_returns_string(self):
        from tradingagents.agents.utils.social_media_tools import get_stocktwits_posts
        with patch(
            "tradingagents.agents.utils.social_media_tools.get_stocktwits_messages",
            return_value="## Stocktwits: $AAPL\n\nSentiment summary: Bullish: 5",
        ) as mock_fetch:
            result = get_stocktwits_posts.invoke({"ticker": "AAPL", "curr_date": "2026-05-02"})
        mock_fetch.assert_called_once_with("AAPL", "2026-05-02")
        assert "Stocktwits" in result

    def test_tool_has_correct_name(self):
        from tradingagents.agents.utils.social_media_tools import get_stocktwits_posts
        assert get_stocktwits_posts.name == "get_stocktwits_posts"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_stocktwits.py::TestGetStocktwitsPostsTool -v
```

Expected: `ImportError` (file doesn't exist yet).

- [ ] **Step 3: Implement `tradingagents/agents/utils/social_media_tools.py`**

```python
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.stocktwits import get_stocktwits_messages


@tool
def get_stocktwits_posts(
    ticker: Annotated[str, "Ticker symbol (e.g. AAPL, CNR.TO)"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve the 30 most recent Stocktwits messages for a ticker.
    Returns community sentiment (Bullish/Bearish labels) and post content.
    Exchange suffixes (.TO, .L, .HK, .T) are handled automatically.
    """
    return get_stocktwits_messages(ticker, curr_date)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_stocktwits.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/social_media_tools.py tests/test_stocktwits.py
git commit -m "feat: add get_stocktwits_posts LangChain tool wrapper"
```

---

## Task 3: Wire into `social_media_analyst`

**Files:**
- Modify: `tradingagents/agents/analysts/social_media_analyst.py`

- [ ] **Step 1: Replace the file contents**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.social_media_tools import get_stocktwits_posts


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_stocktwits_posts,
        ]

        system_message = (
            "You are a social media and sentiment analyst. Your job is to gauge public opinion"
            " and community sentiment for a specific stock over the past week using two tools:\n\n"
            "1. `get_news(ticker, start_date, end_date)` — searches recent news articles and press"
            " coverage. Use this to find company-specific headlines and events that are driving"
            " sentiment.\n"
            "2. `get_stocktwits_posts(ticker, curr_date)` — fetches the 30 most recent messages"
            " from Stocktwits, a social platform used by retail traders and investors. Each message"
            " includes a Bullish or Bearish label where the author chose to add one.\n\n"
            "Write a structured report covering:\n"
            "- **Stocktwits sentiment breakdown**: Bullish/Bearish/Unlabelled counts and the key"
            " themes in community posts (what are bulls saying? what are bears saying?)\n"
            "- **News-driven sentiment**: notable headlines and how the market appears to be"
            " reacting to them\n"
            "- **Overall sentiment verdict**: synthesise both sources into a single directional"
            " read (positive / negative / mixed) with supporting evidence\n\n"
            "Be specific and quote or paraphrase actual posts and headlines. Do not invent data."
            " If Stocktwits returns no messages (e.g. international ticker not listed there),"
            " note that and rely on news only."
            + " Make sure to append a Markdown table at the end summarising key sentiment signals."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
```

- [ ] **Step 2: Run the full unit suite to confirm nothing is broken**

```bash
pytest -m unit -v
```

Expected: all existing unit tests PASS, no new failures.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/agents/analysts/social_media_analyst.py
git commit -m "feat: wire get_stocktwits_posts into social_media_analyst, rewrite prompt"
```
