# Stocktwits Integration for Social Media Analyst

**Date:** 2026-05-03
**Status:** Approved

## Problem

`social_media_analyst` currently only has `get_news` as a tool, but its prompt claims to analyze "social media posts" and "sentiment data." This mismatch causes the LLM to either hallucinate social data or produce a report that's really just a news summary with a misleading label.

## Solution

Add a `get_stocktwits_posts` tool that fetches the 30 most recent Stocktwits messages for a ticker via the public Stocktwits API (no authentication required). Update the analyst prompt to accurately reflect what data it actually has.

## Architecture

### New: `tradingagents/dataflows/stocktwits.py`

Raw data fetching. Calls `https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json` using the existing `requests` dependency (no new packages needed).

Responsibilities:
- Strip exchange suffixes from tickers (`.TO`, `.L`, `.HK`, `.T`) before the API call — Stocktwits only understands plain tickers
- Parse the response: message body, `created_at`, username, and `entities.sentiment.basic` (Bullish / Bearish / None)
- Return a formatted string with per-message details and a sentiment summary (Bullish N / Bearish N / unlabelled N)
- Return a descriptive error string on API failure rather than raising

### New: `tradingagents/agents/utils/social_media_tools.py`

`@tool`-decorated wrapper following the same pattern as `news_data_tools.py`.

```python
@tool
def get_stocktwits_posts(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve the 30 most recent Stocktwits messages for a ticker, including Bullish/Bearish sentiment labels."""
```

Does not go through `route_to_vendor` — Stocktwits is not an interchangeable vendor for an existing data category; it is a standalone new source.

### Modified: `tradingagents/agents/analysts/social_media_analyst.py`

1. Add `get_stocktwits_posts` to the `tools` list alongside `get_news`
2. Rewrite the `system_message` prompt to accurately describe the two available tools and guide the LLM to synthesize news sentiment with Stocktwits community sentiment

### Not changed

- `interface.py` — no new vendor routing
- `default_config.py` — no new config keys
- No new environment variables or API keys required

## Output format (from `get_stocktwits_posts`)

```
## Stocktwits: $AAPL (30 most recent messages as of 2026-05-03)

Sentiment summary: Bullish: 12 | Bearish: 8 | Unlabelled: 10

### 1. @username [Bullish] — 2026-05-02 14:32 UTC
Strong earnings beat, holding calls through next week.

### 2. @username2 [Bearish] — 2026-05-02 13:15 UTC
...
```

## Extensibility

To add another forum (e.g., X/Twitter, Discord) in the future: create a new `dataflows/<source>.py` + `agents/utils/<source>_tools.py`, and add the tool to social_media_analyst's tools list. No changes needed to the vendor routing system.
