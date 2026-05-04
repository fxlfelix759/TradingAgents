"""Stocktwits public API integration — no authentication required."""

import requests

_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_HEADERS = {"User-Agent": "TradingAgent/1.0"}

# Known exchange suffixes to strip before querying Stocktwits.
# Share-class suffixes like .B (BRK.B) and .A are intentionally excluded.
_EXCHANGE_SUFFIXES = {
    ".TO",
    ".L",
    ".HK",
    ".T",
    ".AX",
    ".PA",
    ".DE",
    ".F",
    ".MI",
    ".AS",
    ".BR",
    ".MC",
    ".SW",
    ".ST",
    ".OL",
    ".CO",
    ".HE",
}


def _strip_suffix(ticker: str) -> str:
    """Remove exchange suffix (e.g. .TO, .L, .HK, .T) and uppercase."""
    upper = ticker.upper()
    dot = upper.rfind(".")
    if dot != -1 and upper[dot:] in _EXCHANGE_SUFFIXES:
        return upper[:dot]
    return upper


def get_stocktwits_messages(ticker: str, curr_date: str) -> str:
    """
    Fetch the 30 most recent Stocktwits messages for a ticker.

    The API always returns the 30 most recent messages regardless of the date
    passed; curr_date is used for labelling only.

    Returns a formatted string with per-message details and a sentiment
    summary (Bullish / Bearish / Unlabelled counts).
    """
    try:
        symbol = _strip_suffix(ticker)
        url = _API_URL.format(symbol=symbol)

        resp = requests.get(url, headers=_HEADERS, timeout=10)

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
            sentiment = (
                sentiment_obj.get("basic", "Unlabelled")
                if sentiment_obj
                else "Unlabelled"
            )

            if sentiment == "Bullish":
                bullish += 1
            elif sentiment == "Bearish":
                bearish += 1
            else:
                sentiment = "Unlabelled"
                unlabelled += 1

            lines.append(
                f"### {i}. @{username} [{sentiment}] — {created_at} UTC\n{body}"
            )

        summary = f"Sentiment summary: Bullish: {bullish} | Bearish: {bearish} | Unlabelled: {unlabelled}"
        header = f"## Stocktwits: ${symbol} (most recent messages as of {curr_date})\n\n{summary}"

        return header + "\n\n" + "\n\n".join(lines)
    except Exception as exc:
        return f"Error fetching Stocktwits data for {ticker}: {exc}"
