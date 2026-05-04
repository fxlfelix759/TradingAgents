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
