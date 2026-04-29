from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_options_chain(
    symbol: Annotated[str, "ticker symbol of the company"],
    trade_date: Annotated[str, "current trading date in YYYY-MM-DD format"],
    num_expirations: Annotated[int, "number of nearest expiration dates to analyse (1-5, default 3)"] = 3,
) -> str:
    """
    Fetch options chain data for the nearest expiration dates and compute key metrics:
    - Open interest (OI) distribution at each strike for calls and puts
    - Max Pain strike (price where total option holder loss is maximised — dealers pin price here)
    - Put/Call OI ratio (sentiment signal: <0.7 bullish, >1.3 bearish)
    - OI-weighted average implied volatility for calls and puts
    - Top 5 strikes by OI per expiration

    Use this tool to understand where the market is positioned, identify likely support/resistance
    from dealer hedging, and assess whether options are priced expensively (high IV) or cheaply.

    Args:
        symbol: Ticker symbol, e.g. AAPL, TSLA
        trade_date: Reference date for selecting near-term expirations (YYYY-MM-DD)
        num_expirations: How many expiry dates to analyse (default 3: weekly + monthly)
    Returns:
        Markdown-formatted options chain report with OI, volume, IV, and computed metrics.
    """
    return route_to_vendor("get_options_chain", symbol, trade_date, num_expirations)
