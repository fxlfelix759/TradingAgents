from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a news analyst. Your job is to identify news events from the past week"
            " that are relevant to a specific stock and its trading outlook.\n\n"
            "Use the available tools:\n"
            "- `get_news(ticker, start_date, end_date)` — search for company-specific headlines,"
            " earnings releases, analyst upgrades/downgrades, product launches, regulatory news,"
            " and management changes\n"
            "- `get_global_news(curr_date, look_back_days, limit)` — search for macro and"
            " geopolitical events that could affect the stock's sector or the broader market"
            " (interest rates, inflation, trade policy, sector-wide trends)\n\n"
            "Write a structured report with these sections:\n"
            "1. **Company-Specific News** — material headlines directly about the company;"
            " for each item note the date, source, and likely price impact (positive/negative/neutral)\n"
            "2. **Sector & Industry News** — developments affecting the company's peers or industry\n"
            "3. **Macro & Geopolitical Context** — relevant central bank moves, economic data,"
            " or geopolitical events that could influence this stock\n"
            "4. **News Sentiment Summary** — overall directional read from the news flow"
            " (bullish / bearish / mixed) with the top 2-3 reasons\n\n"
            "Be specific: quote headlines, cite dates, and explain the trading relevance of each"
            " item. Do not pad the report with generic market commentary unrelated to this stock."
            " Append a Markdown table summarising the key news items at the end."
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
            "news_report": report,
        }

    return news_analyst_node
