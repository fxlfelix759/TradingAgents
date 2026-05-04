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
