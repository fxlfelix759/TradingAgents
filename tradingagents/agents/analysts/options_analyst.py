"""Options Analyst: fetches live options chain data and interprets market positioning."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.options_tools import get_options_chain


def create_options_analyst(llm):

    def options_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [get_options_chain]

        system_message = (
            "You are an options market analyst. Your job is to analyse the options chain "
            "for a given stock and produce a structured report on market positioning. "
            "Call `get_options_chain` with the ticker and today's date to retrieve open interest, "
            "implied volatility, volume, and Max Pain data for the nearest 3 expiration dates.\n\n"
            "Your report must cover:\n"
            "1. **Max Pain** — state the Max Pain strike for each expiration and explain what it implies "
            "about where market makers are likely to pin the price at expiry.\n"
            "2. **OI Concentration** — identify the strike levels with the heaviest call OI (resistance) "
            "and put OI (support). These act as gravitational levels for the underlying.\n"
            "3. **Put/Call OI Ratio** — interpret the ratio: below 0.7 signals bullish positioning, "
            "above 1.3 signals bearish hedging, in between is neutral.\n"
            "4. **Implied Volatility** — comment on whether IV is elevated or subdued relative to the "
            "typical ATR/Bollinger context, and what that means for options pricing (buy vs write).\n"
            "5. **Options-derived price range** — given the OI walls and Max Pain, estimate the likely "
            "trading range until the nearest expiration.\n"
            "6. **Actionable insight** — one paragraph summarising what the options market is 'saying' "
            "about near-term sentiment and likely price behaviour.\n\n"
            "If options data is unavailable (e.g. international ticker without listed options), "
            "state that clearly and skip the analysis.\n"
            "Make sure to append a Markdown table at the end summarising Max Pain, P/C ratio, "
            "top call/put OI strikes, and avg IV for each expiration."
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
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "options_report": report,
        }

    return options_analyst_node
