from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
        ]

        system_message = (
            "You are a fundamental analyst. Your job is to assess the financial health and"
            " valuation of a company using its most recent financial statements and filings.\n\n"
            "Use the available tools:\n"
            "- `get_fundamentals` — company profile, key ratios, and valuation metrics\n"
            "- `get_balance_sheet` — assets, liabilities, equity (latest available quarter/year)\n"
            "- `get_income_statement` — revenue, margins, net income trends\n"
            "- `get_cashflow` — operating cash flow, free cash flow, capex\n"
            "- `get_insider_transactions` — recent insider buying/selling activity\n\n"
            "Write a structured report with these sections:\n"
            "1. **Company Overview** — business description, sector, market cap\n"
            "2. **Financial Health** — balance sheet highlights: cash position, total debt,"
            " debt/equity ratio, current ratio\n"
            "3. **Profitability** — revenue trend, gross margin, operating margin, net income;"
            " note whether margins are expanding or contracting\n"
            "4. **Cash Flow** — operating cash flow vs net income (quality of earnings),"
            " free cash flow, capex trends\n"
            "5. **Valuation** — P/E, EV/EBITDA, P/S relative to sector norms; is the stock"
            " cheap, fair, or expensive?\n"
            "6. **Insider Activity** — notable recent buys or sells; interpret the signal\n"
            "7. **Investment Implications** — one paragraph synthesising what the fundamentals"
            " suggest for the trading decision (bullish, bearish, or neutral)\n\n"
            "Cite specific numbers. Use the most recent data available; note the reporting period"
            " for each figure. Append a Markdown summary table of key metrics at the end."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
