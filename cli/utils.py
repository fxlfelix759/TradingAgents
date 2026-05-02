import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType
from tradingagents.agents.schemas import OptionLeg, TargetOption

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    return choice


def _prompt_custom_model_id() -> str:
    """Prompt user to type a custom model ID."""
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
    ).ask().strip()


def _select_model(provider: str, mode: str) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    if provider.lower() == "openrouter":
        return select_openrouter_model()

    if provider.lower() == "azure":
        return questionary.text(
            f"Enter Azure deployment name ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a deployment name.",
        ).ask().strip()

    return _prompt_custom_model_id()


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""
    return _select_model(provider, "quick")


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""
    return _select_model(provider, "deep")

def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    # (display_name, provider_key, base_url)
    PROVIDERS = [
        ("OpenAI", "openai", "https://api.openai.com/v1"),
        ("Google", "google", None),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
        ("xAI", "xai", "https://api.x.ai/v1"),
        ("DeepSeek", "deepseek", "https://api.deepseek.com"),
        ("Qwen", "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("GLM", "glm", "https://open.bigmodel.cn/api/paas/v4/"),
        ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
        ("Azure OpenAI", "azure", None),
        ("Ollama", "ollama", "http://localhost:11434/v1"),
    ]

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(provider_key, url))
            for display, provider_key, url in PROVIDERS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    provider, url = choice
    return provider, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    return questionary.select(
        "Select Effort Level:",
        choices=[
            questionary.Choice("High (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def ask_output_language() -> str:
    """Ask for report output language."""
    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice


# ---------------------------------------------------------------------------
# Option strategy selection
# ---------------------------------------------------------------------------

STRATEGY_TAXONOMY = {
    "Single Leg": [
        ("Long Call", "long_call", [{"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call)"}]),
        ("Long Put", "long_put", [{"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put)"}]),
        ("Short Call", "short_call", [{"action": "sell", "option_type": "call", "label": "Leg 1 (Sell Call)"}]),
        ("Short Put", "short_put", [{"action": "sell", "option_type": "put", "label": "Leg 1 (Sell Put)"}]),
    ],
    "Vertical Spread": [
        ("Call Debit Spread", "call_debit_spread", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call — lower strike)"},
            {"action": "sell", "option_type": "call", "label": "Leg 2 (Sell Call — higher strike)"},
        ]),
        ("Call Credit Spread", "call_credit_spread", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell Call — lower strike)"},
            {"action": "buy", "option_type": "call", "label": "Leg 2 (Buy Call — higher strike)"},
        ]),
        ("Put Debit Spread", "put_debit_spread", [
            {"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put — higher strike)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell Put — lower strike)"},
        ]),
        ("Put Credit Spread", "put_credit_spread", [
            {"action": "sell", "option_type": "put", "label": "Leg 1 (Sell Put — higher strike)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy Put — lower strike)"},
        ]),
    ],
    "Calendar": [
        ("Call Calendar", "call_calendar", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call — far expiry)"},
            {"action": "sell", "option_type": "call", "label": "Leg 2 (Sell Call — near expiry, same strike)"},
        ]),
        ("Put Calendar", "put_calendar", [
            {"action": "buy", "option_type": "put", "label": "Leg 1 (Buy Put — far expiry)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell Put — near expiry, same strike)"},
        ]),
    ],
    "Volatility": [
        ("Straddle", "straddle", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy Call)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy Put — same strike)"},
        ]),
        ("Strangle", "strangle", [
            {"action": "buy", "option_type": "call", "label": "Leg 1 (Buy OTM Call)"},
            {"action": "buy", "option_type": "put", "label": "Leg 2 (Buy OTM Put)"},
        ]),
    ],
    "Multi-Leg": [
        ("Iron Condor", "iron_condor", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell OTM Call)"},
            {"action": "buy", "option_type": "call", "label": "Leg 2 (Buy further OTM Call — wing)"},
            {"action": "sell", "option_type": "put", "label": "Leg 3 (Sell OTM Put)"},
            {"action": "buy", "option_type": "put", "label": "Leg 4 (Buy further OTM Put — wing)"},
        ]),
        ("Iron Butterfly", "iron_butterfly", [
            {"action": "sell", "option_type": "call", "label": "Leg 1 (Sell ATM Call)"},
            {"action": "sell", "option_type": "put", "label": "Leg 2 (Sell ATM Put — same strike)"},
            {"action": "buy", "option_type": "call", "label": "Leg 3 (Buy OTM Call — wing)"},
            {"action": "buy", "option_type": "put", "label": "Leg 4 (Buy OTM Put — wing)"},
        ]),
    ],
}

_OPTION_STYLE = questionary.Style([
    ("selected", "fg:cyan noinherit"),
    ("highlighted", "noinherit"),
])


def ask_option_strategy(ticker: str) -> Optional[TargetOption]:
    """Interactive multi-step prompt to collect an option strategy. Returns None if skipped."""
    wants_eval = questionary.confirm(
        "Evaluate a specific option strategy?",
        default=False,
        style=_OPTION_STYLE,
    ).ask()
    if not wants_eval:
        return None

    category = questionary.select(
        "Select strategy category:",
        choices=list(STRATEGY_TAXONOMY.keys()),
        style=_OPTION_STYLE,
    ).ask()
    if not category:
        return None

    strategies = STRATEGY_TAXONOMY[category]
    selection = questionary.select(
        "Select strategy:",
        choices=[
            questionary.Choice(display, value=(display, key, template))
            for display, key, template in strategies
        ],
        style=_OPTION_STYLE,
    ).ask()
    if not selection:
        return None
    _display, strategy_key, legs_template = selection

    legs = []
    for leg_def in legs_template:
        console.print(f"\n[cyan]{leg_def['label']}[/cyan]")
        strike = questionary.text(
            "Strike price:",
            validate=lambda x: x.replace(".", "", 1).isdigit() or "Enter a numeric strike price.",
        ).ask()
        if strike is None:
            return None

        expiration = questionary.text(
            "Expiration date (YYYY-MM-DD):",
            validate=lambda x: (
                len(x) == 10 and x[4] == "-" and x[7] == "-"
            ) or "Use YYYY-MM-DD format.",
        ).ask()
        if expiration is None:
            return None

        legs.append(OptionLeg(
            action=leg_def["action"],
            option_type=leg_def["option_type"],
            strike=float(strike),
            expiration=expiration,
        ))

    user_notes = questionary.text(
        "Any constraints or context? (optional — press Enter to skip)\n"
        "  e.g. 'Max $200/strategy. Don't widen the spread.'",
        default="",
    ).ask()

    return TargetOption(
        ticker=ticker,
        strategy=strategy_key,
        legs=legs,
        user_notes=user_notes.strip() if user_notes and user_notes.strip() else None,
    )
