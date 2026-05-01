# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (prefer uv)
pip install .
# or: uv sync

# Run the interactive CLI
tradingagents
python -m cli.main          # from source

# Run a single analysis programmatically
python main.py

# Tests
pytest                      # all tests
pytest -m unit              # fast isolated unit tests only
pytest -m smoke             # quick sanity-check tests
pytest tests/test_memory_log.py  # single file
```

Environment: copy `.env.example` to `.env` and fill in API keys before running. At minimum one LLM provider key is required; yfinance (default data vendor) needs no additional key.

## Architecture

TradingAgents is a multi-agent LangGraph pipeline that produces a trade decision (BUY/SELL/HOLD) for a given ticker and date. The main entry point is `TradingAgentsGraph` in `tradingagents/graph/trading_graph.py`.

### Graph execution order (linear unless debate loops)

```
Analyst(s) → [tool calls] → Msg Clear
  → Bull Researcher ↔ Bear Researcher (debate loop)
  → Research Manager → Change Analyst → Trader
  → Aggressive ↔ Conservative ↔ Neutral (risk debate loop)
  → Portfolio Manager → END
```

Graph wiring lives in `tradingagents/graph/setup.py`. Conditional edges (debate continuation, analyst tool use) are in `tradingagents/graph/conditional_logic.py`.

### Key packages

| Path | Purpose |
|------|---------|
| `tradingagents/graph/` | LangGraph orchestration: setup, propagation, checkpointing, reflection, signal processing |
| `tradingagents/agents/` | Agent factory functions (analysts, researchers, trader, risk, managers), state types (`agent_states.py`), shared tools (`utils/`) |
| `tradingagents/llm_clients/` | Provider abstraction — `create_llm_client(provider, model)` factory returns a `BaseLLMClient`; supports openai/xai/deepseek/qwen/glm/ollama/openrouter (OpenAI-compatible), anthropic, google, azure |
| `tradingagents/dataflows/` | Data fetching via yfinance or Alpha Vantage; vendor routing controlled by `config["data_vendors"]` |
| `tradingagents/default_config.py` | Single source of truth for all config keys and defaults |
| `cli/` | Typer-based CLI; `cli/main.py` is the `tradingagents` entry point |

### State

`AgentState` (in `agents/utils/agent_states.py`) is the LangGraph state dict passed through every node. It holds analyst reports, debate histories (`InvestDebateState`, `RiskDebateState`), trader plan, and `final_trade_decision`.

### Data vendors

The `data_vendors` config dict routes each category (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) to either `"yfinance"` or `"alpha_vantage"`. Individual tools can be overridden via `tool_vendors`.

### Persistence

- **Decision log**: every completed run appends to `~/.tradingagents/memory/trading_memory.md`; on the next same-ticker run, past decisions + reflections are injected into the Portfolio Manager prompt.
- **Checkpoint resume**: opt-in via `config["checkpoint_enabled"] = True` or `--checkpoint` CLI flag; SQLite per ticker under `~/.tradingagents/cache/checkpoints/`.

### Tests

Tests live in `tests/`. The `conftest.py` autouse fixture stubs all API key env vars with `"placeholder"` so unit tests never need real keys. Use `mock_llm_client` fixture to avoid hitting actual LLM providers.
