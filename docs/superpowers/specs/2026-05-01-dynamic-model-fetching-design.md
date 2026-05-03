# Dynamic Model Fetching Design

**Date:** 2026-05-01  
**Branch:** feat/option-trade-evaluator  
**Status:** Approved

## Summary

Replace the hardcoded model catalog with live model fetching from each provider's API. The CLI filters the provider list to only those with API keys configured in `.env`, then fetches available models from the selected provider and presents a flat list for the user to pick from. A single model is used for both quick and deep thinking nodes.

## Motivation

- `model_catalog.py` is already outdated and requires manual maintenance
- Ollama's locally-pulled models are entirely dynamic — there is no canonical list
- OpenRouter has hundreds of models; a hardcoded subset is always incomplete
- Users should see exactly what they have access to, not what was hardcoded at dev time

## Architecture

### New file: `tradingagents/llm_clients/model_fetcher.py`

Two public functions and one exception:

```python
class ModelFetchError(Exception): ...

def available_providers() -> list[tuple[str, str, str | None]]:
    """Returns (display_name, provider_key, base_url) for providers with API keys set.
    Ollama is always included (no key required).
    Azure appears if AZURE_OPENAI_API_KEY is set; model selection stays as a manual text prompt.
    """

def fetch_models(provider: str, base_url: str | None = None) -> list[str]:
    """Fetch and return sorted model name strings from the provider's API.
    Raises ModelFetchError on network, auth, or parse failure.
    """
```

### Provider → env var → endpoint mapping

| Provider   | Env Var                | Endpoint                                                          | Auth              |
|------------|------------------------|-------------------------------------------------------------------|-------------------|
| openai     | `OPENAI_API_KEY`       | `GET https://api.openai.com/v1/models`                           | Bearer            |
| anthropic  | `ANTHROPIC_API_KEY`    | `GET https://api.anthropic.com/v1/models`                        | `x-api-key` header + `anthropic-version: 2023-06-01` |
| google     | `GOOGLE_API_KEY`       | `GET https://generativelanguage.googleapis.com/v1beta/models?key=…` | Query param    |
| xai        | `XAI_API_KEY`          | `GET https://api.x.ai/v1/models`                                 | Bearer            |
| deepseek   | `DEEPSEEK_API_KEY`     | `GET https://api.deepseek.com/models`                            | Bearer            |
| qwen       | `DASHSCOPE_API_KEY`    | `GET https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models` | Bearer        |
| glm        | `ZHIPU_API_KEY`        | `GET https://api.z.ai/api/paas/v4/models`                        | Bearer            |
| openrouter | `OPENROUTER_API_KEY`   | `GET https://openrouter.ai/api/v1/models`                        | Bearer            |
| ollama     | *(none)*               | `GET http://localhost:11434/api/tags`                             | None              |
| azure      | `AZURE_OPENAI_API_KEY` | *(no list API — model selection stays as manual text prompt)*    | —                 |

OpenAI, xAI, DeepSeek, Qwen, GLM, and OpenRouter share one OpenAI-compatible fetch function (`GET {base_url}/models` with Bearer token). Anthropic, Google, and Ollama each have a small custom fetcher.

### Model filtering

- **OpenAI**: Exclude models whose ID contains `embedding`, `whisper`, `tts`, `dall-e`, `moderation`, `babbage`, `davinci`, `curie`, `ada`. Keep only chat-capable models.
- **Google**: Include only models where `supportedGenerationMethods` contains `"generateContent"`.
- **Ollama**: Return all models from `/api/tags` — they are all locally pulled and chat-capable.
- **All others**: Return all models as-is.

## Deletions

The following are fully removed — no fallback, no deprecation path:

| Item | Reason |
|------|--------|
| `tradingagents/llm_clients/model_catalog.py` | Replaced by live fetching |
| `tradingagents/llm_clients/validators.py` | No longer needed |
| `warn_if_unknown_model()` in `base_client.py` | No catalog to validate against |
| `validate_model()` calls in all four clients | Same |
| `tests/test_model_validation.py` | Tests a deleted flow |

If a user passes an invalid model name, the provider's inference API will return an error at call time — a better and more current signal than a stale local warning.

## CLI Changes

### `cli/utils.py`

- `select_llm_provider()` calls `available_providers()` to build the choices list dynamically. Only providers with keys in env are shown; Ollama always appears.
- New `select_model(provider, base_url) -> str` replaces both `select_shallow_thinking_agent()` and `select_deep_thinking_agent()`:
  1. Show Rich spinner: *"Fetching models from {provider}…"*
  2. Call `fetch_models(provider, base_url)`
  3. On success: flat `questionary.select` with all model names
  4. On `ModelFetchError`: print `[red]✗ Failed to fetch models: {error}[/red]`, `exit(1)`
- Remove `select_shallow_thinking_agent()`, `select_deep_thinking_agent()`, `_select_model()`, and the `get_model_options` import.

### `cli/main.py`

Step 8 collapses from two model picks to one:

```
Step 7: LLM Provider  →  select_llm_provider()      (filtered by env keys)
Step 8: Model         →  select_model(provider, url) (single flat pick)
Step 9: Thinking mode →  unchanged (reasoning_effort, thinking_level, effort)
```

Both config keys receive the same chosen model:

```python
config["quick_think_llm"] = selections["model"]
config["deep_think_llm"]  = selections["model"]
```

The return dict from `collect_user_selections()` replaces `shallow_thinker` + `deep_thinker` keys with a single `model` key.

## Error Handling

`ModelFetchError` covers three cases:
1. **Network error** — connection refused (Ollama not running), timeout, DNS failure
2. **HTTP error** — 401/403 (bad key), 429 (rate limit), 5xx (provider down); error message includes status code and response body
3. **Parse error** — unexpected response schema

No retry logic. User fixes their setup (start Ollama, correct the key) and re-runs.

## Testing

New file: `tests/test_model_fetcher.py` with `requests` mocked via `unittest.mock.patch`:

- `available_providers()` returns correct subset for given env var combinations
- `available_providers()` always includes Ollama regardless of env
- `fetch_models("ollama")` parses `/api/tags` response format correctly
- `fetch_models("openai")` filters out embedding/tts/whisper/dall-e models
- `fetch_models("google")` filters to `generateContent`-capable models only
- `fetch_models("anthropic")` parses response correctly
- OpenAI-compatible path used for xai, deepseek, qwen, glm, openrouter
- `ModelFetchError` raised on connection error
- `ModelFetchError` raised on HTTP 401
- `ModelFetchError` raised on HTTP 403
