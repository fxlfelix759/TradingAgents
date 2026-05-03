# Dynamic Model Fetching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded model catalog with live API model fetching, filter provider list by configured API keys, and collapse the dual quick/deep model selection into a single flat model picker.

**Architecture:** A new `model_fetcher.py` module in `tradingagents/llm_clients/` owns all provider discovery and model-list fetching logic. The CLI calls `available_providers()` to build the provider list and `fetch_models()` after provider selection. Both `quick_think_llm` and `deep_think_llm` config keys are set to the single chosen model.

**Tech Stack:** `requests` (already a dependency), `rich` console status spinner, `questionary` for selection UI.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `tradingagents/llm_clients/model_fetcher.py` | `ModelFetchError`, `available_providers()`, `fetch_models()`, per-provider fetch functions |
| Create | `tests/test_model_fetcher.py` | Unit tests for all model fetcher functions |
| Delete | `tradingagents/llm_clients/model_catalog.py` | Replaced by live fetching |
| Delete | `tradingagents/llm_clients/validators.py` | No longer needed |
| Delete | `tests/test_model_validation.py` | Tests a deleted flow |
| Modify | `tradingagents/llm_clients/base_client.py` | Remove `warn_if_unknown_model()`, `validate_model()` abstract method, `import warnings` |
| Modify | `tradingagents/llm_clients/openai_client.py` | Remove `validate_model` import and method, remove `warn_if_unknown_model()` call |
| Modify | `tradingagents/llm_clients/anthropic_client.py` | Same as openai_client |
| Modify | `tradingagents/llm_clients/google_client.py` | Same as openai_client |
| Modify | `tradingagents/llm_clients/azure_client.py` | Same as openai_client |
| Modify | `cli/utils.py` | Use `available_providers()` in `select_llm_provider()`, add `select_model()`, remove old model-selection functions |
| Modify | `cli/main.py` | Collapse Steps 7+8, update return dict (`model` key), update `run_analysis()` config mapping |

---

## Task 1: `model_fetcher.py` — `ModelFetchError` and `available_providers()`

**Files:**
- Create: `tradingagents/llm_clients/model_fetcher.py`
- Create: `tests/test_model_fetcher.py`

- [ ] **Step 1: Write the failing tests for `available_providers()`**

Create `tests/test_model_fetcher.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.model_fetcher import (
    available_providers,
    fetch_models,
    ModelFetchError,
)


def test_available_providers_always_includes_ollama():
    with patch.dict(os.environ, {}, clear=True):
        providers = available_providers()
    assert "ollama" in [p[1] for p in providers]


def test_available_providers_filters_providers_without_keys():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        providers = available_providers()
    keys = [p[1] for p in providers]
    assert "openai" in keys
    assert "anthropic" not in keys
    assert "google" not in keys
    assert "ollama" in keys


def test_available_providers_includes_azure_when_key_set():
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "az-key"}, clear=True):
        providers = available_providers()
    assert "azure" in [p[1] for p in providers]


def test_available_providers_excludes_azure_without_key():
    with patch.dict(os.environ, {}, clear=True):
        providers = available_providers()
    assert "azure" not in [p[1] for p in providers]


def test_available_providers_returns_display_name_key_url_tuples():
    with patch.dict(os.environ, {}, clear=True):
        providers = available_providers()
    # Each entry must be a 3-tuple
    for entry in providers:
        assert len(entry) == 3
        display, key, url = entry
        assert isinstance(display, str)
        assert isinstance(key, str)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/test_model_fetcher.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `model_fetcher` doesn't exist yet.

- [ ] **Step 3: Create `tradingagents/llm_clients/model_fetcher.py` with `ModelFetchError` and `available_providers()`**

```python
import os
import requests
from typing import Optional


class ModelFetchError(Exception):
    pass


# (display_name, api_key_env_var, base_url)
_PROVIDER_META: dict[str, tuple[str, Optional[str], Optional[str]]] = {
    "openai":     ("OpenAI",       "OPENAI_API_KEY",      "https://api.openai.com/v1"),
    "anthropic":  ("Anthropic",    "ANTHROPIC_API_KEY",   "https://api.anthropic.com"),
    "google":     ("Google",       "GOOGLE_API_KEY",      "https://generativelanguage.googleapis.com"),
    "xai":        ("xAI",          "XAI_API_KEY",         "https://api.x.ai/v1"),
    "deepseek":   ("DeepSeek",     "DEEPSEEK_API_KEY",    "https://api.deepseek.com"),
    "qwen":       ("Qwen",         "DASHSCOPE_API_KEY",   "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "glm":        ("GLM",          "ZHIPU_API_KEY",       "https://api.z.ai/api/paas/v4"),
    "openrouter": ("OpenRouter",   "OPENROUTER_API_KEY",  "https://openrouter.ai/api/v1"),
    "azure":      ("Azure OpenAI", "AZURE_OPENAI_API_KEY", None),
    "ollama":     ("Ollama",       None,                   "http://localhost:11434"),
}

_OPENAI_COMPAT = {"openai", "xai", "deepseek", "qwen", "glm", "openrouter"}

_OPENAI_EXCLUDE = {
    "embedding", "whisper", "tts", "dall-e", "moderation",
    "babbage", "davinci", "curie", "ada",
}


def available_providers() -> list[tuple[str, str, Optional[str]]]:
    """Return (display_name, provider_key, base_url) for providers with API keys in env.

    Ollama is always included. All others require their env var to be set.
    """
    result = []
    for key, (display, env_var, base_url) in _PROVIDER_META.items():
        if key == "ollama":
            result.append((display, key, base_url))
        elif env_var and os.environ.get(env_var):
            result.append((display, key, base_url))
    return result


def fetch_models(provider: str, base_url: Optional[str] = None) -> list[str]:
    """Fetch available model names from the provider's API.

    Returns a sorted list of model name strings.
    Raises ModelFetchError on network, auth, or parse failure.
    """
    provider_lower = provider.lower()
    meta = _PROVIDER_META.get(provider_lower)
    if not meta:
        raise ModelFetchError(f"Unknown provider: {provider}")

    _, env_var, default_url = meta
    effective_url = base_url or default_url
    api_key = os.environ.get(env_var) if env_var else None

    try:
        if provider_lower == "ollama":
            return _fetch_ollama(effective_url or "http://localhost:11434")
        elif provider_lower == "anthropic":
            return _fetch_anthropic(api_key)
        elif provider_lower == "google":
            return _fetch_google(api_key)
        elif provider_lower in _OPENAI_COMPAT:
            return _fetch_openai_compat(provider_lower, effective_url or "", api_key)
        else:
            raise ModelFetchError(f"Model fetching not supported for provider: {provider}")
    except ModelFetchError:
        raise
    except Exception as exc:
        raise ModelFetchError(str(exc)) from exc


def _raise_for_status(resp: requests.Response, provider: str) -> None:
    if resp.status_code == 200:
        return
    body = resp.text[:300] if resp.text else ""
    raise ModelFetchError(f"HTTP {resp.status_code} from {provider}: {body}")


def _fetch_ollama(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    except requests.ConnectionError:
        raise ModelFetchError(
            f"Connection refused — is Ollama running at {base_url}?"
        )
    except requests.Timeout:
        raise ModelFetchError(f"Request to Ollama timed out at {base_url}")
    _raise_for_status(resp, "ollama")
    return sorted(m["name"] for m in resp.json().get("models", []))


def _fetch_openai_compat(provider: str, base_url: str, api_key: Optional[str]) -> list[str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
        )
    except requests.ConnectionError as exc:
        raise ModelFetchError(f"Connection error for {provider}: {exc}")
    except requests.Timeout:
        raise ModelFetchError(f"Request to {provider} timed out")
    _raise_for_status(resp, provider)
    models = [m["id"] for m in resp.json().get("data", [])]
    if provider == "openai":
        models = [m for m in models if not any(kw in m for kw in _OPENAI_EXCLUDE)]
    return sorted(models)


def _fetch_anthropic(api_key: Optional[str]) -> list[str]:
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(
            "https://api.anthropic.com/v1/models", headers=headers, timeout=10
        )
    except requests.ConnectionError as exc:
        raise ModelFetchError(f"Connection error for anthropic: {exc}")
    except requests.Timeout:
        raise ModelFetchError("Request to anthropic timed out")
    _raise_for_status(resp, "anthropic")
    return sorted(m["id"] for m in resp.json().get("data", []))


def _fetch_google(api_key: Optional[str]) -> list[str]:
    if not api_key:
        raise ModelFetchError("GOOGLE_API_KEY not set")
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=10,
        )
    except requests.ConnectionError as exc:
        raise ModelFetchError(f"Connection error for google: {exc}")
    except requests.Timeout:
        raise ModelFetchError("Request to google timed out")
    _raise_for_status(resp, "google")
    return sorted(
        m["name"].split("/")[-1]
        for m in resp.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )
```

- [ ] **Step 4: Run the `available_providers` tests**

```bash
pytest tests/test_model_fetcher.py::test_available_providers_always_includes_ollama tests/test_model_fetcher.py::test_available_providers_filters_providers_without_keys tests/test_model_fetcher.py::test_available_providers_includes_azure_when_key_set tests/test_model_fetcher.py::test_available_providers_excludes_azure_without_key tests/test_model_fetcher.py::test_available_providers_returns_display_name_key_url_tuples -v
```

Expected: All 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/model_fetcher.py tests/test_model_fetcher.py
git commit -m "feat: add model_fetcher module with available_providers()"
```

---

## Task 2: Ollama model fetching

**Files:**
- Modify: `tests/test_model_fetcher.py`
- `tradingagents/llm_clients/model_fetcher.py` already contains `_fetch_ollama` — tests just need adding

- [ ] **Step 1: Add Ollama fetch tests to `tests/test_model_fetcher.py`**

Append to the end of the file:

```python
# --- Ollama ---

def test_fetch_ollama_returns_sorted_model_names():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen3:latest"}, {"name": "llama3.2:latest"}]
    }
    with patch("tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp):
        result = fetch_models("ollama")
    assert result == ["llama3.2:latest", "qwen3:latest"]


def test_fetch_ollama_empty_models_list():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": []}
    with patch("tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp):
        result = fetch_models("ollama")
    assert result == []


def test_fetch_ollama_connection_refused_raises_fetch_error():
    import requests as req
    with patch(
        "tradingagents.llm_clients.model_fetcher.requests.get",
        side_effect=req.ConnectionError("refused"),
    ):
        with pytest.raises(ModelFetchError, match="Connection refused"):
            fetch_models("ollama")


def test_fetch_ollama_timeout_raises_fetch_error():
    import requests as req
    with patch(
        "tradingagents.llm_clients.model_fetcher.requests.get",
        side_effect=req.Timeout(),
    ):
        with pytest.raises(ModelFetchError, match="timed out"):
            fetch_models("ollama")
```

- [ ] **Step 2: Run Ollama fetch tests**

```bash
pytest tests/test_model_fetcher.py -k "ollama" -v
```

Expected: All 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_fetcher.py
git commit -m "test: add ollama model fetch tests"
```

---

## Task 3: OpenAI-compatible model fetching

**Files:**
- Modify: `tests/test_model_fetcher.py`

- [ ] **Step 1: Add OpenAI-compat fetch tests**

Append to `tests/test_model_fetcher.py`:

```python
# --- OpenAI-compatible providers ---

def test_fetch_openai_filters_non_chat_models():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "gpt-4o"},
            {"id": "gpt-4o-mini"},
            {"id": "text-embedding-ada-002"},
            {"id": "whisper-1"},
            {"id": "tts-1"},
            {"id": "dall-e-3"},
            {"id": "omni-moderation-latest"},
        ]
    }
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("openai")
    assert "gpt-4o" in result
    assert "gpt-4o-mini" in result
    assert "text-embedding-ada-002" not in result
    assert "whisper-1" not in result
    assert "tts-1" not in result
    assert "dall-e-3" not in result
    assert "omni-moderation-latest" not in result


def test_fetch_xai_returns_all_models_unsorted():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"id": "grok-4"}, {"id": "grok-4-mini"}]
    }
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("xai")
    assert result == ["grok-4", "grok-4-mini"]


def test_fetch_openrouter_returns_all_models():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-3-5-sonnet"}]
    }
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("openrouter")
    assert "anthropic/claude-3-5-sonnet" in result
    assert "openai/gpt-4o" in result
```

- [ ] **Step 2: Run OpenAI-compat tests**

```bash
pytest tests/test_model_fetcher.py -k "openai or xai or openrouter" -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_fetcher.py
git commit -m "test: add OpenAI-compatible provider model fetch tests"
```

---

## Task 4: Anthropic model fetching

**Files:**
- Modify: `tests/test_model_fetcher.py`

- [ ] **Step 1: Add Anthropic fetch tests**

Append to `tests/test_model_fetcher.py`:

```python
# --- Anthropic ---

def test_fetch_anthropic_returns_sorted_model_names():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"id": "claude-opus-4-7"}, {"id": "claude-sonnet-4-6"}, {"id": "claude-haiku-4-5"}]
    }
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("anthropic")
    assert result == ["claude-haiku-4-5", "claude-opus-4-7", "claude-sonnet-4-6"]


def test_fetch_anthropic_sends_correct_headers():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": []}
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ) as mock_get:
            fetch_models("anthropic")
    call_kwargs = mock_get.call_args
    headers = call_kwargs.kwargs.get("headers", {})
    assert headers.get("x-api-key") == "ant-key"
    assert headers.get("anthropic-version") == "2023-06-01"
```

- [ ] **Step 2: Run Anthropic tests**

```bash
pytest tests/test_model_fetcher.py -k "anthropic" -v
```

Expected: Both PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_fetcher.py
git commit -m "test: add Anthropic model fetch tests"
```

---

## Task 5: Google model fetching

**Files:**
- Modify: `tests/test_model_fetcher.py`

- [ ] **Step 1: Add Google fetch tests**

Append to `tests/test_model_fetcher.py`:

```python
# --- Google ---

def test_fetch_google_filters_to_generate_content_models():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/embedding-001",
                "supportedGenerationMethods": ["embedContent"],
            },
            {
                "name": "models/gemini-2.5-pro",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "goog-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("google")
    assert "gemini-2.5-flash" in result
    assert "gemini-2.5-pro" in result
    assert "embedding-001" not in result


def test_fetch_google_strips_models_prefix_from_name():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            }
        ]
    }
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "goog-test"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            result = fetch_models("google")
    assert result == ["gemini-2.5-flash"]


def test_fetch_google_raises_when_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ModelFetchError, match="GOOGLE_API_KEY not set"):
            fetch_models("google")
```

- [ ] **Step 2: Run Google tests**

```bash
pytest tests/test_model_fetcher.py -k "google" -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_fetcher.py
git commit -m "test: add Google model fetch tests"
```

---

## Task 6: HTTP error handling

**Files:**
- Modify: `tests/test_model_fetcher.py`

- [ ] **Step 1: Add HTTP error tests**

Append to `tests/test_model_fetcher.py`:

```python
# --- HTTP error handling ---

def test_fetch_models_raises_model_fetch_error_on_401():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    with patch.dict(os.environ, {"OPENAI_API_KEY": "bad-key"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            with pytest.raises(ModelFetchError, match="401"):
                fetch_models("openai")


def test_fetch_models_raises_model_fetch_error_on_403():
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "bad-key"}):
        with patch(
            "tradingagents.llm_clients.model_fetcher.requests.get", return_value=mock_resp
        ):
            with pytest.raises(ModelFetchError, match="403"):
                fetch_models("anthropic")


def test_fetch_models_raises_model_fetch_error_on_unknown_provider():
    with pytest.raises(ModelFetchError, match="Unknown provider"):
        fetch_models("nonexistent_provider")
```

- [ ] **Step 2: Run error handling tests**

```bash
pytest tests/test_model_fetcher.py -k "401 or 403 or unknown" -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Run the full test_model_fetcher suite**

```bash
pytest tests/test_model_fetcher.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_fetcher.py
git commit -m "test: add HTTP error handling tests for model fetcher"
```

---

## Task 7: Strip model catalog — delete files and clean all clients

**Files:**
- Delete: `tradingagents/llm_clients/model_catalog.py`
- Delete: `tradingagents/llm_clients/validators.py`
- Delete: `tests/test_model_validation.py`
- Modify: `tradingagents/llm_clients/base_client.py`
- Modify: `tradingagents/llm_clients/openai_client.py`
- Modify: `tradingagents/llm_clients/anthropic_client.py`
- Modify: `tradingagents/llm_clients/google_client.py`
- Modify: `tradingagents/llm_clients/azure_client.py`

- [ ] **Step 1: Delete the catalog and validation files**

```bash
rm tradingagents/llm_clients/model_catalog.py
rm tradingagents/llm_clients/validators.py
rm tests/test_model_validation.py
```

- [ ] **Step 2: Rewrite `tradingagents/llm_clients/base_client.py`**

Remove `import warnings`, `warn_if_unknown_model()`, `validate_model()` abstract method, and `get_provider_name()` (only used by `warn_if_unknown_model`). Final file:

```python
from abc import ABC, abstractmethod
from typing import Any, Optional


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass
```

- [ ] **Step 3: Clean `tradingagents/llm_clients/openai_client.py`**

Remove `from .validators import validate_model`, the `self.warn_if_unknown_model()` call in `get_llm()`, and the `validate_model()` method. The final class body of `OpenAIClient`:

```python
class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        llm_kwargs = {"model": self.model}

        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        return NormalizedChatOpenAI(**llm_kwargs)
```

- [ ] **Step 4: Clean `tradingagents/llm_clients/anthropic_client.py`**

Remove `from .validators import validate_model`, the `self.warn_if_unknown_model()` call, and `validate_model()` method. Final `AnthropicClient`:

```python
class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)
```

- [ ] **Step 5: Clean `tradingagents/llm_clients/google_client.py`**

Remove `from .validators import validate_model`, `self.warn_if_unknown_model()` call, and `validate_model()` method. Final `GoogleClient`:

```python
class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)
```

- [ ] **Step 6: Clean `tradingagents/llm_clients/azure_client.py`**

Remove `from .validators import validate_model`, `self.warn_if_unknown_model()` call, and `validate_model()` method. Final `AzureOpenAIClient`:

```python
class AzureOpenAIClient(BaseLLMClient):
    """Client for Azure OpenAI deployments.

    Requires environment variables:
        AZURE_OPENAI_API_KEY: API key
        AZURE_OPENAI_ENDPOINT: Endpoint URL (e.g. https://<resource>.openai.azure.com/)
        AZURE_OPENAI_DEPLOYMENT_NAME: Deployment name
        OPENAI_API_VERSION: API version (e.g. 2025-03-01-preview)
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured AzureChatOpenAI instance."""
        llm_kwargs = {
            "model": self.model,
            "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", self.model),
        }

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedAzureChatOpenAI(**llm_kwargs)
```

- [ ] **Step 7: Run all unit tests to verify nothing broke**

```bash
pytest -m unit -v
```

Expected: All PASS. The deleted `test_model_validation.py` is gone; no references remain.

- [ ] **Step 8: Run full test suite**

```bash
pytest -v
```

Expected: All PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove model catalog and validation, clean all LLM clients"
```

---

## Task 8: Update `cli/utils.py`

**Files:**
- Modify: `cli/utils.py`

- [ ] **Step 1: Replace `select_llm_provider()` to use `available_providers()`**

Find the existing `select_llm_provider()` function (starts at line 232) and replace it entirely:

```python
def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider. Only shows providers with API keys set in env."""
    from tradingagents.llm_clients.model_fetcher import available_providers

    providers = available_providers()

    if not providers:
        console.print(
            "[red]No LLM provider API keys found in environment. "
            "Set at least one API key in .env and restart.[/red]"
        )
        exit(1)

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(provider_key, url))
            for display, provider_key, url in providers
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
```

- [ ] **Step 2: Add `select_model()` function**

Add after `select_llm_provider()`:

```python
def select_model(provider: str, base_url: str | None = None) -> str:
    """Fetch available models from provider and prompt user to select one."""
    from tradingagents.llm_clients.model_fetcher import fetch_models, ModelFetchError

    if provider == "azure":
        return questionary.text(
            "Enter Azure deployment name:",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a deployment name.",
        ).ask().strip()

    with console.status(f"[bold green]Fetching models from {provider}…[/bold green]"):
        try:
            models = fetch_models(provider, base_url)
        except ModelFetchError as exc:
            console.print(f"\n[red]✗ Failed to fetch models from {provider}: {exc}[/red]")
            exit(1)

    if not models:
        console.print(f"[red]No models returned from {provider}.[/red]")
        exit(1)

    choice = questionary.select(
        f"Select model ({provider}):",
        choices=[questionary.Choice(m, value=m) for m in models],
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
        console.print("\n[red]No model selected. Exiting...[/red]")
        exit(1)

    return choice
```

- [ ] **Step 3: Remove dead functions and import**

Remove the following from `cli/utils.py`:
- Line 7: `from tradingagents.llm_clients.model_catalog import get_model_options`
- Lines 186–220: the `_select_model(provider, mode)` function (the private helper)
- Lines 223–225: `select_shallow_thinking_agent(provider)`
- Lines 228–230: `select_deep_thinking_agent(provider)`

After removal these three functions are gone entirely; `select_model` (added in Step 2) is the only model-selection function remaining.

- [ ] **Step 4: Verify no remaining references to deleted symbols**

```bash
grep -n "get_model_options\|_select_model\|select_shallow_thinking_agent\|select_deep_thinking_agent" cli/utils.py
```

Expected: No output.

- [ ] **Step 5: Commit**

```bash
git add cli/utils.py
git commit -m "feat: update CLI provider selection to use dynamic model fetching"
```

---

## Task 9: Update `cli/main.py`

**Files:**
- Modify: `cli/main.py`

- [ ] **Step 1: Collapse Steps 7+8 in `get_user_selections()`**

Find lines 582–597 (Steps 7 and 8). Replace:

```python
    # Step 7: LLM Provider
    console.print(
        create_question_box(
            "Step 7: LLM Provider", "Select your LLM provider"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Step 8: Thinking agents
    console.print(
        create_question_box(
            "Step 8: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)
```

With:

```python
    # Step 7: LLM Provider
    console.print(
        create_question_box(
            "Step 7: LLM Provider", "Select your LLM provider"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Step 8: Model selection
    console.print(
        create_question_box(
            "Step 8: Model", "Select the model for analysis"
        )
    )
    selected_model = select_model(selected_llm_provider, backend_url)
```

- [ ] **Step 2: Verify no import change needed in `cli/main.py`**

`cli/main.py` uses `from cli.utils import *` (line 30), so `select_model` is automatically available once added to `cli/utils.py`. No import change required. Confirm:

```bash
grep -n "from cli.utils" cli/main.py
```

Expected output: `30:from cli.utils import *`

- [ ] **Step 3: Update the return dict in `get_user_selections()`**

Find lines 630–644 (the `return { ... }` block). Replace `shallow_thinker` and `deep_thinker` keys:

```python
        "model": selected_model,
```

Remove:
```python
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
```

- [ ] **Step 4: Update `run_analysis()` to use `selections["model"]`**

Find lines 978–979:

```python
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
```

Replace with:

```python
    config["quick_think_llm"] = selections["model"]
    config["deep_think_llm"] = selections["model"]
```

- [ ] **Step 5: Verify no remaining references to removed keys**

```bash
grep -n "shallow_thinker\|deep_thinker\|select_shallow\|select_deep" cli/main.py
```

Expected: No output.

- [ ] **Step 6: Run the full test suite**

```bash
pytest -v
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add cli/main.py
git commit -m "feat: collapse quick/deep model selection into single dynamic model picker"
```

---

## Done

The feature is complete when:
- `pytest -v` passes with no failures
- Running `tradingagents` shows only providers with API keys (+ Ollama always)
- Selecting Ollama fetches locally pulled models live from `http://localhost:11434/api/tags`
- Selecting any cloud provider fetches its live model list
- A single model is selected and used for both `quick_think_llm` and `deep_think_llm`
