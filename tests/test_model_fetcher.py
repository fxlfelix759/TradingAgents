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
    for entry in providers:
        assert len(entry) == 3
        display, key, url = entry
        assert isinstance(display, str)
        assert isinstance(key, str)


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
