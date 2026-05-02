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
