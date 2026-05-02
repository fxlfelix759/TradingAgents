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
