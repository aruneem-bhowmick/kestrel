"""Unit tests for the zai backend's parameter translation and credentials.

These cases are deterministic and network-free, in the same spirit as the
adjacent normalization suite: they exercise ``_litellm_params`` and
``_require_api_key`` directly against hand-built registry entries, rather
than driving a real network call -- the mock-server-backed integration
suite is what proves this routing actually works against genuine litellm
objects.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from kestrel.provider.errors import AuthError
from kestrel.provider.litellm_client import _litellm_params, _require_api_key
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p006, pytest.mark.unit]


def _entry(**overrides: Any) -> ModelEntry:
    """Build a valid zai ModelEntry, overriding only the fields a test cares about."""
    fields: dict[str, Any] = {
        "id": "glm-5.2-zai",
        "backend": "zai",
        "provider_model": "glm-5.2",
        "endpoint": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
        "context_window": 200_000,
        "max_output": 16_384,
        "usd_per_mtok_input": Decimal("0.60"),
        "usd_per_mtok_output": Decimal("2.20"),
        "usd_per_mtok_cached": Decimal("0.11"),
        "supports_tools": True,
        "supports_cache": True,
    }
    fields.update(overrides)
    return ModelEntry(**fields)


@pytest.mark.sanity
def test_litellm_params_for_zai_entry_uses_registry_endpoint_verbatim() -> None:
    """Given a zai registry entry, when translated, then the model is
    prefixed for litellm's OpenAI-compatible path and api_base is taken
    verbatim from the entry's own endpoint -- no environment-variable
    seam, unlike the OpenRouter branch."""
    entry = _entry(
        provider_model="glm-5.2", endpoint="https://api.z.ai/api/paas/v4"
    )

    params = _litellm_params(entry)

    assert params == {
        "model": "openai/glm-5.2",
        "api_base": "https://api.z.ai/api/paas/v4",
    }


def test_litellm_params_for_zai_entry_follows_endpoint_override() -> None:
    """Given a zai registry entry pointed at a non-default endpoint (e.g.
    a mock server in a test), when translated, then api_base follows the
    entry's endpoint exactly -- the registry, not an env var, drives
    routing for this backend."""
    entry = _entry(endpoint="http://127.0.0.1:9999/v1")

    params = _litellm_params(entry)

    assert params["api_base"] == "http://127.0.0.1:9999/v1"


def test_require_api_key_raises_auth_error_naming_zai_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ZAI_API_KEY is not set, when the credential is required for a
    zai entry, then AuthError names that environment variable."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    entry = _entry(api_key_env="ZAI_API_KEY")

    with pytest.raises(AuthError, match="ZAI_API_KEY"):
        _require_api_key(entry)


def test_require_api_key_returns_value_for_zai_entry_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ZAI_API_KEY is set, when the credential is required for a zai
    entry, then its value is returned -- credential lookup is backend-
    agnostic and needs no zai-specific branch."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-test-value")
    entry = _entry(api_key_env="ZAI_API_KEY")

    assert _require_api_key(entry) == "sk-zai-test-value"


@pytest.mark.sanity
def test_litellm_params_for_default_openrouter_entry_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the packaged default's OpenRouter entry for glm-5.2, when
    translated, then the resulting params are byte-for-byte what they were
    before the zai branch existed -- pinned to guard against the zai
    addition accidentally regressing the existing OpenRouter path."""
    monkeypatch.delenv("KESTREL_OPENROUTER_BASE_URL", raising=False)
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )

    params = _litellm_params(entry)

    assert params == {
        "model": "openrouter/z-ai/glm-5.2",
        "api_base": "https://openrouter.ai/api/v1",
    }
