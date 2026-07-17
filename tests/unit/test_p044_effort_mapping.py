"""Unit tests for ``_effort_kwargs``'s per-backend reasoning-depth mapping.

These cases are deterministic and network-free: they exercise
``_effort_kwargs`` directly against hand-built registry entries, one per
backend, rather than driving a real network call -- the mock-server-backed
integration suite is what proves the mapped keys actually reach the
outgoing request against genuine litellm objects.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from kestrel.provider.base import Effort
from kestrel.provider.litellm_client import _effort_kwargs
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p044, pytest.mark.unit]


def _entry(**overrides: Any) -> ModelEntry:
    """Build a valid openrouter-shaped ModelEntry, overriding only the
    fields a test cares about."""
    fields: dict[str, Any] = {
        "id": "glm-5.2",
        "backend": "openrouter",
        "provider_model": "z-ai/glm-5.2",
        "api_key_env": "OPENROUTER_API_KEY",
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
def test_openrouter_high_effort_maps_to_medium_reasoning() -> None:
    """Given an openrouter entry and effort="high", when mapped, then the
    result carries OpenRouter's own "medium" reasoning-effort rung --
    Kestrel's two-level scale reserves "high" (OpenRouter's top rung) for
    Kestrel's own "max"."""
    entry = _entry(backend="openrouter")

    assert _effort_kwargs(entry, "high") == {
        "extra_body": {"reasoning": {"effort": "medium"}}
    }


@pytest.mark.sanity
def test_openrouter_max_effort_maps_to_high_reasoning() -> None:
    """Given an openrouter entry and effort="max", when mapped, then the
    result carries OpenRouter's own top "high" reasoning-effort rung."""
    entry = _entry(backend="openrouter")

    assert _effort_kwargs(entry, "max") == {
        "extra_body": {"reasoning": {"effort": "high"}}
    }


@pytest.mark.sanity
def test_zai_high_effort_passes_through_unchanged() -> None:
    """Given a zai entry and effort="high", when mapped, then the result
    carries GLM's own "thinking" object with "high" passed through
    verbatim -- Z.ai's vocabulary already matches Kestrel's own."""
    entry = _entry(
        backend="zai",
        provider_model="glm-5.2",
        endpoint="https://api.z.ai/api/paas/v4",
        api_key_env="ZAI_API_KEY",
    )

    assert _effort_kwargs(entry, "high") == {
        "extra_body": {"thinking": {"type": "enabled", "effort": "high"}}
    }


@pytest.mark.sanity
def test_zai_max_effort_passes_through_unchanged() -> None:
    """Given a zai entry and effort="max", when mapped, then the result
    carries GLM's own "thinking" object with "max" passed through
    verbatim, matching the same shape as the "high" case."""
    entry = _entry(
        backend="zai",
        provider_model="glm-5.2",
        endpoint="https://api.z.ai/api/paas/v4",
        api_key_env="ZAI_API_KEY",
    )

    assert _effort_kwargs(entry, "max") == {
        "extra_body": {"thinking": {"type": "enabled", "effort": "max"}}
    }


@pytest.mark.sanity
@pytest.mark.parametrize("effort", ["high", "max"])
@pytest.mark.parametrize("backend", ["anthropic", "ollama"])
def test_unimplemented_backends_map_to_an_empty_dict(
    backend: str, effort: Effort
) -> None:
    """Given an anthropic or ollama entry, when mapped at either effort
    level, then the result is an empty dict -- both backends are
    unreachable today since ``_litellm_params`` already raises
    ``ServerError`` for them before this mapping would ever run; the
    case exists only to keep the mapping's own match exhaustive."""
    entry = _entry(
        backend=backend,
        provider_model="placeholder-model",
        endpoint="https://placeholder.invalid" if backend == "ollama" else None,
        api_key_env="PLACEHOLDER_API_KEY",
    )

    assert _effort_kwargs(entry, effort) == {}
