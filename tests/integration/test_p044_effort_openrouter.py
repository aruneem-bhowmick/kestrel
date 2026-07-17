"""Integration tests: effort-to-reasoning mapping on LiteLLMClient's
OpenRouter path, against a mock server that captures the raw outgoing
request body.

Unlike the unit suite, these tests drive genuine litellm objects -- a
real ``litellm.acompletion`` call against a real (if local) HTTP server
-- so they are the tests-of-record for the mapped ``effort`` value
actually reaching the wire, not just the pure mapping logic in
isolation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.cost import compute_turn_cost
from kestrel.provider.events import UsageEvent
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p044, pytest.mark.integration, pytest.mark.api]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_HELLO_CASSETTE = _CASSETTES / "openrouter_glm52_hello.sse"


def _openrouter_registry() -> Registry:
    """Build a single-entry Registry for the packaged default's OpenRouter route."""
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
    return Registry(models={"glm-5.2": entry}, source=None)


async def _collect(client: LiteLLMClient, **complete_kwargs: Any) -> list[Any]:
    """Run one complete() call to exhaustion and return its full event list."""
    defaults: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": None,
        "model_id": "glm-5.2",
        "effort": "high",
        "stream": True,
    }
    defaults.update(complete_kwargs)
    return [event async for event in client.complete(**defaults)]


@pytest.fixture
def openrouter_client(monkeypatch: pytest.MonkeyPatch) -> LiteLLMClient:
    """A LiteLLMClient bound to a single OpenRouter entry with a test API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    return LiteLLMClient(_openrouter_registry())


async def test_high_and_max_effort_produce_different_reasoning_bodies(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given two calls against the same entry, one at effort="high" and one
    at effort="max", when both are streamed, then the two captured raw
    request bodies differ exactly at ``reasoning.effort``, matching
    ``_effort_kwargs``'s own documented mapping."""
    capture: list[bytes] = []
    base_url = mock_openai_server(_HELLO_CASSETTE, capture=capture)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    await _collect(openrouter_client, effort="high")
    await _collect(openrouter_client, effort="max")

    assert len(capture) == 2
    high_body = json.loads(capture[0])
    max_body = json.loads(capture[1])
    assert high_body["reasoning"]["effort"] == "medium"
    assert max_body["reasoning"]["effort"] == "high"


@pytest.mark.cost_regression
async def test_turn_cost_is_identical_regardless_of_effort(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server's own scripted usage figures, when a turn is
    streamed at effort="high" and again at effort="max", then both price
    identically via compute_turn_cost -- the mapping changes the
    *request*, never the response this mock replays, so the priced
    result must not move with it."""
    base_url = mock_openai_server(_HELLO_CASSETTE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)
    entry = _openrouter_registry().get("glm-5.2")

    high_events = await _collect(openrouter_client, effort="high")
    max_events = await _collect(openrouter_client, effort="max")

    high_usage = next(e for e in high_events if isinstance(e, UsageEvent))
    max_usage = next(e for e in max_events if isinstance(e, UsageEvent))
    assert compute_turn_cost(high_usage, entry) == Decimal("0.000041")
    assert compute_turn_cost(max_usage, entry) == compute_turn_cost(high_usage, entry)
