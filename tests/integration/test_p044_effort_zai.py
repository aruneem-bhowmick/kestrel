"""Integration tests: effort-to-reasoning mapping on LiteLLMClient's zai
path, against a mock server that captures the raw outgoing request body.

As with the OpenRouter suite, the registry entry built here points its
``endpoint`` field directly at the mock server's base URL -- the zai
branch has no environment-variable seam, so pointing a test at a fake
backend is done the same way pointing a real deployment at Z.ai's own
endpoint would be: by setting the registry entry's own field.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p044, pytest.mark.integration, pytest.mark.api]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_HELLO_CASSETTE = _CASSETTES / "zai_glm52_hello.sse"


def _zai_registry(*, endpoint: str) -> Registry:
    """Build a single-entry Registry for a zai route pointed at ``endpoint``."""
    entry = ModelEntry(
        id="glm-5.2-zai",
        backend="zai",
        provider_model="glm-5.2",
        endpoint=endpoint,
        api_key_env="ZAI_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={"glm-5.2-zai": entry}, source=None)


async def _collect(client: LiteLLMClient, **complete_kwargs: Any) -> list[Any]:
    """Run one complete() call to exhaustion and return its full event list."""
    defaults: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": None,
        "model_id": "glm-5.2-zai",
        "effort": "high",
        "stream": True,
    }
    defaults.update(complete_kwargs)
    return [event async for event in client.complete(**defaults)]


async def test_high_and_max_effort_produce_different_thinking_bodies(
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given two calls against the same zai entry, one at effort="high" and
    one at effort="max", when both are streamed, then the two captured raw
    request bodies differ exactly at ``thinking.effort``, matching
    ``_effort_kwargs``'s own documented pass-through mapping."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-key")
    capture: list[bytes] = []
    base_url = mock_openai_server(_HELLO_CASSETTE, capture=capture)
    client = LiteLLMClient(_zai_registry(endpoint=base_url))

    await _collect(client, effort="high")
    await _collect(client, effort="max")

    assert len(capture) == 2
    high_body = json.loads(capture[0])
    max_body = json.loads(capture[1])
    assert high_body["thinking"] == {"type": "enabled", "effort": "high"}
    assert max_body["thinking"] == {"type": "enabled", "effort": "max"}
