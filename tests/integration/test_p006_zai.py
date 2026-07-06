"""Integration tests: LiteLLMClient's zai path against a mock server.

Unlike the OpenRouter suite, the registry entry built here points its
``endpoint`` field directly at the mock server's base URL -- the zai
branch has no environment-variable seam, so pointing a test at a fake
backend is done the same way pointing a real deployment at Z.ai's own
endpoint would be: by setting the registry entry's own field.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.provider.events import UsageEvent, validate_stream_order
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p006, pytest.mark.integration, pytest.mark.api]

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


async def test_hello_cassette_streams_expected_text_and_usage(
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server replays the "hello" cassette at the registry
    entry's own endpoint, when a turn is streamed, then the concatenated
    text and usage match the cassette exactly, and the whole event
    sequence satisfies the stream grammar."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-key")
    base_url = mock_openai_server(_HELLO_CASSETTE)
    client = LiteLLMClient(_zai_registry(endpoint=base_url))

    events = await _collect(client)

    assert validate_stream_order(events)
    text = "".join(event.text for event in events if hasattr(event, "text"))
    assert text == "Hello from Z.ai GLM"
    usage = next(event for event in events if isinstance(event, UsageEvent))
    assert usage == UsageEvent(input_tokens=40, output_tokens=6, cached_tokens=0)
