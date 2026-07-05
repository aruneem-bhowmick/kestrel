"""Integration tests: LiteLLMClient's OpenRouter path against a mock server.

Unlike the unit suite, these tests drive genuine litellm objects -- a real
``litellm.acompletion`` call against a real (if local) HTTP server -- so
they are the tests-of-record for the OpenRouter routing and error-mapping
behavior actually working end to end, not just the pure normalization
logic in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.provider.errors import AuthError, RateLimitError, ServerError
from kestrel.provider.events import UsageEvent, validate_stream_order
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p005, pytest.mark.integration, pytest.mark.api]

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


async def test_hello_cassette_streams_expected_text_and_usage(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server replays the "hello" cassette, when a turn is
    streamed, then the concatenated text and usage match the cassette
    exactly, and the whole event sequence satisfies the stream grammar."""
    base_url = mock_openai_server(_HELLO_CASSETTE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    events = await _collect(openrouter_client)

    assert validate_stream_order(events)
    text = "".join(event.text for event in events if hasattr(event, "text"))
    assert text == "Hello from GLM-5.2"
    usage = next(event for event in events if isinstance(event, UsageEvent))
    assert usage == UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0)


async def test_401_response_raises_auth_error(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the backend rejects the request with 401, when a turn is
    streamed, then AuthError is raised."""
    base_url = mock_openai_server(status_code=401)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    with pytest.raises(AuthError):
        await _collect(openrouter_client)


async def test_429_response_raises_rate_limit_error_with_retry_after(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the backend throttles the request with 429 and a Retry-After
    header, when a turn is streamed, then RateLimitError is raised
    carrying that header's value."""
    base_url = mock_openai_server(status_code=429, extra_headers={"retry-after": "7"})
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    with pytest.raises(RateLimitError) as exc_info:
        await _collect(openrouter_client)

    assert exc_info.value.retry_after_s == 7.0


async def test_500_response_raises_server_error(
    openrouter_client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the backend fails with a 500, when a turn is streamed, then
    ServerError is raised naming the status code."""
    base_url = mock_openai_server(status_code=500)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    with pytest.raises(ServerError) as exc_info:
        await _collect(openrouter_client)

    assert exc_info.value.status == 500
