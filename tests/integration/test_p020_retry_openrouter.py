"""Integration test: retrying a real LiteLLMClient call through a mock server.

Unlike the unit suite's scripted fake client, this drives genuine litellm
objects end to end -- a 429 response followed by a successful cassette,
scripted via the mock server's ``cassette_sequence`` mode -- proving the
retry wrapper actually recovers from a real (if local) rate-limit
response, not just from a hand-built exception.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.provider.events import UsageEvent, validate_stream_order
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.provider.retry import RetryPolicy, complete_with_retry
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p020, pytest.mark.integration]

_HELLO_CASSETTE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "cassettes"
    / "openrouter_glm52_hello.sse"
)


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


async def test_429_then_success_recovers_on_second_attempt(
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server fails the first request with 429 and succeeds
    with the "hello" cassette on the second, when the call is wrapped in
    complete_with_retry, then it recovers on the second attempt: no error
    reaches the caller, the streamed text and usage match the cassette,
    and the sequence still satisfies the stream grammar."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(cassette_sequence=[429, _HELLO_CASSETTE])
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)
    client = LiteLLMClient(_openrouter_registry())

    delays: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        delays.append(seconds)

    events = [
        event
        async for event in complete_with_retry(
            client,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model_id="glm-5.2",
            effort="high",
            policy=RetryPolicy(max_attempts=2),
            sleep_fn=_no_sleep,
            jitter_fn=lambda: 0.1,
        )
    ]

    assert len(delays) == 1
    assert validate_stream_order(events)
    text = "".join(event.text for event in events if hasattr(event, "text"))
    assert text == "Hello from GLM-5.2"
    usage = next(event for event in events if isinstance(event, UsageEvent))
    assert usage == UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0)
