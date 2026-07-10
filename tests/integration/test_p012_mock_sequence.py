"""Integration tests for the mock server's scripted cassette-sequence mode.

These tests exercise the multi-turn replay capability added to
``MockOpenAIServer``: serving an ordered list of cassettes across
successive requests instead of the same cassette every time, refusing an
ambiguous combination of construction arguments, and -- end to end,
through a real ``LiteLLMClient`` -- normalizing a tool-call cassette's
fragmented chunks into the provider event grammar in the right order.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from kestrel.provider.events import ToolCallEvent, UsageEvent, validate_stream_order
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p012, pytest.mark.integration]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_HELLO_CASSETTE = _CASSETTES / "openrouter_glm52_hello.sse"
_ZAI_HELLO_CASSETTE = _CASSETTES / "zai_glm52_hello.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_TOOLCALL_READ_FILE_CASSETTE = _CASSETTES / "toolcall_read_file.sse"


@pytest.mark.sanity
async def test_cassette_sequence_replays_in_order_then_clamps(
    mock_openai_server: Callable[..., str],
) -> None:
    """Given a server started with an ordered cassette sequence, when more
    requests arrive than the sequence has entries, then each request gets
    its matching cassette in order and every request past the end of the
    script replays the last entry."""
    base_url = mock_openai_server(
        cassette_sequence=[_HELLO_CASSETTE, _ZAI_HELLO_CASSETTE, _DONE_CASSETTE]
    )

    async with httpx.AsyncClient() as http:
        bodies = [
            (await http.post(f"{base_url}/chat/completions", json={})).text
            for _ in range(4)
        ]

    assert bodies[0] == _HELLO_CASSETTE.read_text(encoding="utf-8")
    assert bodies[1] == _ZAI_HELLO_CASSETTE.read_text(encoding="utf-8")
    assert bodies[2] == _DONE_CASSETTE.read_text(encoding="utf-8")
    assert bodies[3] == _DONE_CASSETTE.read_text(encoding="utf-8")


def test_cassette_path_and_sequence_are_mutually_exclusive(
    mock_openai_server: Callable[..., str],
) -> None:
    """Given both cassette_path and cassette_sequence are supplied, when the
    server is constructed, then it raises ValueError before ever starting
    (and never binds a socket)."""
    with pytest.raises(ValueError):
        mock_openai_server(
            cassette_path=_HELLO_CASSETTE, cassette_sequence=[_HELLO_CASSETTE]
        )


def test_empty_cassette_sequence_is_rejected(
    mock_openai_server: Callable[..., str],
) -> None:
    """Given an empty cassette_sequence, when the server is constructed,
    then it raises ValueError before ever starting, rather than
    constructing successfully and only failing on the first request."""
    with pytest.raises(ValueError):
        mock_openai_server(cassette_sequence=[])


def _toolcall_registry() -> Registry:
    """Build a single-entry Registry for an OpenRouter route with tools enabled."""
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
        "messages": [{"role": "user", "content": "read src/greet.py"}],
        "tools": None,
        "model_id": "glm-5.2",
        "effort": "high",
        "stream": True,
    }
    defaults.update(complete_kwargs)
    return [event async for event in client.complete(**defaults)]


async def test_toolcall_cassette_yields_tool_call_before_usage_and_stop(
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server replays a cassette whose chunks carry a
    fragmented tool_calls delta, when the turn is streamed through
    LiteLLMClient, then the assembled ToolCallEvent names the right tool
    and arguments and precedes the closing UsageEvent/StopEvent pair, and
    the whole sequence satisfies the stream grammar."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(_TOOLCALL_READ_FILE_CASSETTE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)
    client = LiteLLMClient(_toolcall_registry())

    events = await _collect(client)

    assert validate_stream_order(events)
    tool_call = next(event for event in events if isinstance(event, ToolCallEvent))
    assert tool_call.name == "read_file"
    assert tool_call.arguments_json == '{"path": "src/greet.py"}'
    usage_index = next(
        index for index, event in enumerate(events) if isinstance(event, UsageEvent)
    )
    assert events.index(tool_call) < usage_index
