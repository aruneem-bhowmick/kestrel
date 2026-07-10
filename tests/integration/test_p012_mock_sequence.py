"""Integration tests for the mock server's scripted cassette-sequence mode.

These tests exercise the multi-turn replay capability added to
``MockOpenAIServer``: serving an ordered list of cassettes across
successive requests instead of the same cassette every time, and
refusing an ambiguous combination of construction arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.p012, pytest.mark.integration]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_HELLO_CASSETTE = _CASSETTES / "openrouter_glm52_hello.sse"
_ZAI_HELLO_CASSETTE = _CASSETTES / "zai_glm52_hello.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"


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
