"""Integration tests: `unload_aux_model` against a mock Ollama server.

Mirrors `test_p056_ollama_embed.py`'s own shape: a genuine request
against `mock_ollama_server`, proving the actual wire request this
function builds -- not just its error-handling logic in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal

import pytest

from kestrel.managers.resource_guard import UnloadAuxError, unload_aux_model
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p064, pytest.mark.integration]


def _ollama_entry(*, endpoint: str) -> ModelEntry:
    """Build a single ollama-backed `ModelEntry` pointed at `endpoint` --
    a mock server's own base_url in every test here."""
    return ModelEntry(
        id="nomic-embed-text",
        backend="ollama",
        provider_model="nomic-embed-text",
        endpoint=endpoint,
        context_window=8192,
        max_output=1,
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=False,
        supports_cache=False,
        tags=frozenset({"local"}),
    )


async def test_unload_aux_model_sends_the_documented_unload_request_body(
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given a mock Ollama server that accepts the request, when
    `unload_aux_model` is called, then it POSTs a generate request naming
    the entry's own model and a zero `keep_alive` -- Ollama's own
    documented immediate-unload idiom."""
    capture: list[bytes] = []
    base_url = mock_ollama_server(embeddings=[], capture=capture)
    entry = _ollama_entry(endpoint=base_url)

    await unload_aux_model(entry=entry)

    assert len(capture) == 1
    body = json.loads(capture[0])
    assert body == {"model": "nomic-embed-text", "keep_alive": 0}


async def test_unload_aux_model_raises_on_a_non_2xx_response(
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given a mock Ollama server that fails every request with a 500,
    when `unload_aux_model` is called, then `UnloadAuxError` is raised
    naming the model and endpoint rather than letting a raw HTTP
    exception escape this module."""
    base_url = mock_ollama_server(status_code=500)
    entry = _ollama_entry(endpoint=base_url)

    with pytest.raises(UnloadAuxError) as exc_info:
        await unload_aux_model(entry=entry)

    assert "nomic-embed-text" in str(exc_info.value)
