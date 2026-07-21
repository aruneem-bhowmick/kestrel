"""Integration tests: OllamaEmbeddingClient against a mock Ollama server.

Unlike the unit suite, these tests drive a genuine `litellm.aembedding`
call against a real (if local) HTTP server, exercising the actual
"ollama/" route litellm builds -- proving the wire contract this adapter
depends on actually holds, not just the pure error-handling logic checked
in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal

import pytest

from kestrel.kb.embeddings import EmbeddingError, OllamaEmbeddingClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p056, pytest.mark.integration]


def _ollama_registry(*, endpoint: str) -> Registry:
    """Build a single-entry Registry for an ollama-backed embedding model
    pointed at ``endpoint`` -- a mock server's own base_url in every test here."""
    entry = ModelEntry(
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
    return Registry(models={"nomic-embed-text": entry}, source=None)


async def test_embed_returns_vectors_in_order_and_sends_expected_request_body(
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given the mock server replays a scripted two-vector reply, when a
    batch of two strings is embedded, then the returned vectors match, in
    order, and the request body's own "input" and "model" fields round-trip
    exactly what was asked for."""
    capture: list[bytes] = []
    base_url = mock_ollama_server(embeddings=[[0.1, 0.2], [0.3, 0.4]], capture=capture)
    client = OllamaEmbeddingClient(_ollama_registry(endpoint=base_url))

    vectors = await client.embed(["a", "b"], model_id="nomic-embed-text")

    assert vectors == ((0.1, 0.2), (0.3, 0.4))
    assert len(capture) == 1
    body = json.loads(capture[0])
    assert body["input"] == ["a", "b"]
    assert body["model"] == "nomic-embed-text"


async def test_500_response_surfaces_as_embedding_error(
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given the mock server fails every request with a 500, when a batch
    is embedded, then EmbeddingError is raised -- never a raw litellm
    exception escaping this adapter."""
    base_url = mock_ollama_server(status_code=500)
    client = OllamaEmbeddingClient(_ollama_registry(endpoint=base_url))

    with pytest.raises(EmbeddingError):
        await client.embed(["a"], model_id="nomic-embed-text")
