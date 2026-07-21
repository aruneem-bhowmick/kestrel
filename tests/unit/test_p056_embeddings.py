"""Unit tests for OllamaEmbeddingClient's own error handling and row parsing.

These cases are deterministic and network-free: both of `embed`'s failure
paths documented in `kestrel.kb.embeddings` (empty input, a non-"ollama"
registry entry) are checked entirely before any network call is placed, so
a bare in-memory `Registry` with no reachable endpoint is enough to
exercise them -- the mock-server-backed integration suite is what proves a
real embedding call actually works end to end.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from kestrel.kb.embeddings import (
    EmbeddingError,
    OllamaEmbeddingClient,
    _extract_embedding,
)
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p056, pytest.mark.unit, pytest.mark.sanity]


def _openrouter_entry() -> ModelEntry:
    """A valid, non-ollama registry entry: the packaged default's own glm-5.2 route."""
    return ModelEntry(
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


def _registry(*entries: ModelEntry) -> Registry:
    """Build a bare in-memory Registry from hand-built entries -- no file
    read, no endpoint reachable, since these tests never place a real call."""
    return Registry(models={entry.id: entry for entry in entries}, source=None)


def test_extract_embedding_returns_the_rows_own_embedding_field_as_a_tuple() -> None:
    """Given a hand-built `EmbeddingResponse.data` row, when extracted,
    then its own "embedding" field comes back as a plain tuple, in order."""
    row: dict[str, Any] = {
        "object": "embedding",
        "index": 0,
        "embedding": [0.1, 0.2, 0.3],
    }

    assert _extract_embedding(row) == (0.1, 0.2, 0.3)


def test_extract_embedding_does_not_coerce_element_types() -> None:
    """Given a row whose own "embedding" field holds plain ints, when
    extracted, then those values are carried through unchanged -- this
    module wraps the sequence in a tuple, it does not cast each element to
    float."""
    row: dict[str, Any] = {"object": "embedding", "index": 0, "embedding": [1, 2, 3]}

    assert _extract_embedding(row) == (1, 2, 3)


async def test_embed_empty_texts_raises_embedding_error() -> None:
    """Given an empty batch, when embedded, then EmbeddingError is raised
    naming the empty input -- before any registry lookup or network call
    is attempted."""
    client = OllamaEmbeddingClient(_registry(_openrouter_entry()))

    with pytest.raises(EmbeddingError, match="texts must not be empty"):
        await client.embed([], model_id="glm-5.2")


async def test_embed_non_ollama_entry_raises_embedding_error_naming_backend() -> None:
    """Given a registry entry backed by a non-"ollama" backend, when
    embedded, then EmbeddingError names the offending backend -- checked
    before any network call, so no mock server is needed either way."""
    client = OllamaEmbeddingClient(_registry(_openrouter_entry()))

    with pytest.raises(EmbeddingError, match="openrouter"):
        await client.embed(["x"], model_id="glm-5.2")
