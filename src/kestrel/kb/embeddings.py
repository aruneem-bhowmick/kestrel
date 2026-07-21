"""A local Ollama-backed embedding client.

`OllamaEmbeddingClient` turns a batch of plain strings into their own
embedding vectors through `litellm.aembedding`'s `"ollama/"` route, rather
than a hand-rolled HTTP client -- reusing the same vendor-abstraction
library `kestrel.provider.litellm_client` already depends on instead of
adding a second HTTP stack. This is a deliberate second, independent
adapter next to `LiteLLMClient`, not a reuse of it: `ProviderClient.complete`
is a chat-completion contract (`messages`/`tools`/`effort`) with no
embedding shape, and `_litellm_params` already raises `ServerError` for the
`"ollama"` backend on that path (chat completions are not implemented for
it) -- this module calls `litellm.aembedding` directly and never touches
`LiteLLMClient` or `_litellm_params` at all.

Confirmed by reading the installed `litellm` package directly
(`litellm/llms/ollama/completion/handler.py`): litellm's `"ollama/"`
embedding route POSTs `{"model": ..., "input": [...]}` to
`{api_base}/api/embed` (Ollama's newer, batched endpoint, not the older
singular `/api/embeddings`) and expects back `{"embeddings": [[...],
...]}`, which litellm folds into `EmbeddingResponse.data` as `[{"object":
"embedding", "index": i, "embedding": [...]}, ...]`. If a future litellm
upgrade changes this wire contract, `_extract_embedding` is the one line
in this module to adjust, not a design change.

`litellm.suppress_debug_info = True` is already set globally by
`kestrel.provider.litellm_client`'s own import-time side effect, so this
module does not repeat it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

import litellm

from kestrel.registry.model import Registry

_EMBED_TIMEOUT_S: Final[float] = 60.0


class EmbeddingError(Exception):
    """A batch could not be embedded: empty input, a non-`"ollama"`
    registry entry named, or the underlying `litellm.aembedding` call
    raised. `str(self)` names the remedy, never a raw litellm
    exception -- callers never need to import litellm's own exception
    types to handle this module's failures."""


class EmbeddingClient(Protocol):
    """Vendor-neutral embedding contract, mirroring
    `kestrel.provider.base.ProviderClient`'s own role for chat
    completions -- a distinct protocol, not a shared base, since the
    two shapes (streamed chat events vs. a batch of vectors) share no
    useful surface."""

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Embed every string in `texts`, in order; returns one vector
        per input, same order, same length."""
        ...


def _extract_embedding(row: Mapping[str, Any]) -> tuple[float, ...]:
    """One `EmbeddingResponse.data` row's own `"embedding"` field, as a
    plain tuple of floats. The one line this module adjusts if a future
    litellm version changes this row's shape."""
    return tuple(row["embedding"])


class OllamaEmbeddingClient:
    """`EmbeddingClient` over `litellm.aembedding`'s `"ollama/"` route.
    Bound to a `Registry` exactly like `LiteLLMClient`, so it can serve
    any `"ollama"`-backed entry in it."""

    def __init__(self, registry: Registry) -> None:
        """Bind this client to the registry used to resolve `model_id`."""
        self._registry = registry

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Batch-embed `texts` in one `litellm.aembedding` call.

        Raises:
            EmbeddingError: `texts` is empty; `model_id`'s registry
                entry is not `backend="ollama"` (a caller-contract
                assertion -- every real call site resolves `model_id`
                against a `"local"`-tagged entry, which this codebase's
                own registry validation already requires to be reachable
                as a direct backend, i.e. `endpoint` is always set);
                the underlying `litellm.aembedding` call raises for any
                reason (network failure, malformed response, Ollama not
                running) -- wrapped, never propagated raw.
        """
        if not texts:
            raise EmbeddingError("embed: texts must not be empty")
        entry = self._registry.get(model_id)
        if entry.backend != "ollama":
            raise EmbeddingError(
                f"model {model_id!r} is not an ollama-backed entry "
                f"(backend={entry.backend!r})"
            )
        try:
            response = await litellm.aembedding(
                model=f"ollama/{entry.provider_model}",
                input=list(texts),
                api_base=entry.endpoint,
                timeout=_EMBED_TIMEOUT_S,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"embedding call failed for model {model_id!r}: {exc}"
            ) from exc
        ordered = sorted(response.data, key=lambda row: row["index"])
        return tuple(_extract_embedding(row) for row in ordered)
