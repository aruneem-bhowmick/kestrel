"""Compose an embedding client and one or two knowledge stores into the
single object every retrieval or writeback call actually needs.

`KbService` is the seam between `kestrel.kb.embeddings.EmbeddingClient`
(turns text into a vector) and `kestrel.kb.store.KnowledgeStore` (turns a
vector into a search result, or persists one alongside its note). A
caller never embeds text or opens a store directly: `search` and
`add_note` each do both steps in one call, deciding for themselves, from
`KbConfig.global_namespace`, whether the per-repo store alone or both
the per-repo and a global store participate.

A `KnowledgeStore` connection is opened fresh and closed again within a
single `search`/`add_note` call rather than held open across calls --
retrieval and writeback each happen at most a handful of times per task,
so pooling a connection across calls would add complexity this module
has no use for.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kestrel.config import KbConfig
from kestrel.kb.embeddings import EmbeddingClient, EmbeddingError
from kestrel.kb.store import (
    KnowledgeNote,
    KnowledgeStore,
    KnowledgeStoreError,
    ScoredNote,
    resolve_kb_path,
)


class KbServiceError(Exception):
    """A `KbService` call could not complete, whether the embedding step
    or one of the underlying stores is what actually failed -- wraps
    both `EmbeddingError` and `KnowledgeStoreError` behind one type, so a
    caller several layers up (a tool executor, a CLI command) only ever
    has to handle "the knowledge base is unavailable right now" once,
    regardless of which layer actually raised it."""


@dataclass(frozen=True, slots=True)
class KbService:
    """The knowledge base as one repo-scoped (plus optional global)
    collaborator: embed, then store or search, following `config`'s own
    global-namespace setting.

    Attributes:
        repo_root: The repo this service is scoped to -- both the
            per-repo store's own location and the `repo` field every
            note it adds carries are derived from this path.
        config: This repo's own knowledge-base settings.
        embedding_client: Computes a vector for a note's or a query's
            text.
        embedding_model_id: The registry id `embedding_client.embed`
            resolves against.
        embedding_dim: The fixed vector length every store this service
            opens is created at, or must already match.
    """

    repo_root: Path
    config: KbConfig
    embedding_client: EmbeddingClient
    embedding_model_id: str
    embedding_dim: int

    def _open(self, *, global_: bool) -> KnowledgeStore:
        """A fresh `KnowledgeStore` at this service's own repo-scoped or
        global path, at `embedding_dim` -- opened for the duration of a
        single call and closed again by that same call, never held open
        across calls."""
        return KnowledgeStore(
            db_path=resolve_kb_path(self.repo_root, global_=global_),
            embedding_dim=self.embedding_dim,
        )

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Embed `query`, search the per-repo store (and, when
        `config.global_namespace` is set, the global store too), and
        return the top `config.top_k` results across both, ordered by
        score descending. The two stores are physically separate
        databases with independently assigned note ids, so no
        deduplication step is needed to merge their results.

        Raises:
            KbServiceError: the embedding call or either store's own
                search fails.
        """
        try:
            (embedding,) = await self.embedding_client.embed(
                [query], model_id=self.embedding_model_id
            )
        except EmbeddingError as exc:
            raise KbServiceError(f"search: embedding failed: {exc}") from exc

        scopes = (False, True) if self.config.global_namespace else (False,)
        results: list[ScoredNote] = []
        for global_ in scopes:
            store = self._open(global_=global_)
            try:
                results.extend(
                    store.search(embedding, top_k=self.config.top_k, tags=tags)
                )
            except KnowledgeStoreError as exc:
                raise KbServiceError(f"search: store query failed: {exc}") from exc
            finally:
                store.close()

        results.sort(key=lambda scored: scored.score, reverse=True)
        return tuple(results[: self.config.top_k])

    async def add_note(
        self, text: str, *, tags: Sequence[str], source_task: str
    ) -> tuple[KnowledgeNote, ...]:
        """Embed `text` once and insert it into the per-repo store
        (always) and, when `config.global_namespace` is set, the global
        store too -- one persisted `KnowledgeNote` comes back per store
        written to (one normally, two with the global namespace
        enabled), each with its own store-assigned id. Every persisted
        copy shares the identical embedding, text, tags, source task,
        and timestamp; only which database holds it differs.

        Raises:
            KbServiceError: the embedding call or either store's own
                insert fails.
        """
        try:
            (embedding,) = await self.embedding_client.embed(
                [text], model_id=self.embedding_model_id
            )
        except EmbeddingError as exc:
            raise KbServiceError(f"add_note: embedding failed: {exc}") from exc

        note = KnowledgeNote(
            id=None,
            text=text,
            embedding=embedding,
            repo=str(self.repo_root.resolve()),
            tags=tuple(tags),
            source_task=source_task,
            timestamp=time.time(),
        )

        scopes = (False, True) if self.config.global_namespace else (False,)
        persisted: list[KnowledgeNote] = []
        for global_ in scopes:
            store = self._open(global_=global_)
            try:
                persisted.append(store.add_note(note))
            except KnowledgeStoreError as exc:
                raise KbServiceError(f"add_note: store insert failed: {exc}") from exc
            finally:
                store.close()
        return tuple(persisted)
