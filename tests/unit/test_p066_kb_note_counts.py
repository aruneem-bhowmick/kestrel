"""Unit tests for the note-count queries the `/kb` palette entry reads:
`KnowledgeStore.count` and `KbService.count_notes`.

Every `KbService` case here is wired to an embedding client that raises
if it is ever called -- proof that counting notes never embeds
anything, the same "count is a plain read, not a search" contract
`search`/`add_note`'s own dedicated fakes prove for their own calls in
`test_p058_kb_service.py`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

from kestrel.config import KbConfig
from kestrel.kb import store as kb_store
from kestrel.kb.embeddings import EmbeddingError
from kestrel.kb.service import KbService, KbServiceError
from kestrel.kb.store import (
    KnowledgeNote,
    KnowledgeStore,
    KnowledgeStoreError,
    resolve_kb_path,
)

pytestmark = [pytest.mark.p066, pytest.mark.unit, pytest.mark.sanity]

_DIM = 4


@dataclasses.dataclass
class _NeverCalledEmbeddingClient:
    """An `EmbeddingClient` whose `embed` always raises -- wiring this
    into a `KbService` under test proves a call under test never
    embeds anything."""

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Raise unconditionally; no case in this module expects this
        to ever run."""
        raise EmbeddingError("count_notes must never embed anything")


@pytest.fixture
def global_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user data
    directory a global-namespace store would otherwise resolve to."""
    return tmp_path_factory.mktemp("globaldata")


@pytest.fixture(autouse=True)
def _patch_global_path(monkeypatch: pytest.MonkeyPatch, global_data_dir: Path) -> None:
    """Point `resolve_kb_path`'s own global-path lookup at `global_data_dir`
    so no test in this module ever touches a real per-user directory."""
    monkeypatch.setattr(
        kb_store.platformdirs,
        "user_data_dir",
        lambda appname: str(global_data_dir),  # noqa: ARG005
    )


def _service(tmp_path: Path, *, global_namespace: bool) -> KbService:
    """A `KbService` scoped to `tmp_path`, wired to an embedding client
    that must never be called."""
    return KbService(
        repo_root=tmp_path,
        config=KbConfig(global_namespace=global_namespace),
        embedding_client=_NeverCalledEmbeddingClient(),
        embedding_model_id="fake-embed",
        embedding_dim=_DIM,
    )


def _seed_note(db_path: Path, *, text: str, repo: str) -> None:
    """Insert one note directly into the store at `db_path`, bypassing
    `KbService` entirely -- an arbitrary, unit-length vector, since no
    case here ever searches by similarity."""
    store = KnowledgeStore(db_path=db_path, embedding_dim=_DIM)
    try:
        store.add_note(
            KnowledgeNote(
                id=None,
                text=text,
                embedding=(1.0, 0.0, 0.0, 0.0),
                repo=repo,
                tags=(),
                source_task="seed",
                timestamp=0.0,
            )
        )
    finally:
        store.close()


def test_store_count_is_zero_for_a_fresh_store(tmp_path: Path) -> None:
    """Given a freshly opened store with no notes, when counted, then
    it reports zero."""
    store = KnowledgeStore(db_path=tmp_path / "kb.sqlite3", embedding_dim=_DIM)
    try:
        assert store.count() == 0
    finally:
        store.close()


def test_store_count_reflects_every_added_note(tmp_path: Path) -> None:
    """Given three added notes, when counted, then the store reports
    three, regardless of note content."""
    store = KnowledgeStore(db_path=tmp_path / "kb.sqlite3", embedding_dim=_DIM)
    try:
        for i in range(3):
            store.add_note(
                KnowledgeNote(
                    id=None,
                    text=f"note {i}",
                    embedding=(1.0, 0.0, 0.0, 0.0),
                    repo="repo",
                    tags=(),
                    source_task="task-1",
                    timestamp=0.0,
                )
            )

        assert store.count() == 3
    finally:
        store.close()


def test_store_count_wraps_a_raw_sqlite_error_as_knowledge_store_error(
    tmp_path: Path,
) -> None:
    """Given a store whose underlying connection has already been
    closed out from under it, when counted, then the raw `sqlite3.Error`
    surfaces as `KnowledgeStoreError` instead, matching every other
    query this class runs."""
    store = KnowledgeStore(db_path=tmp_path / "kb.sqlite3", embedding_dim=_DIM)
    store._conn.close()

    with pytest.raises(KnowledgeStoreError, match="count: query failed"):
        store.count()


async def test_count_notes_with_global_namespace_disabled_reports_only_the_per_repo_count(
    tmp_path: Path,
) -> None:
    """Given `global_namespace=False` and two notes in the per-repo
    store (plus one in a directly-opened global store, which
    `count_notes` must never touch), when counted, then the result is
    `(2, None)`."""
    repo = str(tmp_path.resolve())
    _seed_note(resolve_kb_path(tmp_path, global_=False), text="one", repo=repo)
    _seed_note(resolve_kb_path(tmp_path, global_=False), text="two", repo=repo)
    _seed_note(resolve_kb_path(tmp_path, global_=True), text="global-only", repo=repo)
    service = _service(tmp_path, global_namespace=False)

    assert service.count_notes() == (2, None)


async def test_count_notes_with_global_namespace_enabled_reports_both_counts(
    tmp_path: Path,
) -> None:
    """Given `global_namespace=True`, one note in the per-repo store,
    and two in the global store, when counted, then the result is
    `(1, 2)`."""
    repo = str(tmp_path.resolve())
    _seed_note(resolve_kb_path(tmp_path, global_=False), text="one", repo=repo)
    _seed_note(resolve_kb_path(tmp_path, global_=True), text="two", repo=repo)
    _seed_note(resolve_kb_path(tmp_path, global_=True), text="three", repo=repo)
    service = _service(tmp_path, global_namespace=True)

    assert service.count_notes() == (1, 2)


async def test_count_notes_never_opens_a_store_that_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    """Given a repo whose knowledge base has never been written to,
    when counted, then the result is `(0, None)` -- `count_notes`
    itself creates an (empty) store on first open, matching every
    other `KbService` call's own lazy-creation behavior, rather than
    raising for a knowledge base that simply has nothing in it yet."""
    service = _service(tmp_path, global_namespace=False)

    assert service.count_notes() == (0, None)


async def test_count_notes_open_failure_surfaces_as_kb_service_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a store that fails to open (its own `__init__` raises,
    modeling a disk or permissions failure), when counted, then the raw
    error never escapes -- it surfaces as `KbServiceError` instead."""

    def _raise(*args: object, **kwargs: object) -> None:
        """Stand in for `KnowledgeStore.__init__`, always failing."""
        raise RuntimeError("disk full")

    monkeypatch.setattr(KnowledgeStore, "__init__", _raise)
    service = _service(tmp_path, global_namespace=False)

    with pytest.raises(KbServiceError, match="failed to open the per-repo store"):
        service.count_notes()
