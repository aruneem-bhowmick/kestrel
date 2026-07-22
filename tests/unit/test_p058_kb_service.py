"""Unit tests for `KbService`: embed-then-store/search composition over
a fake `EmbeddingClient` and real, temp-file-backed `KnowledgeStore`
instances.

Every case here builds its own fake embedding client returning a
hand-picked vector (or raising) regardless of the text it is given --
the real Ollama network seam is already covered elsewhere, so nothing
in this module places a real embedding call.
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
from kestrel.kb.store import KnowledgeNote, KnowledgeStore, resolve_kb_path

pytestmark = [pytest.mark.p058, pytest.mark.unit]

_DIM = 4


@dataclasses.dataclass
class _FakeEmbeddingClient:
    """An `EmbeddingClient` returning a fixed vector for every text (or
    raising a fixed error instead), regardless of what it is given --
    these tests only care that `KbService` calls `embed` and handles
    what comes back correctly, not that a returned vector is
    semantically meaningful."""

    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    error: EmbeddingError | None = None

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Return `self.vector` once per input text, or raise
        `self.error` when set instead of returning anything."""
        if self.error is not None:
            raise self.error
        return tuple(self.vector for _ in texts)


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


def _service(
    tmp_path: Path,
    *,
    global_namespace: bool,
    client: _FakeEmbeddingClient | None = None,
) -> KbService:
    """A `KbService` scoped to `tmp_path`, at this module's shared
    dimension, wired to `client` (or a default-vector fake when
    omitted)."""
    return KbService(
        repo_root=tmp_path,
        config=KbConfig(global_namespace=global_namespace),
        embedding_client=client or _FakeEmbeddingClient(),
        embedding_model_id="fake-embed",
        embedding_dim=_DIM,
    )


def _seed_note(
    db_path: Path, *, text: str, embedding: tuple[float, ...], repo: str
) -> None:
    """Insert one note directly into the store at `db_path`, bypassing
    `KbService` entirely -- used to give the per-repo and global stores
    distinct, hand-picked vectors that a shared fake embedding client
    could not itself produce."""
    store = KnowledgeStore(db_path=db_path, embedding_dim=_DIM)
    try:
        store.add_note(
            KnowledgeNote(
                id=None,
                text=text,
                embedding=embedding,
                repo=repo,
                tags=(),
                source_task="seed",
                timestamp=0.0,
            )
        )
    finally:
        store.close()


async def test_add_note_with_global_namespace_disabled_writes_only_the_per_repo_store(
    tmp_path: Path,
) -> None:
    """Given `global_namespace=False`, when a note is added, then it
    comes back as exactly one `KnowledgeNote`, present in the per-repo
    store's own search, and absent from a directly-opened global store
    at the same (monkeypatched) path."""
    service = _service(tmp_path, global_namespace=False)

    added = await service.add_note("hello", tags=("x",), source_task="task-1")

    assert len(added) == 1
    assert added[0].id is not None
    assert added[0].text == "hello"

    per_repo = KnowledgeStore(
        db_path=resolve_kb_path(tmp_path, global_=False), embedding_dim=_DIM
    )
    try:
        [found] = per_repo.search((1.0, 0.0, 0.0, 0.0), top_k=5)
        assert found.note.text == "hello"
    finally:
        per_repo.close()

    global_store = KnowledgeStore(
        db_path=resolve_kb_path(tmp_path, global_=True), embedding_dim=_DIM
    )
    try:
        assert global_store.search((1.0, 0.0, 0.0, 0.0), top_k=5) == ()
    finally:
        global_store.close()


async def test_add_note_with_global_namespace_enabled_writes_both_stores(
    tmp_path: Path,
) -> None:
    """Given `global_namespace=True`, when a note is added, then it
    comes back as two `KnowledgeNote`s, one present in each store."""
    service = _service(tmp_path, global_namespace=True)

    added = await service.add_note("hello", tags=(), source_task="task-1")

    assert len(added) == 2
    assert {note.id for note in added} == {added[0].id, added[1].id}

    per_repo = KnowledgeStore(
        db_path=resolve_kb_path(tmp_path, global_=False), embedding_dim=_DIM
    )
    global_store = KnowledgeStore(
        db_path=resolve_kb_path(tmp_path, global_=True), embedding_dim=_DIM
    )
    try:
        assert len(per_repo.search((1.0, 0.0, 0.0, 0.0), top_k=5)) == 1
        assert len(global_store.search((1.0, 0.0, 0.0, 0.0), top_k=5)) == 1
    finally:
        per_repo.close()
        global_store.close()


async def test_add_note_shares_identical_fields_across_both_stores(
    tmp_path: Path,
) -> None:
    """Given `global_namespace=True`, when a note is added, then both
    persisted copies share the identical text, embedding, tags, source
    task, and timestamp -- only their store-assigned `id` may differ."""
    service = _service(tmp_path, global_namespace=True)

    first, second = await service.add_note(
        "hello", tags=("a", "b"), source_task="task-1"
    )

    assert first.text == second.text == "hello"
    assert first.embedding == second.embedding
    assert first.tags == second.tags == ("a", "b")
    assert first.source_task == second.source_task == "task-1"
    assert first.timestamp == second.timestamp


async def test_search_with_global_namespace_merges_and_orders_by_score(
    tmp_path: Path,
) -> None:
    """Given one note in the per-repo store and a different, farther
    note in the global store, when searched with `global_namespace=
    True`, then both come back, merged into one result set and ordered
    by score descending."""
    repo = str(tmp_path.resolve())
    _seed_note(
        resolve_kb_path(tmp_path, global_=False),
        text="near",
        embedding=(1.0, 0.0, 0.0, 0.0),
        repo=repo,
    )
    _seed_note(
        resolve_kb_path(tmp_path, global_=True),
        text="far",
        embedding=(0.0, 1.0, 0.0, 0.0),
        repo=repo,
    )
    service = _service(
        tmp_path,
        global_namespace=True,
        client=_FakeEmbeddingClient(vector=(1.0, 0.0, 0.0, 0.0)),
    )

    results = await service.search("query")

    assert [r.note.text for r in results] == ["near", "far"]
    assert results[0].score > results[1].score


async def test_search_with_global_namespace_disabled_ignores_the_global_store(
    tmp_path: Path,
) -> None:
    """Given a note only in the global store, when searched with
    `global_namespace=False`, then it never comes back."""
    repo = str(tmp_path.resolve())
    _seed_note(
        resolve_kb_path(tmp_path, global_=True),
        text="global-only",
        embedding=(1.0, 0.0, 0.0, 0.0),
        repo=repo,
    )
    service = _service(
        tmp_path,
        global_namespace=False,
        client=_FakeEmbeddingClient(vector=(1.0, 0.0, 0.0, 0.0)),
    )

    assert await service.search("query") == ()


async def test_search_respects_top_k_across_merged_stores(tmp_path: Path) -> None:
    """Given more matching notes across both stores than `config.top_k`,
    when searched, then only `top_k` results come back."""
    repo = str(tmp_path.resolve())
    _seed_note(
        resolve_kb_path(tmp_path, global_=False),
        text="one",
        embedding=(1.0, 0.0, 0.0, 0.0),
        repo=repo,
    )
    _seed_note(
        resolve_kb_path(tmp_path, global_=True),
        text="two",
        embedding=(0.9, 0.1, 0.0, 0.0),
        repo=repo,
    )
    service = KbService(
        repo_root=tmp_path,
        config=KbConfig(global_namespace=True, top_k=1),
        embedding_client=_FakeEmbeddingClient(vector=(1.0, 0.0, 0.0, 0.0)),
        embedding_model_id="fake-embed",
        embedding_dim=_DIM,
    )

    results = await service.search("query")

    assert len(results) == 1
    assert results[0].note.text == "one"


async def test_search_embedding_failure_surfaces_as_kb_service_error(
    tmp_path: Path,
) -> None:
    """Given an `EmbeddingClient` that raises `EmbeddingError`, when
    searched, then `KbServiceError` is raised instead of the raw
    embedding error."""
    service = _service(
        tmp_path,
        global_namespace=False,
        client=_FakeEmbeddingClient(error=EmbeddingError("boom")),
    )

    with pytest.raises(KbServiceError, match="boom"):
        await service.search("query")


async def test_add_note_embedding_failure_surfaces_as_kb_service_error(
    tmp_path: Path,
) -> None:
    """Given an `EmbeddingClient` that raises `EmbeddingError`, when a
    note is added, then `KbServiceError` is raised instead of the raw
    embedding error."""
    service = _service(
        tmp_path,
        global_namespace=False,
        client=_FakeEmbeddingClient(error=EmbeddingError("boom")),
    )

    with pytest.raises(KbServiceError, match="boom"):
        await service.add_note("hello", tags=(), source_task="task-1")


async def test_search_store_failure_surfaces_as_kb_service_error(
    tmp_path: Path,
) -> None:
    """Given an embedding client returning the zero vector -- rejected
    by `KnowledgeStore.search` itself, since normalizing it would divide
    by zero -- when searched, then `KbServiceError` wraps that store
    failure rather than letting it propagate raw."""
    service = _service(
        tmp_path,
        global_namespace=False,
        client=_FakeEmbeddingClient(vector=(0.0, 0.0, 0.0, 0.0)),
    )

    with pytest.raises(KbServiceError, match="store query failed"):
        await service.search("query")


async def test_add_note_store_failure_surfaces_as_kb_service_error(
    tmp_path: Path,
) -> None:
    """Given an embedding client returning the zero vector -- rejected
    by `KnowledgeStore.add_note` itself for the identical reason -- when
    a note is added, then `KbServiceError` wraps that store failure
    rather than letting it propagate raw."""
    service = _service(
        tmp_path,
        global_namespace=False,
        client=_FakeEmbeddingClient(vector=(0.0, 0.0, 0.0, 0.0)),
    )

    with pytest.raises(KbServiceError, match="store insert failed"):
        await service.add_note("hello", tags=(), source_task="task-1")


@pytest.mark.cost_regression
def test_kb_service_carries_no_cost_meter_field() -> None:
    """`KbService` carries no `CostMeter` (or any other cost-accounting
    collaborator) among its own dataclass fields -- proof by
    construction that neither `search` nor `add_note` can bill a task,
    since neither call has any meter to bill through in the first
    place."""
    field_names = {field.name for field in dataclasses.fields(KbService)}

    assert field_names == {
        "repo_root",
        "config",
        "embedding_client",
        "embedding_model_id",
        "embedding_dim",
    }
