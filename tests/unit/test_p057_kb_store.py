"""Tests for `KnowledgeStore`: insertion, nearest-neighbor search and its
tag filter, dimension-mismatch and zero-vector rejection, and on-disk
persistence across separate store instances.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.kb.store import KnowledgeNote, KnowledgeStore, KnowledgeStoreError

pytestmark = [pytest.mark.p057, pytest.mark.unit]

_DIM = 4


def _note(
    text: str,
    embedding: tuple[float, ...],
    *,
    tags: tuple[str, ...] = (),
    source_task: str = "task-1",
    timestamp: float = 0.0,
) -> KnowledgeNote:
    """A `KnowledgeNote` with `id=None`, ready for `add_note`."""
    return KnowledgeNote(
        id=None,
        text=text,
        embedding=embedding,
        repo="repo",
        tags=tags,
        source_task=source_task,
        timestamp=timestamp,
    )


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    """A fresh `KnowledgeStore` backed by a `tmp_path` file, at the
    dimension every case in this module shares."""
    return KnowledgeStore(db_path=tmp_path / "kb.sqlite3", embedding_dim=_DIM)


def test_search_with_the_identical_vector_returns_that_note_first(
    store: KnowledgeStore,
) -> None:
    """Given a note added with a given vector, when searched with that
    exact vector, then it comes back first with `score` approximately
    1.0."""
    added = store.add_note(_note("mine", (1.0, 0.0, 0.0, 0.0)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=5)

    assert results[0].note.id == added.id
    assert results[0].note.text == "mine"
    assert results[0].score == pytest.approx(1.0)


def test_notes_at_increasing_angular_distance_rank_in_expected_order(
    store: KnowledgeStore,
) -> None:
    """Given three notes at increasing angular distance from a query
    vector, when searched, then they rank nearest-first, and their
    scores fall in matching descending order."""
    store.add_note(_note("parallel", (1.0, 0.0, 0.0, 0.0)))
    store.add_note(_note("orthogonal", (0.0, 1.0, 0.0, 0.0)))
    store.add_note(_note("opposite", (-1.0, 0.0, 0.0, 0.0)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=5)

    assert [r.note.text for r in results] == ["parallel", "orthogonal", "opposite"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)
    assert results[2].score == pytest.approx(-1.0)


def test_tag_filter_excludes_notes_without_a_matching_tag(
    store: KnowledgeStore,
) -> None:
    """Given notes carrying different tags, when searched with a `tags`
    filter, then only notes carrying at least one of those tags come
    back."""
    store.add_note(_note("tagged x", (1.0, 0.0, 0.0, 0.0), tags=("x",)))
    store.add_note(_note("tagged y", (0.9, 0.1, 0.0, 0.0), tags=("y",)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=5, tags=frozenset({"x"}))

    assert [r.note.text for r in results] == ["tagged x"]


def test_tag_filter_none_applies_no_filter(store: KnowledgeStore) -> None:
    """Given notes carrying different tags, when searched with `tags=
    None`, then every note comes back regardless of its own tags."""
    store.add_note(_note("tagged x", (1.0, 0.0, 0.0, 0.0), tags=("x",)))
    store.add_note(_note("tagged y", (0.9, 0.1, 0.0, 0.0), tags=("y",)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=5, tags=None)

    assert {r.note.text for r in results} == {"tagged x", "tagged y"}


def test_tag_filter_with_no_matching_notes_returns_empty_not_an_error(
    store: KnowledgeStore,
) -> None:
    """Given a store with notes but none carrying the requested tag,
    when searched, then an empty result comes back rather than an
    error."""
    store.add_note(_note("tagged x", (1.0, 0.0, 0.0, 0.0), tags=("x",)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=5, tags=frozenset({"z"}))

    assert results == ()


def test_add_note_dimension_mismatch_raises_knowledge_store_error(
    store: KnowledgeStore,
) -> None:
    """Given an embedding whose length does not match the store's own
    `embedding_dim`, when added, then `KnowledgeStoreError` is raised."""
    with pytest.raises(KnowledgeStoreError, match="dimensions"):
        store.add_note(_note("short", (1.0, 0.0)))


def test_search_dimension_mismatch_raises_knowledge_store_error(
    store: KnowledgeStore,
) -> None:
    """Given a query embedding whose length does not match the store's
    own `embedding_dim`, when searched, then `KnowledgeStoreError` is
    raised."""
    store.add_note(_note("mine", (1.0, 0.0, 0.0, 0.0)))

    with pytest.raises(KnowledgeStoreError, match="dimensions"):
        store.search((1.0, 0.0), top_k=5)


def test_zero_vector_note_raises_knowledge_store_error(store: KnowledgeStore) -> None:
    """Given a note whose embedding is the all-zero vector, when added,
    then `KnowledgeStoreError` is raised rather than dividing by zero to
    normalize it."""
    with pytest.raises(KnowledgeStoreError, match="zero vector"):
        store.add_note(_note("zero", (0.0, 0.0, 0.0, 0.0)))


def test_zero_vector_query_raises_knowledge_store_error(store: KnowledgeStore) -> None:
    """Given a query embedding that is the all-zero vector, when
    searched, then `KnowledgeStoreError` is raised for the identical
    reason a zero-vector note is rejected on `add_note`."""
    store.add_note(_note("mine", (1.0, 0.0, 0.0, 0.0)))

    with pytest.raises(KnowledgeStoreError, match="zero vector"):
        store.search((0.0, 0.0, 0.0, 0.0), top_k=5)


def test_zero_vector_query_raises_even_when_top_k_is_zero(
    store: KnowledgeStore,
) -> None:
    """Given a query embedding that is the all-zero vector and `top_k=0`,
    when searched, then `KnowledgeStoreError` is still raised -- the
    zero-vector check runs before the `top_k` short-circuit, not after
    it, so an invalid query is never silently swallowed into an empty
    result."""
    store.add_note(_note("mine", (1.0, 0.0, 0.0, 0.0)))

    with pytest.raises(KnowledgeStoreError, match="zero vector"):
        store.search((0.0, 0.0, 0.0, 0.0), top_k=0)


def test_zero_vector_query_raises_even_against_an_empty_store(
    store: KnowledgeStore,
) -> None:
    """Given a query embedding that is the all-zero vector and a store
    with no notes at all, when searched, then `KnowledgeStoreError` is
    still raised -- the zero-vector check runs before the empty-store
    short-circuit, not after it."""
    with pytest.raises(KnowledgeStoreError, match="zero vector"):
        store.search((0.0, 0.0, 0.0, 0.0), top_k=5)


def test_add_note_with_id_already_set_raises_knowledge_store_error(
    store: KnowledgeStore,
) -> None:
    """Given a `KnowledgeNote` whose `id` is already populated, when
    added, then `KnowledgeStoreError` is raised rather than silently
    reinserting it."""
    already_added = store.add_note(_note("mine", (1.0, 0.0, 0.0, 0.0)))

    with pytest.raises(KnowledgeStoreError, match="note.id must be None"):
        store.add_note(already_added)


def test_closing_and_reopening_preserves_every_note(tmp_path: Path) -> None:
    """Given notes added through one `KnowledgeStore` instance, when it
    is closed and a fresh instance is opened at the same `db_path`,
    then a search against the new instance still returns every
    previously added note -- proving on-disk persistence, not merely an
    in-memory cache."""
    db_path = tmp_path / "kb.sqlite3"
    first = KnowledgeStore(db_path=db_path, embedding_dim=_DIM)
    first.add_note(_note("one", (1.0, 0.0, 0.0, 0.0)))
    first.add_note(_note("two", (0.0, 1.0, 0.0, 0.0)))
    first.close()

    second = KnowledgeStore(db_path=db_path, embedding_dim=_DIM)
    results = second.search((1.0, 0.0, 0.0, 0.0), top_k=5)
    second.close()

    assert {r.note.text for r in results} == {"one", "two"}


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Given an already-closed store, when closed again, then no error
    is raised."""
    once_closed = KnowledgeStore(db_path=tmp_path / "kb.sqlite3", embedding_dim=_DIM)

    once_closed.close()
    once_closed.close()


def test_search_against_an_empty_store_returns_empty(store: KnowledgeStore) -> None:
    """Given a store with no notes at all, when searched, then an empty
    result comes back rather than an error."""
    assert store.search((1.0, 0.0, 0.0, 0.0), top_k=5) == ()


def test_search_top_k_limits_the_result_count(store: KnowledgeStore) -> None:
    """Given more notes than `top_k`, when searched, then only `top_k`
    results come back, nearest first."""
    store.add_note(_note("a", (1.0, 0.0, 0.0, 0.0)))
    store.add_note(_note("b", (0.9, 0.1, 0.0, 0.0)))
    store.add_note(_note("c", (0.5, 0.5, 0.0, 0.0)))

    results = store.search((1.0, 0.0, 0.0, 0.0), top_k=2)

    assert len(results) == 2
    assert [r.note.text for r in results] == ["a", "b"]


def test_search_top_k_zero_returns_empty(store: KnowledgeStore) -> None:
    """Given `top_k=0`, when searched, then an empty result comes back
    without querying the underlying store for any candidates."""
    store.add_note(_note("a", (1.0, 0.0, 0.0, 0.0)))

    assert store.search((1.0, 0.0, 0.0, 0.0), top_k=0) == ()


def test_add_note_normalizes_the_stored_and_returned_embedding(
    store: KnowledgeStore,
) -> None:
    """Given a note whose embedding is not already unit length, when
    added, then both the returned note and the note read back by a
    later search carry the L2-normalized vector, not the raw input."""
    added = store.add_note(_note("scaled", (2.0, 0.0, 0.0, 0.0)))

    assert added.embedding == pytest.approx((1.0, 0.0, 0.0, 0.0))

    [result] = store.search((1.0, 0.0, 0.0, 0.0), top_k=5)
    assert result.note.embedding == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)
