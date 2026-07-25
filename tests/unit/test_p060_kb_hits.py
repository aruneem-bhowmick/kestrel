"""Unit tests for `kestrel.tools.search._kb_hits` and `_render_kb_hit`:
the seam between `search`'s own `semantic` branch and a real
`KbService.search`, covering the "no service", "real hits", and
"service failure" cases without a real embedding model or store.

Every case here drives `_kb_hits` against a small fake standing in for
`KbService` -- a plain object declaring an async `search` method, never
a real `KbService` instance -- since `_kb_hits` only ever calls that one
method and this module's own annotation is a `TYPE_CHECKING`-only
import, so nothing here needs a real embedding client or on-disk store.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kestrel.kb.service import KbServiceError
from kestrel.kb.store import KnowledgeNote, ScoredNote
from kestrel.tools.search import _kb_hits, _KbHit, _render_kb_hit

pytestmark = [pytest.mark.p060, pytest.mark.unit]


def _note(text: str, *, source_task: str) -> KnowledgeNote:
    """A `KnowledgeNote` carrying `text`/`source_task`, with every other
    field set to an arbitrary, otherwise-unused fixed value."""
    return KnowledgeNote(
        id=1,
        text=text,
        embedding=(0.0,),
        repo="repo",
        tags=(),
        source_task=source_task,
        timestamp=0.0,
    )


class _FakeKbService:
    """Stands in for `KbService`: `search` always returns a fixed,
    already-scored tuple, regardless of the query text it is given."""

    def __init__(self, scored: Sequence[ScoredNote]) -> None:
        """Remember `scored` as the fixed reply every `search` call
        returns."""
        self._scored = tuple(scored)

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Ignore `query`/`tags` entirely and return this instance's own
        fixed `scored` tuple."""
        return self._scored


class _FailingKbService:
    """Stands in for a `KbService` whose own store or embedding step is
    unavailable: `search` always raises `KbServiceError`."""

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Raise `KbServiceError`, ignoring `query`/`tags` entirely."""
        raise KbServiceError("kb unavailable")


@pytest.mark.sanity
def test_kb_hits_returns_empty_list_when_kb_is_none() -> None:
    """Given `kb=None` (knowledge base disabled), when `_kb_hits` is
    called, then it returns `[]` without raising."""
    assert _kb_hits(None, "pattern", max_results=10) == []


def test_kb_hits_renders_scored_notes_in_order_capped_at_max_results() -> None:
    """Given a fake `KbService` returning three `ScoredNote`s, when
    `_kb_hits` is called with `max_results=2`, then it returns the first
    two, in the same order, each rendered into a `_KbHit`."""
    scored = (
        ScoredNote(note=_note("first", source_task="t1"), score=0.9),
        ScoredNote(note=_note("second", source_task="t2"), score=0.5),
        ScoredNote(note=_note("third", source_task="t3"), score=0.1),
    )
    kb = _FakeKbService(scored)

    hits = _kb_hits(kb, "pattern", max_results=2)

    assert hits == [
        _KbHit(text="first", score=0.9, source_task="t1"),
        _KbHit(text="second", score=0.5, source_task="t2"),
    ]


def test_kb_hits_returns_all_scored_notes_when_under_the_cap() -> None:
    """Given a fake `KbService` returning one `ScoredNote`, when
    `_kb_hits` is called with a `max_results` well above that count, then
    every note comes back, none dropped."""
    scored = (ScoredNote(note=_note("only", source_task="t1"), score=0.42),)
    kb = _FakeKbService(scored)

    hits = _kb_hits(kb, "pattern", max_results=50)

    assert hits == [_KbHit(text="only", score=0.42, source_task="t1")]


def test_kb_service_error_becomes_an_empty_list_rather_than_propagating() -> None:
    """Given a `KbService` whose `search` raises `KbServiceError`, when
    `_kb_hits` is called, then it returns `[]` rather than letting the
    error escape -- a knowledge-base outage degrades `search` to
    `rg`-only behavior, never fails the whole call."""
    hits = _kb_hits(_FailingKbService(), "pattern", max_results=10)

    assert hits == []


@pytest.mark.sanity
def test_render_kb_hit_renders_the_documented_shape() -> None:
    """Given one `_KbHit`, when rendered, then it reads as
    `[kb score={score:.2f} task={source_task}] {text}`."""
    hit = _KbHit(text="some note", score=0.876, source_task="task-9")

    assert _render_kb_hit(hit) == "[kb score=0.88 task=task-9] some note"
