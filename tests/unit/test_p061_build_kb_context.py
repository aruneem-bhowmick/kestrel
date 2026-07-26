"""Unit tests for `kestrel.kb.retrieval.build_kb_context`: the seam
between a task's own description and a real `KbService.search`, covering
the "no service", "real hits", "empty result", and "service failure"
cases without a real embedding model or store.

Every case here drives `build_kb_context` against a small fake standing
in for `KbService` -- a plain object declaring an async `search` method,
never a real `KbService` instance -- matching `test_p060_kb_hits.py`'s
own fake-collaborator pattern.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kestrel.kb.retrieval import build_kb_context
from kestrel.kb.service import KbServiceError
from kestrel.kb.store import KnowledgeNote, ScoredNote

pytestmark = [pytest.mark.p061, pytest.mark.unit]

_EMBEDDING_MODEL_ID = "nomic-embed-text"


def _note(text: str) -> KnowledgeNote:
    """A `KnowledgeNote` carrying `text`, with every other field set to
    an arbitrary, otherwise-unused fixed value."""
    return KnowledgeNote(
        id=1,
        text=text,
        embedding=(0.0,),
        repo="repo",
        tags=(),
        source_task="prior-task",
        timestamp=0.0,
    )


class _FakeKbService:
    """Stands in for `KbService`: `search` always returns a fixed,
    already-scored tuple, regardless of the query text it is given, and
    exposes the same `embedding_model_id` attribute `build_kb_context`
    reads as `frame_untrusted`'s `origin`."""

    def __init__(self, scored: Sequence[ScoredNote]) -> None:
        """Remember `scored` as the fixed reply every `search` call
        returns."""
        self._scored = tuple(scored)
        self.embedding_model_id = _EMBEDDING_MODEL_ID

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Ignore `query`/`tags` entirely and return this instance's own
        fixed `scored` tuple."""
        return self._scored


class _FailingKbService:
    """Stands in for a `KbService` whose own store or embedding step is
    unavailable: `search` always raises `KbServiceError`."""

    embedding_model_id = _EMBEDDING_MODEL_ID

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Raise `KbServiceError`, ignoring `query`/`tags` entirely."""
        raise KbServiceError("kb unavailable")


@pytest.mark.sanity
async def test_returns_none_when_kb_is_none() -> None:
    """Given `kb=None` (knowledge base disabled for this task), when
    `build_kb_context` is called, then it returns `None` without
    embedding anything or touching a store."""
    assert await build_kb_context("do the thing", kb=None) is None


async def test_returns_none_when_search_finds_no_notes() -> None:
    """Given a fake `KbService` whose `search` returns an empty tuple,
    when `build_kb_context` is called, then it returns `None` rather
    than an empty, degenerate framed block."""
    kb = _FakeKbService(())

    assert await build_kb_context("do the thing", kb=kb) is None


async def test_renders_every_scored_note_as_one_framed_block() -> None:
    """Given a fake `KbService` returning two `ScoredNote`s, when
    `build_kb_context` is called, then the result is one `frame_untrusted`
    block, `source="kb"`, carrying both notes' own text as one
    `"- {text}"` line each, in the order `search` returned them."""
    scored = (
        ScoredNote(note=_note("prefer tabs in this repo"), score=0.9),
        ScoredNote(note=_note("run tests via uv run pytest"), score=0.5),
    )
    kb = _FakeKbService(scored)

    context = await build_kb_context("how should I format this file", kb=kb)

    assert context is not None
    assert context.startswith(f"<<<UNTRUSTED:kb:{_EMBEDDING_MODEL_ID}>>>\n")
    assert context.endswith("<<<END_UNTRUSTED>>>")
    assert "- prefer tabs in this repo" in context
    assert "- run tests via uv run pytest" in context
    lines = context.splitlines()
    assert lines.index("- prefer tabs in this repo") < lines.index(
        "- run tests via uv run pytest"
    )


async def test_kb_service_error_becomes_none_rather_than_propagating() -> None:
    """Given a `KbService` whose `search` raises `KbServiceError`, when
    `build_kb_context` is called, then it returns `None` rather than
    letting the error escape -- a knowledge-base outage degrades
    retrieval to "found nothing" rather than failing the whole task
    before its first turn even runs."""
    context = await build_kb_context("do the thing", kb=_FailingKbService())

    assert context is None
