"""Red-team proof that `build_kb_context` cannot smuggle a hostile note
out of its own frame: every payload in the injection corpus, standing in
for a knowledge-base note's own text, still comes back framed by exactly
the real opening/closing markers -- `"kb"` being a brand-new untrusted-
content source kind gets the same adversarial coverage every other
source kind already has.
"""

from __future__ import annotations

import pytest

from kestrel.kb.retrieval import build_kb_context
from kestrel.kb.store import KnowledgeNote, ScoredNote
from kestrel.security.corpus import load_corpus

pytestmark = [pytest.mark.p061, pytest.mark.unit, pytest.mark.redteam]


class _HostileKbService:
    """A `KbService` stand-in whose one scored note carries a hostile
    payload as its own text."""

    embedding_model_id = "nomic-embed-text"

    def __init__(self, payload: str) -> None:
        """Remember `payload` as the one note this service's `search`
        returns."""
        self._payload = payload

    async def search(
        self, query: str, *, tags: frozenset[str] | None = None
    ) -> tuple[ScoredNote, ...]:
        """Ignore `query`/`tags` and return one `ScoredNote` carrying
        this instance's own hostile payload as its note text."""
        note = KnowledgeNote(
            id=1,
            text=self._payload,
            embedding=(0.0,),
            repo="repo",
            tags=(),
            source_task="hostile-task",
            timestamp=0.0,
        )
        return (ScoredNote(note=note, score=0.99),)


async def test_every_corpus_payload_stays_framed_exactly_once() -> None:
    """Given every payload in the injection corpus, each standing in for
    a knowledge-base note's own text, when `build_kb_context` runs
    against a knowledge base whose one match carries that payload, then
    the returned block still starts with the real opening header, still
    ends with the real closing delimiter, and none of that case's
    forbidden markers appear more than the one time `frame_untrusted`
    itself emits them.
    """
    for case in load_corpus():
        context = await build_kb_context(
            "irrelevant query", kb=_HostileKbService(case.payload)
        )

        assert context is not None
        assert context.startswith("<<<UNTRUSTED:kb:")
        assert context.endswith("<<<END_UNTRUSTED>>>")
        for marker in case.forbidden_markers:
            assert context.count(marker) == 1
