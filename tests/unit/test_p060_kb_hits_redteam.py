"""Red-team proof that `search`'s knowledge-base section cannot smuggle a
hostile note out of its own frame: every payload in the injection
corpus, standing in for a knowledge-base note's own text, still comes
back through `search` wrapped by the real frame markers exactly once --
the whole return value shares one frame, not a second, separately-framed
section for the knowledge-base half.

`rg` itself is monkeypatched to a fixed "no matches" result rather than
requiring a real binary: the property under test is `search`'s own
merging and framing of a knowledge-base hit, independent of whatever
`rg` itself found.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kestrel.kb.store import KnowledgeNote, ScoredNote
from kestrel.security.corpus import load_corpus
from kestrel.tools.search import SearchArgs, search

pytestmark = [pytest.mark.p060, pytest.mark.unit, pytest.mark.redteam]

_PATTERN = "anything"


class _HostileKbService:
    """A `KbService` stand-in whose one scored note carries a hostile
    payload as its own text."""

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


def _no_matches(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    """A canned `rg` reply reporting no matches, regardless of the
    invocation -- `rg`'s own behavior is not what this suite proves."""
    return subprocess.CompletedProcess(args=["rg"], returncode=1, stdout="", stderr="")


def test_every_corpus_payload_stays_framed_exactly_once_as_a_kb_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given every payload in the injection corpus, each standing in for
    a knowledge-base note's own text, when `search` runs in `semantic`
    mode against a knowledge base whose one match carries that payload,
    then the returned value still starts with the real opening header,
    still ends with the real closing delimiter, and none of that case's
    forbidden markers appear more than the one time `frame_untrusted`
    itself emits them -- proving the knowledge-base section is covered
    by the same single frame as the rest of the response, not a second,
    separately-escaped one.
    """
    monkeypatch.setattr(subprocess, "run", _no_matches)

    for case in load_corpus():
        framed = search(
            SearchArgs(pattern=_PATTERN, semantic=True),
            repo_root=tmp_path,
            kb=_HostileKbService(case.payload),
        )

        assert framed.startswith("<<<UNTRUSTED:search_result:")
        assert framed.endswith("<<<END_UNTRUSTED>>>")
        for marker in case.forbidden_markers:
            assert framed.count(marker) == 1
