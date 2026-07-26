"""Unit tests for `kestrel.kb.writeback.commit_learnings`: committing
only the approved subset of a batch of proposed learnings, against a
fake `KbService` recording every call it receives -- never a real
embedding client or store.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kestrel.kb.store import KnowledgeNote
from kestrel.kb.writeback import ProposedLearning, WritebackDecision, commit_learnings

pytestmark = [pytest.mark.p062, pytest.mark.unit]


class _RecordingKbService:
    """Stands in for `KbService`: `add_note` records the exact
    `text`/`tags`/`source_task` it was called with and returns one
    `KnowledgeNote`, assigning ids in call order starting at 1."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    async def add_note(
        self, text: str, *, tags: Sequence[str], source_task: str
    ) -> tuple[KnowledgeNote, ...]:
        """Record this call and return one persisted-looking
        `KnowledgeNote` built from its own arguments."""
        self.calls.append((text, tuple(tags), source_task))
        return (
            KnowledgeNote(
                id=len(self.calls),
                text=text,
                embedding=(0.0,),
                repo="repo",
                tags=tuple(tags),
                source_task=source_task,
                timestamp=0.0,
            ),
        )


class _MultiNoteKbService:
    """Stands in for a `KbService` with `global_namespace` enabled:
    `add_note` returns two persisted `KnowledgeNote`s per call, one per
    store written to."""

    async def add_note(
        self, text: str, *, tags: Sequence[str], source_task: str
    ) -> tuple[KnowledgeNote, ...]:
        """Return two `KnowledgeNote`s sharing identical fields, standing
        in for a per-repo-plus-global write."""
        note = KnowledgeNote(
            id=1,
            text=text,
            embedding=(0.0,),
            repo="repo",
            tags=tuple(tags),
            source_task=source_task,
            timestamp=0.0,
        )
        return (note, note)


def _learnings(*texts: str) -> tuple[ProposedLearning, ...]:
    """Build one `ProposedLearning` per `texts` entry, each with a fixed,
    otherwise-unused single tag."""
    return tuple(ProposedLearning(text=text, tags=("tag",)) for text in texts)


async def test_commits_only_the_approved_indexed_entries() -> None:
    """Given three learnings whose decisions are approve/skip/approve,
    when committed, then the fake service records exactly the first and
    third learnings' own text and tags -- the skipped middle entry is
    never passed to `add_note`."""
    kb = _RecordingKbService()
    learnings = _learnings("keep this one", "drop this one", "keep this too")
    decisions: tuple[WritebackDecision, ...] = ("approve", "skip", "approve")

    result = await commit_learnings(
        learnings, decisions=decisions, task_id="task-1", kb=kb
    )

    assert [call[0] for call in kb.calls] == ["keep this one", "keep this too"]
    assert all(call[1] == ("tag",) for call in kb.calls)
    assert all(call[2] == "task-1" for call in kb.calls)
    assert len(result) == 2
    assert {note.text for note in result} == {"keep this one", "keep this too"}


async def test_no_approvals_never_calls_add_note() -> None:
    """Given every decision is `"skip"`, when committed, then `add_note`
    is never called and the result is empty."""
    kb = _RecordingKbService()
    learnings = _learnings("a", "b")

    result = await commit_learnings(
        learnings, decisions=("skip", "skip"), task_id="task-1", kb=kb
    )

    assert kb.calls == []
    assert result == ()


async def test_flattens_multiple_persisted_notes_per_commit() -> None:
    """Given a `KbService` whose `add_note` returns two notes per call
    (standing in for a global-namespace write), when two learnings are
    both approved, then the result flattens all four persisted notes
    into one tuple, not a tuple of two two-element groups."""
    learnings = _learnings("first", "second")

    result = await commit_learnings(
        learnings,
        decisions=("approve", "approve"),
        task_id="task-1",
        kb=_MultiNoteKbService(),
    )

    assert len(result) == 4


async def test_mismatched_lengths_raise_before_add_note_is_ever_called() -> None:
    """Given more learnings than decisions, when committed, then
    `ValueError` is raised before the fake service's `add_note` is
    called even once."""
    kb = _RecordingKbService()
    learnings = _learnings("a", "b")

    with pytest.raises(ValueError, match="2 learnings but 1 decisions"):
        await commit_learnings(
            learnings, decisions=("approve",), task_id="task-1", kb=kb
        )

    assert kb.calls == []
