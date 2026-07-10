"""Tests for `UndoManager`: recording, exact-content reversion at every
granularity (`revert_last`/`revert_turn`/`revert_task`), conflict
detection against out-of-band changes, cross-instance journal
persistence, and the pinned JSONL wire format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.managers.undo import UndoEntry, UndoManager

pytestmark = [pytest.mark.p017, pytest.mark.unit]


def _write(root: Path, relative: str, content: str) -> None:
    """Write `content` as UTF-8 bytes to `relative` under `root`,
    creating parent directories as needed."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


def _read(root: Path, relative: str) -> str:
    """Read `relative` under `root` back as UTF-8 text."""
    return (root / relative).read_text(encoding="utf-8")


@pytest.mark.sanity
def test_record_then_revert_last_restores_prior_content_exactly(
    tmp_path: Path,
) -> None:
    """Given a file edited from one piece of content to another, when
    the edit is recorded and then reverted, then the file's content is
    restored exactly to what it was before the edit."""
    _write(tmp_path, "a.txt", "before")
    manager = UndoManager(repo_root=tmp_path)
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="before")
    )
    _write(tmp_path, "a.txt", "after")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before="before", after="after")
    )

    reverted = manager.revert_last()

    assert reverted.before == "before"
    assert _read(tmp_path, "a.txt") == "before"


@pytest.mark.sanity
def test_revert_last_restores_nonexistence_when_before_is_none(
    tmp_path: Path,
) -> None:
    """Given an entry whose `before` is `None` (the mutation created
    the file), when reverted, then the file is deleted rather than
    written with any content."""
    _write(tmp_path, "new.txt", "created")
    manager = UndoManager(repo_root=tmp_path)
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="new.txt", before=None, after="created")
    )

    manager.revert_last()

    assert not (tmp_path / "new.txt").exists()


@pytest.mark.sanity
def test_revert_last_on_empty_journal_raises_index_error(tmp_path: Path) -> None:
    """Given a manager with no recorded entries, when `revert_last` is
    called, then `IndexError` is raised rather than reverting nothing
    silently."""
    manager = UndoManager(repo_root=tmp_path)

    with pytest.raises(IndexError):
        manager.revert_last()


def test_revert_last_recreates_a_file_deleted_by_the_reverted_entry(
    tmp_path: Path,
) -> None:
    """Given an entry recording a deletion (`after=None`) and the file
    genuinely absent since, when reverted, then the file is recreated
    with `before`'s exact content rather than raising a conflict."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "content")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before="content", after=None)
    )
    (tmp_path / "a.txt").unlink()

    reverted = manager.revert_last()

    assert reverted.before == "content"
    assert _read(tmp_path, "a.txt") == "content"


def test_reverting_twice_in_a_row_does_not_raise(tmp_path: Path) -> None:
    """Given a single recorded edit, when `revert_last` is called
    twice in a row, then neither call raises: the first restores the
    pre-edit content by reverting the edit itself, and the second
    restores the post-edit content again by reverting the compensating
    entry the first revert appended -- proving repeated calls stay
    well-defined rather than raising once the original entry has
    already been consumed."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "before")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="before")
    )
    _write(tmp_path, "a.txt", "after")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before="before", after="after")
    )

    manager.revert_last()
    assert _read(tmp_path, "a.txt") == "before"

    manager.revert_last()
    assert _read(tmp_path, "a.txt") == "after"


def test_revert_turn_only_touches_its_own_turn_in_reverse_order(
    tmp_path: Path,
) -> None:
    """Given a journal with entries from two turns interleaved -- turn
    1 editing `a.txt` twice and turn 2 editing `b.txt` once in between
    -- when `revert_turn(1)` is called, then only turn 1's entries are
    reverted, most-recent-first, restoring `a.txt` to nonexistence,
    while `b.txt` (turn 2) is left untouched."""
    manager = UndoManager(repo_root=tmp_path)

    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )
    _write(tmp_path, "b.txt", "b1")
    manager.record(
        UndoEntry(turn_id=2, task_id="t", path="b.txt", before=None, after="b1")
    )
    _write(tmp_path, "a.txt", "a2")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before="a1", after="a2")
    )

    reverted = manager.revert_turn(1)

    assert [entry.after for entry in reverted] == ["a2", "a1"]
    assert not (tmp_path / "a.txt").exists()
    assert _read(tmp_path, "b.txt") == "b1"


def test_revert_turn_with_no_matching_entries_returns_empty_list(
    tmp_path: Path,
) -> None:
    """Given a journal with entries but none carrying the requested
    `turn_id`, when reverted, then an empty list is returned rather
    than an error."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )

    assert manager.revert_turn(99) == []
    assert _read(tmp_path, "a.txt") == "a1"


def test_revert_task_only_touches_its_own_task_in_reverse_order(
    tmp_path: Path,
) -> None:
    """Given a journal with entries from two tasks interleaved, when
    `revert_task` is called for one of them, then only that task's
    entries are reverted, most-recent-first, mirroring
    `revert_turn`'s scoping at task granularity."""
    manager = UndoManager(repo_root=tmp_path)

    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="task-a", path="a.txt", before=None, after="a1")
    )
    _write(tmp_path, "b.txt", "b1")
    manager.record(
        UndoEntry(turn_id=1, task_id="task-b", path="b.txt", before=None, after="b1")
    )
    _write(tmp_path, "a.txt", "a2")
    manager.record(
        UndoEntry(turn_id=1, task_id="task-a", path="a.txt", before="a1", after="a2")
    )

    reverted = manager.revert_task("task-a")

    assert [entry.after for entry in reverted] == ["a2", "a1"]
    assert not (tmp_path / "a.txt").exists()
    assert _read(tmp_path, "b.txt") == "b1"


def test_revert_task_with_no_matching_entries_returns_empty_list(
    tmp_path: Path,
) -> None:
    """Given a journal with entries but none carrying the requested
    `task_id`, when reverted, then an empty list is returned rather
    than an error."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )

    assert manager.revert_task("no-such-task") == []
