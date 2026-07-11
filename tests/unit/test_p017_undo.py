"""Tests for `UndoManager`: recording, exact-content reversion at every
granularity (`revert_last`/`revert_turn`/`revert_task`), conflict
detection against out-of-band changes, cross-instance journal
persistence, and the pinned JSONL wire format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kestrel.managers.undo import UndoConflictError, UndoEntry, UndoManager
from kestrel.security.corpus import load_corpus

pytestmark = [pytest.mark.p017, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p017_undo_entry.golden"
)


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


def test_journal_persists_across_manager_instances(tmp_path: Path) -> None:
    """Given entries recorded through one `UndoManager` instance, when
    a second instance is constructed pointed at the same
    `journal_path`, then it can revert those entries too -- the
    journal, not the in-process list, is the source of truth."""
    first = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "a1")
    first.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )

    second = UndoManager(repo_root=tmp_path, journal_path=first.journal_path)
    reverted = second.revert_last()

    assert reverted.after == "a1"
    assert not (tmp_path / "a.txt").exists()


def test_record_creates_the_journal_directory_and_file_lazily(tmp_path: Path) -> None:
    """Given a fresh repo with no `.kestrel` directory yet, when a
    manager is constructed, then nothing is created on disk; only the
    first `record` call creates the directory and the journal file."""
    manager = UndoManager(repo_root=tmp_path)
    assert not manager.journal_path.exists()
    assert not manager.journal_path.parent.exists()

    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )

    assert manager.journal_path == tmp_path / ".kestrel" / "undo.jsonl"
    assert manager.journal_path.exists()


def test_explicit_journal_path_overrides_the_default(tmp_path: Path) -> None:
    """Given an explicit `journal_path`, when a manager is
    constructed, then it uses that path instead of the default
    `.kestrel/undo.jsonl` location."""
    custom = tmp_path / "custom" / "journal.jsonl"
    manager = UndoManager(repo_root=tmp_path, journal_path=custom)

    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )

    assert custom.exists()
    assert not (tmp_path / ".kestrel").exists()


def test_out_of_band_change_raises_conflict_and_leaves_the_file_untouched(
    tmp_path: Path,
) -> None:
    """Given a file whose content has changed since the entry being
    reverted was recorded, when reverted, then `UndoConflictError`
    names the path and the file's (tampered) content is left exactly
    as it was, not overwritten."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before=None, after="a1")
    )
    _write(tmp_path, "a.txt", "tampered out of band")

    with pytest.raises(UndoConflictError, match="a.txt"):
        manager.revert_last()

    assert _read(tmp_path, "a.txt") == "tampered out of band"


def test_conflict_also_applies_when_a_deletion_is_reverted_over_a_recreated_file(
    tmp_path: Path,
) -> None:
    """Given an entry recording a deletion (`after=None`), when the
    file has since been recreated out of band before the revert runs,
    then `UndoConflictError` is raised rather than deleting the
    recreated file."""
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "a.txt", "a1")
    manager.record(
        UndoEntry(turn_id=1, task_id="t", path="a.txt", before="a1", after=None)
    )
    _write(tmp_path, "a.txt", "recreated out of band")

    with pytest.raises(UndoConflictError, match="a.txt"):
        manager.revert_last()

    assert _read(tmp_path, "a.txt") == "recreated out of band"


@pytest.mark.parametrize(
    "case_id",
    ["zero_width_smuggled_instruction", "readme_ignore_previous_instructions"],
)
def test_unicode_content_round_trips_byte_exact(tmp_path: Path, case_id: str) -> None:
    """Given a file whose content is one of the injection corpus's
    Unicode-laden payloads -- proving this manager's journaling and
    reversion has no interaction with `frame_untrusted`'s own escaping,
    since nothing here ever frames anything -- when recorded and then
    reverted, then the exact original bytes come back, unchanged
    down to every zero-width and multi-byte character."""
    payload = next(case.payload for case in load_corpus() if case.id == case_id)
    manager = UndoManager(repo_root=tmp_path)
    _write(tmp_path, "payload.txt", payload)
    manager.record(
        UndoEntry(
            turn_id=1, task_id="t", path="payload.txt", before=None, after=payload
        )
    )
    _write(tmp_path, "payload.txt", "replaced")
    manager.record(
        UndoEntry(
            turn_id=1,
            task_id="t",
            path="payload.txt",
            before=payload,
            after="replaced",
        )
    )

    manager.revert_last()

    assert (tmp_path / "payload.txt").read_bytes() == payload.encode("utf-8")


@pytest.mark.regression
def test_journal_entry_wire_format_matches_golden_snapshot(tmp_path: Path) -> None:
    """One canonical `UndoEntry`, recorded and read back as raw bytes,
    matches a pinned snapshot byte-for-byte -- the journal's line
    format is a durable contract once anything else starts reading it
    (a `/undo` command, a future session-log reader), not an
    implementation detail free to drift."""
    manager = UndoManager(repo_root=tmp_path)

    manager.record(
        UndoEntry(
            turn_id=7,
            task_id="task-42",
            path="src/example.py",
            before="def old():\n    pass\n",
            after="def new():\n    return 1\n",
        )
    )

    assert manager.journal_path.read_bytes() == _GOLDEN_FILE.read_bytes()


def test_path_validation_rejects_unsafe_paths(tmp_path: Path) -> None:
    """Validate that directory traversal, absolute paths, and escaping symlinks raise ValueError."""
    manager = UndoManager(repo_root=tmp_path)

    # 1. Absolute paths
    with pytest.raises(ValueError, match="absolute path not allowed"):
        manager._current_content("/etc/passwd")

    # 2. Directory traversal escaping repo root
    with pytest.raises(ValueError, match="escapes the repository root"):
        manager._current_content("../outside.txt")

    # 3. Restoring an absolute path entry raises ValueError
    entry_abs = UndoEntry(1, "t", "/etc/passwd", None, "after")
    with pytest.raises(ValueError, match="absolute path not allowed"):
        manager._restore(entry_abs)

    # 4. Restoring an escaping path entry raises ValueError
    entry_escape = UndoEntry(1, "t", "../outside.txt", None, "after")
    with pytest.raises(ValueError, match="escapes the repository root"):
        manager._restore(entry_escape)


def test_path_validation_symlink_escape(tmp_path: Path) -> None:
    """Verify that a symlink pointing outside the repo root is rejected."""
    manager = UndoManager(repo_root=tmp_path)
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret")

    inside_symlink = tmp_path / "unsafe_link"
    try:
        inside_symlink.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlinks are not supported on this platform/configuration")

    # Reverting/reading the symlink should fail because its resolved path escapes the repo root
    with pytest.raises(ValueError, match="escapes the repository root"):
        manager._current_content("unsafe_link")


def test_load_existing_entries_tolerates_malformed_trailing_line(
    tmp_path: Path,
) -> None:
    """Verify that a malformed trailing journal line is tolerated and skipped,
    while valid entries before it are successfully loaded."""
    journal = tmp_path / "undo.jsonl"
    entry1 = UndoEntry(1, "t", "a.txt", None, "content")

    # Write one valid entry and one trailing malformed entry
    journal.write_bytes(
        (
            f"{json.dumps({'turn_id': 1, 'task_id': 't', 'path': 'a.txt', 'before': None, 'after': 'content'})}\n"
            '{"turn_id": 2, "task_id": "t", "path": "b.txt", "before": "content"\n'  # incomplete/malformed
        ).encode("utf-8")
    )

    manager = UndoManager(repo_root=tmp_path, journal_path=journal)
    assert len(manager._entries) == 1
    assert manager._entries[0] == entry1


def test_load_existing_entries_raises_on_malformed_middle_line(tmp_path: Path) -> None:
    """Verify that a malformed line in the middle of the journal raises an exception."""
    journal = tmp_path / "undo.jsonl"

    # Write a valid entry, a malformed entry, and another valid entry
    journal.write_bytes(
        (
            f"{json.dumps({'turn_id': 1, 'task_id': 't', 'path': 'a.txt', 'before': None, 'after': 'content'})}\n"
            '{"turn_id": 2, "task_id": "t", "path": "b.txt", "before": "content"\n'  # incomplete/malformed
            f"{json.dumps({'turn_id': 3, 'task_id': 't', 'path': 'c.txt', 'before': None, 'after': 'content'})}\n"
        ).encode("utf-8")
    )

    with pytest.raises((json.JSONDecodeError, KeyError, TypeError, ValueError)):
        UndoManager(repo_root=tmp_path, journal_path=journal)
