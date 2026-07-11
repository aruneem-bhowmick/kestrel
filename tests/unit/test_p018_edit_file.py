"""Tests for the `edit_file` tool: unique-anchor replacement, the
missing/non-unique anchor refusals, dry-run diffing, undo-journal
recording and reversion, the shared path-containment and binary-content
guards, write-then-record ordering, and argument parsing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from kestrel.managers.undo import UndoManager
from kestrel.tools._paths import resolve_repo_path as _real_resolve_repo_path
from kestrel.tools.edit_file import EditFileArgs, EditFileError, edit_file

pytestmark = [pytest.mark.p018, pytest.mark.unit]

# See `test_p016_execute_redteam.py` for why this module is resolved via
# `importlib` rather than `import kestrel.tools.edit_file as edit_file_module`:
# `kestrel.tools.__init__` rebinds the `edit_file` *attribute* on the
# `kestrel.tools` package to the function of the same name, so a plain
# attribute lookup after the package finishes importing would not reach
# the submodule's own `resolve_repo_path` reference.
_edit_file_module = importlib.import_module("kestrel.tools.edit_file")

_TURN_ID = 1
_TASK_ID = "task-edit"


def _write(root: Path, relative: str, content: str) -> None:
    """Write `content` as UTF-8 bytes to `relative` under `root`,
    creating parent directories as needed. Writing bytes directly
    (rather than through text-mode newline translation) keeps line
    endings exactly as given, regardless of platform."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


def _read(root: Path, relative: str) -> str:
    """Read `relative` under `root` back as UTF-8 text."""
    return (root / relative).read_text(encoding="utf-8")


def _edit(
    root: Path, undo: UndoManager, *, path: str, old: str, new: str, dry_run: bool = False
) -> str:
    """Call `edit_file` with `_TURN_ID`/`_TASK_ID` filled in, so
    individual test bodies only need to name what actually varies."""
    return edit_file(
        EditFileArgs(path=path, old=old, new=new, dry_run=dry_run),
        repo_root=root,
        undo=undo,
        turn_id=_TURN_ID,
        task_id=_TASK_ID,
    )


@pytest.mark.sanity
def test_unique_anchor_is_replaced_and_file_content_matches_exactly(
    tmp_path: Path,
) -> None:
    """Given a file whose content contains one occurrence of an anchor,
    when edited, then the file's new content has the anchor replaced
    exactly, with everything else unchanged."""
    _write(tmp_path, "greet.py", "print('hello world')\n")
    undo = UndoManager(repo_root=tmp_path)

    _edit(tmp_path, undo, path="greet.py", old="world", new="there")

    assert _read(tmp_path, "greet.py") == "print('hello there')\n"


@pytest.mark.sanity
def test_missing_anchor_raises_edit_file_error(tmp_path: Path) -> None:
    """Given a file whose content does not contain the anchor, when
    edited, then `EditFileError` names the anchor as not found and the
    file is left untouched."""
    _write(tmp_path, "greet.py", "print('hello world')\n")
    undo = UndoManager(repo_root=tmp_path)

    with pytest.raises(EditFileError, match="anchor not found"):
        _edit(tmp_path, undo, path="greet.py", old="goodbye", new="hi")

    assert _read(tmp_path, "greet.py") == "print('hello world')\n"


@pytest.mark.sanity
def test_non_unique_anchor_raises_naming_the_count_and_leaves_file_unmodified(
    tmp_path: Path,
) -> None:
    """Given a file whose content contains the anchor twice, when
    edited, then `EditFileError` names the exact occurrence count
    rather than guessing which one was meant, and the file is left
    byte-identical to before the call."""
    _write(tmp_path, "dup.txt", "dup line\ndup line\n")
    undo = UndoManager(repo_root=tmp_path)

    with pytest.raises(EditFileError, match=r"anchor not unique \(2 occurrences\)"):
        _edit(tmp_path, undo, path="dup.txt", old="dup line", new="single line")

    assert _read(tmp_path, "dup.txt") == "dup line\ndup line\n"


@pytest.mark.sanity
def test_dry_run_returns_a_diff_and_leaves_the_file_byte_identical(
    tmp_path: Path,
) -> None:
    """Given a unique anchor and `dry_run=True`, when edited, then the
    result is a framed unified diff naming the change, and the file on
    disk is byte-identical to what it was before the call."""
    original = "print('hello world')\n"
    _write(tmp_path, "greet.py", original)
    undo = UndoManager(repo_root=tmp_path)

    framed = _edit(
        tmp_path, undo, path="greet.py", old="world", new="there", dry_run=True
    )

    assert framed.startswith("<<<UNTRUSTED:tool_stdout:greet.py>>>\n")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert "-print('hello world')" in framed
    assert "+print('hello there')" in framed
    assert (tmp_path / "greet.py").read_bytes() == original.encode("utf-8")


def test_successful_edit_produces_exactly_one_undo_entry_matching_content(
    tmp_path: Path,
) -> None:
    """Given a unique anchor and `dry_run=False`, when edited, then
    exactly one new `UndoManager` journal entry is recorded, and its
    `before`/`after` match the file's real content before and after the
    call."""
    before_content = "print('hello world')\n"
    after_content = "print('hello there')\n"
    _write(tmp_path, "greet.py", before_content)
    undo = UndoManager(repo_root=tmp_path)
    entries_before = len(undo._entries)

    _edit(tmp_path, undo, path="greet.py", old="world", new="there")

    assert len(undo._entries) == entries_before + 1
    entry = undo._entries[-1]
    assert entry.path == "greet.py"
    assert entry.before == before_content
    assert entry.after == after_content
    assert entry.turn_id == _TURN_ID
    assert entry.task_id == _TASK_ID


def test_reverting_the_recorded_entry_restores_the_original_file_exactly(
    tmp_path: Path,
) -> None:
    """Given a successful edit, when the resulting journal entry is
    reverted via `UndoManager.revert_last`, then the file's content is
    restored exactly to what it was before the edit."""
    original = "print('hello world')\n"
    _write(tmp_path, "greet.py", original)
    undo = UndoManager(repo_root=tmp_path)

    _edit(tmp_path, undo, path="greet.py", old="world", new="there")
    assert _read(tmp_path, "greet.py") == "print('hello there')\n"

    undo.revert_last()

    assert _read(tmp_path, "greet.py") == original


def test_path_escaping_repo_root_raises_via_the_shared_resolve_repo_path_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a path that climbs above `repo_root` with `..`, when
    edited, then `EditFileError` names the escape exactly as
    `read_file` would for the same path, and this happens by way of
    the shared `resolve_repo_path` helper -- proved here by a
    monkeypatch spy wrapping the real function -- so this suite cannot
    silently drift from `read_file`'s own containment behavior if that
    helper changes underneath both tools."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    calls: list[str] = []

    def _spy(path: str, *, repo_root: Path) -> Path:
        calls.append(path)
        return _real_resolve_repo_path(path, repo_root=repo_root)

    monkeypatch.setattr(_edit_file_module, "resolve_repo_path", _spy)
    undo = UndoManager(repo_root=repo_root)

    with pytest.raises(EditFileError, match="escapes the repository root"):
        _edit(repo_root, undo, path="../secret.txt", old="x", new="y")

    assert calls == ["../secret.txt"]


def test_binary_content_raises_naming_the_binary_guard(tmp_path: Path) -> None:
    """Given a file whose content begins with a PNG file signature,
    when edited, then `EditFileError` names the binary guard -- the
    same refusal, for the same reason, that `read_file` raises for an
    identical file."""
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    (tmp_path / "image.png").write_bytes(png_header)
    undo = UndoManager(repo_root=tmp_path)

    with pytest.raises(EditFileError, match="binary guard"):
        _edit(tmp_path, undo, path="image.png", old="x", new="y")


def test_missing_file_raises_naming_no_such_file_rather_than_creating_it(
    tmp_path: Path,
) -> None:
    """Given a path with no file on disk, when edited, then
    `EditFileError` names it as missing rather than creating a new file
    from `new` -- `edit_file` never creates files, even implicitly."""
    undo = UndoManager(repo_root=tmp_path)

    with pytest.raises(EditFileError, match="no such file"):
        _edit(tmp_path, undo, path="new-file.txt", old="", new="content")

    assert not (tmp_path / "new-file.txt").exists()


def test_write_then_record_ordering_leaves_the_file_written_when_record_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a successful anchor match, when `UndoManager.record` is
    monkeypatched to raise, then the exception propagates but the
    file's new content is already on disk -- proving `edit_file` writes
    before it records rather than the reverse, so a broken journal
    entry after a real write is a recoverable, half-finished operation
    rather than a silently un-recorded one."""
    _write(tmp_path, "greet.py", "print('hello world')\n")
    undo = UndoManager(repo_root=tmp_path)

    def _raise_on_record(self: UndoManager, entry: object) -> None:
        raise RuntimeError("journal write failed")

    monkeypatch.setattr(UndoManager, "record", _raise_on_record)

    with pytest.raises(RuntimeError, match="journal write failed"):
        _edit(tmp_path, undo, path="greet.py", old="world", new="there")

    assert _read(tmp_path, "greet.py") == "print('hello there')\n"
