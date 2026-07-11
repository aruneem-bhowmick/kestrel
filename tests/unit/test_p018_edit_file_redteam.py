"""Red-team proof that a `dry_run` edit's diff output still comes back
through `edit_file` wrapped by the real frame markers, even when the
file's unchanged surrounding content is one of the injection corpus's
hostile payloads -- a hostile file cannot smuggle itself out of its own
frame just because it is being viewed as a diff rather than as raw
file content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.managers.undo import UndoManager
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.edit_file import EditFileArgs, edit_file

pytestmark = [pytest.mark.p018, pytest.mark.unit, pytest.mark.redteam]

_HOSTILE_CASE_ID = "readme_ignore_previous_instructions"

_ANCHOR = "ANCHOR_LINE\n"
_REPLACEMENT = "REPLACED_LINE\n"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


@pytest.fixture(scope="session")
def hostile_surrounding_content_case() -> InjectionCase:
    """The corpus case used to prove `edit_file`'s dry-run diff framing
    survives a real prompt-injection payload sitting as unchanged
    context around the edited line. Looked up lazily (rather than at
    module import time) so a lookup failure surfaces as a normal test
    error tied to whichever test requested it."""
    return _find_case(_HOSTILE_CASE_ID)


def _write_file_with_hostile_surroundings(
    tmp_path: Path, *, relative: str, case: InjectionCase
) -> None:
    """Write `relative` under `tmp_path` with `case.payload` split
    around a one-line anchor, positioned so a default-context unified
    diff on that anchor includes every line of the payload -- proving
    the whole hostile payload, not just a fragment of it, passes
    through `edit_file`'s dry-run diff still framed."""
    payload_lines = case.payload.splitlines(keepends=True)
    split_at = 2
    content = (
        "".join(payload_lines[:split_at]) + _ANCHOR + "".join(payload_lines[split_at:])
    )
    (tmp_path / relative).write_bytes(content.encode("utf-8"))


def test_dry_run_diff_over_hostile_surrounding_content_still_carries_the_real_frame_markers(
    tmp_path: Path, hostile_surrounding_content_case: InjectionCase
) -> None:
    """Given a file whose unchanged surrounding content is one of the
    injection corpus's hostile payloads, when a `dry_run` edit changes
    an unrelated single line, then the result still starts with the
    real opening header naming this file, still ends with the real
    closing delimiter, and the hostile payload's own text survives
    intact inside the frame as unchanged diff context."""
    _write_file_with_hostile_surroundings(
        tmp_path,
        relative="README.md",
        case=hostile_surrounding_content_case,
    )
    undo = UndoManager(repo_root=tmp_path)

    framed = edit_file(
        EditFileArgs(path="README.md", old=_ANCHOR, new=_REPLACEMENT, dry_run=True),
        repo_root=tmp_path,
        undo=undo,
        turn_id=1,
        task_id="task-redteam",
    )

    assert framed.startswith("<<<UNTRUSTED:tool_stdout:README.md>>>\n")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    for line in hostile_surrounding_content_case.payload.splitlines():
        if line:
            assert line in framed
    for marker in hostile_surrounding_content_case.forbidden_markers:
        assert framed.count(marker) == 1


def test_dry_run_over_hostile_content_never_writes_or_records(
    tmp_path: Path, hostile_surrounding_content_case: InjectionCase
) -> None:
    """Given the same hostile-surroundings file, when the same
    `dry_run` edit runs, then the file on disk is untouched and no
    entry is added to the undo journal -- a dry run over hostile
    content is exactly as inert as a dry run over benign content."""
    _write_file_with_hostile_surroundings(
        tmp_path,
        relative="README.md",
        case=hostile_surrounding_content_case,
    )
    original_bytes = (tmp_path / "README.md").read_bytes()
    undo = UndoManager(repo_root=tmp_path)

    edit_file(
        EditFileArgs(path="README.md", old=_ANCHOR, new=_REPLACEMENT, dry_run=True),
        repo_root=tmp_path,
        undo=undo,
        turn_id=1,
        task_id="task-redteam",
    )

    assert (tmp_path / "README.md").read_bytes() == original_bytes
    assert undo._entries == []
