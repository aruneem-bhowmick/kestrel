"""Tests for `kestrel.agent.walkthrough`: folding a finished task's own
`LoopResult`, its undo journal, and its verification history into one
`Walkthrough`, rendering that back to markdown, and persisting it under
`.kestrel/artifacts/`.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.agent.walkthrough import (
    Walkthrough,
    WalkthroughError,
    build_walkthrough,
    persist_walkthrough,
    render_walkthrough_markdown,
)
from kestrel.managers.undo import UndoEntry, UndoManager
from kestrel.tools.verify import VerificationCommandResult, VerificationReport

pytestmark = [pytest.mark.p053, pytest.mark.unit]

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
_WITH_VERIFICATION_GOLDEN = _GOLDEN_DIR / "p053_walkthrough_with_verification.golden"
_NO_VERIFICATION_GOLDEN = _GOLDEN_DIR / "p053_walkthrough_no_verification.golden"


def _loop_result(
    *,
    reason: TerminationReason = TerminationReason.TASK_COMPLETE,
    turns_used: int = 3,
    total_usd: Decimal = Decimal("0.0456"),
) -> LoopResult:
    """A minimal `LoopResult` carrying only the fields `build_walkthrough`
    reads -- its `history` is irrelevant here, unlike `kestrel.agent.plan`."""
    return LoopResult(
        reason=reason,
        turns_used=turns_used,
        total_usd=total_usd,
        history=(),
    )


def _verification_report(
    *, task_id: str, turn_id: int, passed: bool = True
) -> VerificationReport:
    """A minimal one-command `VerificationReport` for a given task/turn."""
    return VerificationReport(
        task_id=task_id,
        turn_id=turn_id,
        commands=(
            VerificationCommandResult(
                name="test",
                command="pytest -q",
                exit_code=0 if passed else 1,
                timed_out=False,
                stdout="5 passed in 0.42s" if passed else "1 failed",
                stderr="",
            ),
        ),
        passed=passed,
    )


# -- build_walkthrough ----------------------------------------------------


def test_touched_paths_include_only_entries_for_this_task_sorted_and_deduped(
    tmp_path: Path,
) -> None:
    """Given undo entries spanning two tasks and a repeated path within
    the target task, when built, then `touched_paths` names only the
    target task's own distinct paths, sorted, with the repeat collapsed
    to one entry."""
    undo = UndoManager(repo_root=tmp_path)
    undo.record(
        UndoEntry(turn_id=1, task_id="task-1", path="b.py", before=None, after="b")
    )
    undo.record(
        UndoEntry(turn_id=2, task_id="task-1", path="a.py", before=None, after="a")
    )
    undo.record(
        UndoEntry(turn_id=3, task_id="task-1", path="b.py", before="b", after="b2")
    )
    undo.record(
        UndoEntry(turn_id=1, task_id="task-2", path="c.py", before=None, after="c")
    )

    walkthrough = build_walkthrough(
        _loop_result(), task_id="task-1", undo=undo, verification_reports=()
    )

    assert walkthrough.touched_paths == ("a.py", "b.py")


def test_no_verification_reports_yields_none_verification(tmp_path: Path) -> None:
    """Given an empty `verification_reports` sequence, when built, then
    `Walkthrough.verification` is `None`."""
    undo = UndoManager(repo_root=tmp_path)

    walkthrough = build_walkthrough(
        _loop_result(), task_id="task-1", undo=undo, verification_reports=()
    )

    assert walkthrough.verification is None


def test_the_last_of_several_verification_reports_is_kept(tmp_path: Path) -> None:
    """Given two verification reports for the same task, when built, then
    `Walkthrough.verification` is the *last* one, not the first."""
    undo = UndoManager(repo_root=tmp_path)
    first = _verification_report(task_id="task-1", turn_id=1, passed=False)
    second = _verification_report(task_id="task-1", turn_id=2, passed=True)

    walkthrough = build_walkthrough(
        _loop_result(),
        task_id="task-1",
        undo=undo,
        verification_reports=(first, second),
    )

    assert walkthrough.verification is second


def test_build_walkthrough_carries_the_loop_results_own_fields_verbatim(
    tmp_path: Path,
) -> None:
    """Given a `LoopResult`, when built, then the returned `Walkthrough`
    carries its `reason`, `turns_used`, and `total_usd` unchanged,
    alongside the given `task_id`."""
    undo = UndoManager(repo_root=tmp_path)
    result = _loop_result(
        reason=TerminationReason.WALL_CLOCK_CAP,
        turns_used=7,
        total_usd=Decimal("1.2300"),
    )

    walkthrough = build_walkthrough(
        result, task_id="task-9", undo=undo, verification_reports=()
    )

    assert walkthrough.task_id == "task-9"
    assert walkthrough.reason is TerminationReason.WALL_CLOCK_CAP
    assert walkthrough.turns_used == 7
    assert walkthrough.total_usd == Decimal("1.2300")


# -- render_walkthrough_markdown -------------------------------------------


@pytest.mark.sanity
def test_render_walkthrough_markdown_with_verification_matches_golden_snapshot() -> (
    None
):
    """The exact byte output of one canonical walkthrough carrying a
    verification report must match a pinned snapshot."""
    walkthrough = Walkthrough(
        task_id="demo-task",
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=3,
        total_usd=Decimal("0.0456"),
        touched_paths=(
            "src/kestrel/agent/walkthrough.py",
            "tests/unit/test_p053_walkthrough.py",
        ),
        verification=_verification_report(task_id="demo-task", turn_id=3, passed=True),
    )

    rendered = render_walkthrough_markdown(walkthrough)

    assert rendered == _WITH_VERIFICATION_GOLDEN.read_text(encoding="utf-8")


@pytest.mark.sanity
def test_render_walkthrough_markdown_without_verification_matches_golden_snapshot() -> (
    None
):
    """The exact byte output of one canonical walkthrough with no files
    touched and no verification run must match a pinned snapshot."""
    walkthrough = Walkthrough(
        task_id="demo-task-2",
        reason=TerminationReason.USER_STOP,
        turns_used=1,
        total_usd=Decimal("0"),
        touched_paths=(),
        verification=None,
    )

    rendered = render_walkthrough_markdown(walkthrough)

    assert rendered == _NO_VERIFICATION_GOLDEN.read_text(encoding="utf-8")


# -- persist_walkthrough ----------------------------------------------------


def _demo_walkthrough(task_id: str = "task-persist") -> Walkthrough:
    """A minimal `Walkthrough` with no files touched and no verification
    run, for persistence tests that don't care about rendered content."""
    return Walkthrough(
        task_id=task_id,
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0"),
        touched_paths=(),
        verification=None,
    )


def test_persist_walkthrough_writes_the_expected_path_and_content(
    tmp_path: Path,
) -> None:
    """Given a walkthrough, when persisted, then the exact rendered
    markdown is written to `.kestrel/artifacts/walkthrough-<task_id>.md`."""
    walkthrough = _demo_walkthrough()

    written = persist_walkthrough(walkthrough, repo_root=tmp_path)

    expected_path = tmp_path / ".kestrel" / "artifacts" / "walkthrough-task-persist.md"
    assert written == expected_path
    assert written.read_text(encoding="utf-8") == render_walkthrough_markdown(
        walkthrough
    )


def test_persist_walkthrough_a_second_call_gets_the_numeric_suffix_path(
    tmp_path: Path,
) -> None:
    """Given two walkthroughs persisted for the same task id, when the
    second is written, then it lands at the `-1`-suffixed path rather
    than overwriting the first."""
    walkthrough = _demo_walkthrough()

    first_path = persist_walkthrough(walkthrough, repo_root=tmp_path)
    second_path = persist_walkthrough(walkthrough, repo_root=tmp_path)

    assert first_path != second_path
    assert second_path.name == "walkthrough-task-persist-1.md"
    assert first_path.is_file()
    assert second_path.is_file()


def test_persist_walkthrough_refuses_a_symlinked_artifacts_dir_naming_the_remedy(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Given `.kestrel` replaced with a symlink pointing outside the repo
    root, when a walkthrough is persisted, then `WalkthroughError` names
    the escape and nothing is written through the symlink."""
    outside = tmp_path_factory.mktemp("walkthrough-outside-target")
    (tmp_path / ".kestrel").symlink_to(outside, target_is_directory=True)
    walkthrough = _demo_walkthrough()

    with pytest.raises(WalkthroughError, match="escapes the repository root"):
        persist_walkthrough(walkthrough, repo_root=tmp_path)

    assert list(outside.iterdir()) == []
