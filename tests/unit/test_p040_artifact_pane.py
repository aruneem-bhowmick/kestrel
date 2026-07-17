"""Unit tests for `kestrel.tui.app.ArtifactPane.show_report` and
`kestrel.tui.observer_bridge.TuiLoopObserver.on_verification`: proves
`show_report` renders exactly `sanitize_terminal(render_verification_markdown(report))`
for both a passing and a failing `VerificationReport`, and that
`on_verification` forwards a report to a wired artifact pane.

`Markdown.update` reads `self.app` (for the active theme), which an
unmounted widget does not have, so these tests replace `ArtifactPane.
update` with a spy the same way `test_p039_tool_log_diff.py` replaces
`RichLog.write`/`Static.update` -- keeping this suite independent of
Textual's own mounted-app requirement.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.managers.mode import ModeManager
from kestrel.managers.undo import UndoManager
from kestrel.repl import sanitize_terminal
from kestrel.tools.verify import (
    VerificationCommandResult,
    VerificationReport,
    render_verification_markdown,
)
from kestrel.tui.app import ArtifactPane
from kestrel.tui.observer_bridge import TuiLoopObserver

pytestmark = [pytest.mark.p040, pytest.mark.unit, pytest.mark.sanity]

_TASK_ID = "task-p040"
_TURN_ID = 1


def _spy_update(pane: ArtifactPane, updates: list[str]) -> None:
    """Replace `pane.update` with a spy recording each call's own first
    (markdown) argument verbatim, instead of the real `Markdown.update`
    -- keeps these tests independent of Textual's own app-context
    requirement for an unmounted widget."""
    pane.update = updates.append  # type: ignore[method-assign]


def _passing_report() -> VerificationReport:
    """A report whose single `lint` command passed cleanly."""
    return VerificationReport(
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        commands=(
            VerificationCommandResult(
                name="lint",
                command="ruff check",
                exit_code=0,
                timed_out=False,
                stdout="all clean\n",
                stderr="",
            ),
        ),
        passed=True,
    )


def _failing_report() -> VerificationReport:
    """A report whose single `test` command failed."""
    return VerificationReport(
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        commands=(
            VerificationCommandResult(
                name="test",
                command="pytest -q",
                exit_code=1,
                timed_out=False,
                stdout="",
                stderr="1 failed\n",
            ),
        ),
        passed=False,
    )


def test_show_report_renders_a_passing_report_via_markdown_update() -> None:
    """Given a passing `VerificationReport`, when `show_report` renders
    it, then `Markdown.update` is called exactly once with
    `sanitize_terminal(render_verification_markdown(report))`."""
    pane = ArtifactPane()
    updates: list[str] = []
    _spy_update(pane, updates)
    report = _passing_report()

    pane.show_report(report)

    assert updates == [sanitize_terminal(render_verification_markdown(report))]


def test_show_report_renders_a_failing_report_via_markdown_update() -> None:
    """Given a failing `VerificationReport`, when `show_report` renders
    it, then `Markdown.update` is called exactly once with
    `sanitize_terminal(render_verification_markdown(report))`."""
    pane = ArtifactPane()
    updates: list[str] = []
    _spy_update(pane, updates)
    report = _failing_report()

    pane.show_report(report)

    assert updates == [sanitize_terminal(render_verification_markdown(report))]


class _FakeArtifactPane:
    """A minimal stand-in for `ArtifactPane`: records every report it
    was shown, in order, instead of rendering anything."""

    def __init__(self) -> None:
        self.shown: list[VerificationReport] = []

    def show_report(self, report: VerificationReport) -> None:
        """Record `report` verbatim."""
        self.shown.append(report)


class _FakeConversation:
    """A minimal stand-in for `ConversationPane` -- unused by
    `on_verification`, but required to construct a `TuiLoopObserver`."""

    def append_delta(self, text: str) -> None:
        """Do nothing; `on_verification` never calls this."""

    def flush_pending_line(self) -> None:
        """Do nothing; `on_verification` never calls this."""

    def write(self, text: str) -> None:
        """Do nothing; `on_verification` never calls this."""


class _FakeStatusBar:
    """A minimal stand-in for `StatusBar` -- unused by
    `on_verification`, but required to construct a `TuiLoopObserver`."""

    def show(self, snapshot: object) -> None:
        """Do nothing; `on_verification` never calls this."""


def test_on_verification_forwards_the_report_to_a_wired_artifact_pane(
    tmp_path: Path,
) -> None:
    """Given an observer built with an `artifact_pane` collaborator,
    when `on_verification` fires, then that pane's own `show_report` is
    called with the exact report given, exactly once."""
    artifact_pane = _FakeArtifactPane()
    observer = TuiLoopObserver(
        conversation=_FakeConversation(),  # type: ignore[arg-type]
        status_bar=_FakeStatusBar(),  # type: ignore[arg-type]
        undo=UndoManager(repo_root=tmp_path),
        model_id="glm-5.2",
        mode_manager=ModeManager(),
        context_window=200_000,
        session_cap_usd=None,
        day_cap_usd=None,
        spent_day_usd_baseline=Decimal(0),
        artifact_pane=artifact_pane,  # type: ignore[arg-type]
    )
    report = _passing_report()

    result = observer.on_verification(report)

    assert result is None
    assert artifact_pane.shown == [report]
