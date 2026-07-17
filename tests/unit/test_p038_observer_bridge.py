"""Unit tests for `kestrel.tui.observer_bridge.TuiLoopObserver`: every
one of `LoopObserver`'s seven hooks is callable against plain stand-in
panes -- no live Textual app needed -- and each drives the collaborator
its own docstring promises: the status bar refreshing on a turn
boundary, the conversation pane growing (sanitized) on a text delta,
running spend accumulating across turns, and a terse termination
summary landing once the task ends.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.cost.meter import TurnCost
from kestrel.managers.mode import ModeManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport
from kestrel.tui.observer_bridge import TuiLoopObserver
from kestrel.tui.status import StatusSnapshot

pytestmark = [pytest.mark.p038, pytest.mark.unit, pytest.mark.sanity]


class _FakeConversation:
    """A minimal stand-in for `ConversationPane`: records every call
    instead of rendering anything, so a test can assert on exactly what
    the bridge sent it."""

    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.written: list[str] = []
        self.flush_calls = 0

    def append_delta(self, text: str) -> None:
        """Record `text` verbatim."""
        self.deltas.append(text)

    def flush_pending_line(self) -> None:
        """Record that a flush happened."""
        self.flush_calls += 1

    def write(self, text: str) -> None:
        """Record `text` verbatim."""
        self.written.append(text)


class _FakeStatusBar:
    """A minimal stand-in for `StatusBar`: records every snapshot it
    was shown, in order."""

    def __init__(self) -> None:
        self.snapshots: list[StatusSnapshot] = []

    def show(self, snapshot: StatusSnapshot) -> None:
        """Record `snapshot` verbatim."""
        self.snapshots.append(snapshot)


def _observer(
    *,
    conversation: _FakeConversation,
    status_bar: _FakeStatusBar,
    tmp_path: Path,
    session_cap_usd: Decimal | None = None,
    day_cap_usd: Decimal | None = None,
    spent_day_usd_baseline: Decimal = Decimal(0),
) -> TuiLoopObserver:
    """Build a `TuiLoopObserver` against `conversation`/`status_bar`
    stand-ins and otherwise representative collaborators."""
    return TuiLoopObserver(
        conversation=conversation,  # type: ignore[arg-type]
        status_bar=status_bar,  # type: ignore[arg-type]
        undo=UndoManager(repo_root=tmp_path),
        model_id="glm-5.2",
        mode_manager=ModeManager(),
        context_window=200_000,
        session_cap_usd=session_cap_usd,
        day_cap_usd=day_cap_usd,
        spent_day_usd_baseline=spent_day_usd_baseline,
    )


def test_on_turn_started_refreshes_status_bar_with_unbilled_context(
    tmp_path: Path,
) -> None:
    """Given a fresh observer, when `on_turn_started` fires, then
    `status_bar` is shown a snapshot naming the active model, with
    `context_used_tokens=None` and `session_usd=Decimal(0)` -- nothing
    has billed yet."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation, status_bar=status_bar, tmp_path=tmp_path
    )

    observer.on_turn_started(turn_id=1, active_model_id="glm-5.2")

    assert len(status_bar.snapshots) == 1
    snapshot = status_bar.snapshots[0]
    assert snapshot.model_id == "glm-5.2"
    assert snapshot.context_used_tokens is None
    assert snapshot.session_usd == Decimal(0)


def test_on_text_delta_appends_sanitized_text(tmp_path: Path) -> None:
    """Given a delta carrying a control byte, when `on_text_delta`
    fires, then `conversation.append_delta` receives it already
    sanitized, never the raw byte."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation, status_bar=status_bar, tmp_path=tmp_path
    )

    observer.on_text_delta("hello \x1b[2Jworld")

    assert conversation.deltas == ["hello world"]


def test_on_turn_finished_accumulates_cost_across_turns(tmp_path: Path) -> None:
    """Given two successive turns, when each finishes, then the
    bridge's own running session total accumulates rather than resets,
    and each turn's own billed input tokens become the current
    context-usage figure."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation, status_bar=status_bar, tmp_path=tmp_path
    )

    first = TurnCost(
        model_id="glm-5.2",
        input_tokens=100,
        output_tokens=20,
        cached_tokens=0,
        usd=Decimal("0.001"),
    )
    observer.on_turn_finished(turn_id=1, turn_cost=first, active_model_id="glm-5.2")

    assert len(status_bar.snapshots) == 1
    assert status_bar.snapshots[0].context_used_tokens == 100
    assert status_bar.snapshots[0].session_usd == Decimal("0.001")
    assert status_bar.snapshots[0].day_usd == Decimal("0.001")

    second = TurnCost(
        model_id="glm-5.2",
        input_tokens=50,
        output_tokens=10,
        cached_tokens=0,
        usd=Decimal("0.002"),
    )
    observer.on_turn_finished(turn_id=2, turn_cost=second, active_model_id="glm-5.2")

    assert status_bar.snapshots[-1].context_used_tokens == 50
    assert status_bar.snapshots[-1].session_usd == Decimal("0.003")


def test_on_turn_finished_adds_session_spend_on_top_of_the_day_baseline(
    tmp_path: Path,
) -> None:
    """Given a nonzero `spent_day_usd_baseline`, when a turn finishes,
    then the shown snapshot's `day_usd` is the baseline plus this
    task's own running session spend, not the session spend alone."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation,
        status_bar=status_bar,
        tmp_path=tmp_path,
        spent_day_usd_baseline=Decimal("2.5000"),
    )

    turn_cost = TurnCost(
        model_id="glm-5.2",
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        usd=Decimal("0.0010"),
    )
    observer.on_turn_finished(turn_id=1, turn_cost=turn_cost, active_model_id="glm-5.2")

    assert status_bar.snapshots[0].day_usd == Decimal("2.5010")


def test_on_termination_flushes_pending_text_then_writes_a_summary(
    tmp_path: Path,
) -> None:
    """Given a task ends, when `on_termination` fires, then
    `conversation.flush_pending_line` is called exactly once (so a
    trailing line with no closing newline is never dropped) before one
    summary line naming the termination reason, turn count, and total
    cost is written."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation, status_bar=status_bar, tmp_path=tmp_path
    )

    result = LoopResult(
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=3,
        total_usd=Decimal("0.0050"),
        history=(),
    )
    observer.on_termination(result)

    assert conversation.flush_calls == 1
    assert len(conversation.written) == 1
    assert "TASK_COMPLETE" in conversation.written[0]
    assert "3 turn(s)" in conversation.written[0]
    assert "$0.0050" in conversation.written[0]


def test_tool_call_hooks_and_verification_touch_neither_conversation_nor_status_bar(
    tmp_path: Path,
) -> None:
    """Given `on_tool_call_started`/`on_tool_call_finished` -- which
    drive `tool_log`/`diff_pane`, not tested here -- and `on_verification`
    -- still a no-op, pending a later change wiring an artifact pane to
    it -- when each is called with a representative argument, then each
    returns `None` without raising and without touching `conversation`
    or `status_bar`, the two collaborators this file's own suite
    covers."""
    conversation = _FakeConversation()
    status_bar = _FakeStatusBar()
    observer = _observer(
        conversation=conversation, status_bar=status_bar, tmp_path=tmp_path
    )

    call = ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="ok")
    report = VerificationReport(task_id="t-1", turn_id=1, commands=(), passed=True)

    assert observer.on_tool_call_started(call) is None
    assert observer.on_tool_call_finished(call, result) is None
    assert observer.on_verification(report) is None
    assert conversation.deltas == []
    assert conversation.written == []
    assert status_bar.snapshots == []
