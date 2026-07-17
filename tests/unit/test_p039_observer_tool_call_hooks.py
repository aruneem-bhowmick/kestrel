"""Unit tests for `kestrel.tui.observer_bridge.TuiLoopObserver`'s
tool-call hooks: `on_tool_call_started`/`on_tool_call_finished` now
track an in-flight count (notifying `on_inflight_change` of it),
forward started/finished lines to a wired `tool_log`, and -- for an
`edit_file` call that actually recorded a new `UndoEntry` -- render
that mutation's own before/after through a wired `diff_pane`. Every
case here is driven against plain stand-in collaborators, matching
`test_p038_observer_bridge.py`'s own no-live-Textual-app approach.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.managers.mode import ModeManager
from kestrel.managers.undo import UndoEntry, UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.registry import ToolResult
from kestrel.tui import observer_bridge as observer_bridge_module
from kestrel.tui.observer_bridge import TuiLoopObserver

pytestmark = [pytest.mark.p039, pytest.mark.unit, pytest.mark.sanity]


class _FakeConversation:
    """A minimal stand-in for `ConversationPane`: records nothing these
    tests care about, but must exist for `TuiLoopObserver`'s own
    required constructor argument."""

    def append_delta(self, text: str) -> None:
        """Ignore."""

    def flush_pending_line(self) -> None:
        """Ignore."""

    def write(self, text: str) -> None:
        """Ignore."""


class _FakeStatusBar:
    """A minimal stand-in for `StatusBar`, likewise required but not
    asserted on by these tests."""

    def show(self, snapshot: object) -> None:
        """Ignore."""


class _FakeToolLog:
    """Records every `append_started`/`append_finished` call verbatim,
    in order, instead of rendering anything."""

    def __init__(self) -> None:
        """Start both recorded-call lists empty."""
        self.started: list[ToolCallEvent] = []
        self.finished: list[tuple[ToolCallEvent, float]] = []

    def append_started(self, call: ToolCallEvent) -> None:
        """Record `call`."""
        self.started.append(call)

    def append_finished(self, call: ToolCallEvent, *, elapsed_s: float) -> None:
        """Record `(call, elapsed_s)`."""
        self.finished.append((call, elapsed_s))


class _FakeDiffPane:
    """Records every `show_diff` call's own arguments verbatim."""

    def __init__(self) -> None:
        """Start the recorded-call list empty."""
        self.calls: list[tuple[str, str | None, str | None]] = []

    def show_diff(self, path: str, before: str | None, after: str | None) -> None:
        """Record `(path, before, after)`."""
        self.calls.append((path, before, after))


def _observer(
    *,
    tmp_path: Path,
    on_inflight_change: object = None,
    tool_log: _FakeToolLog | None = None,
    diff_pane: _FakeDiffPane | None = None,
) -> TuiLoopObserver:
    """Build a `TuiLoopObserver` against fake conversation/status-bar
    collaborators and a real `UndoManager`, wiring in only the
    tool-call-hook collaborators each test actually cares about."""
    kwargs: dict[str, object] = {
        "conversation": _FakeConversation(),
        "status_bar": _FakeStatusBar(),
        "undo": UndoManager(repo_root=tmp_path),
        "model_id": "glm-5.2",
        "mode_manager": ModeManager(),
        "context_window": 200_000,
        "session_cap_usd": None,
        "day_cap_usd": None,
        "spent_day_usd_baseline": Decimal(0),
        "tool_log": tool_log,
        "diff_pane": diff_pane,
    }
    if on_inflight_change is not None:
        kwargs["on_inflight_change"] = on_inflight_change
    return TuiLoopObserver(**kwargs)  # type: ignore[arg-type]


def test_on_tool_call_started_increments_inflight_and_notifies(tmp_path: Path) -> None:
    """Given a fresh observer, when `on_tool_call_started` fires, then
    `on_inflight_change` is called with `1` and, when `tool_log` is
    wired, the call is forwarded to it."""
    inflight_changes: list[int] = []
    tool_log = _FakeToolLog()
    observer = _observer(
        tmp_path=tmp_path, on_inflight_change=inflight_changes.append, tool_log=tool_log
    )
    call = ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")

    observer.on_tool_call_started(call)

    assert inflight_changes == [1]
    assert tool_log.started == [call]


def test_on_tool_call_finished_decrements_inflight_and_reports_elapsed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a call that started and then finished 1.5 (fake) seconds
    later, when `on_tool_call_finished` fires, then `on_inflight_change`
    is called with `0` and `tool_log.append_finished` receives that
    exact elapsed time."""
    times = iter([100.0, 101.5])
    monkeypatch.setattr(observer_bridge_module.time, "monotonic", lambda: next(times))
    inflight_changes: list[int] = []
    tool_log = _FakeToolLog()
    observer = _observer(
        tmp_path=tmp_path, on_inflight_change=inflight_changes.append, tool_log=tool_log
    )
    call = ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="ok")

    observer.on_tool_call_started(call)
    observer.on_tool_call_finished(call, result)

    assert inflight_changes == [1, 0]
    assert tool_log.finished == [(call, 1.5)]


def test_on_tool_call_hooks_are_safe_with_no_wired_collaborators(
    tmp_path: Path,
) -> None:
    """Given an observer built with every optional collaborator left at
    its default (including `on_inflight_change`), when both tool-call
    hooks fire, then neither raises."""
    observer = _observer(tmp_path=tmp_path)
    call = ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="ok")

    observer.on_tool_call_started(call)
    observer.on_tool_call_finished(call, result)


def test_on_tool_call_finished_shows_the_diff_for_a_recorded_edit_file_mutation(
    tmp_path: Path,
) -> None:
    """Given an `edit_file` call that -- between its own started and
    finished hooks -- recorded a real `UndoEntry`, when
    `on_tool_call_finished` fires, then `diff_pane.show_diff` is called
    with that entry's own path, before, and after content."""
    diff_pane = _FakeDiffPane()
    undo = UndoManager(repo_root=tmp_path)
    observer = TuiLoopObserver(
        conversation=_FakeConversation(),  # type: ignore[arg-type]
        status_bar=_FakeStatusBar(),  # type: ignore[arg-type]
        undo=undo,
        model_id="glm-5.2",
        mode_manager=ModeManager(),
        context_window=200_000,
        session_cap_usd=None,
        day_cap_usd=None,
        spent_day_usd_baseline=Decimal(0),
        diff_pane=diff_pane,  # type: ignore[arg-type]
    )
    call = ToolCallEvent(id="call-1", name="edit_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="ok")

    observer.on_tool_call_started(call)
    undo.record(
        UndoEntry(turn_id=1, task_id="t-1", path="greet.py", before="old", after="new")
    )
    observer.on_tool_call_finished(call, result)

    assert diff_pane.calls == [("greet.py", "old", "new")]


def test_on_tool_call_finished_skips_the_diff_for_a_non_edit_file_call(
    tmp_path: Path,
) -> None:
    """Given a non-`edit_file` call (which never mutates the undo
    journal in practice) finishing while `diff_pane` is wired, when
    `on_tool_call_finished` fires, then `diff_pane.show_diff` is never
    called, regardless of the call's own name."""
    diff_pane = _FakeDiffPane()
    observer = _observer(tmp_path=tmp_path, diff_pane=diff_pane)
    call = ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="ok")

    observer.on_tool_call_started(call)
    observer.on_tool_call_finished(call, result)

    assert diff_pane.calls == []


def test_on_tool_call_finished_skips_the_diff_for_a_no_op_edit_file_call(
    tmp_path: Path,
) -> None:
    """Given an `edit_file` call that finishes without having recorded
    any new `UndoEntry` (e.g. one refused for an ambiguous anchor),
    when `on_tool_call_finished` fires, then `diff_pane.show_diff` is
    never called -- there is no real mutation to render."""
    diff_pane = _FakeDiffPane()
    observer = _observer(tmp_path=tmp_path, diff_pane=diff_pane)
    call = ToolCallEvent(id="call-1", name="edit_file", arguments_json="{}")
    result = ToolResult(tool_call_id="call-1", content="refused")

    observer.on_tool_call_started(call)
    observer.on_tool_call_finished(call, result)

    assert diff_pane.calls == []


def test_on_tool_call_finished_uses_each_calls_own_undo_baseline_when_calls_overlap(
    tmp_path: Path,
) -> None:
    """Given two `edit_file` calls whose hooks interleave -- `call_1`
    starts and records a mutation, then `call_2` starts before `call_1`
    finishes -- when each finishes, then `call_1`'s diff still renders
    against its own start-time baseline rather than `call_2`'s later
    one, and `call_2` -- which recorded no mutation of its own --
    renders no diff. A single shared baseline (rather than one keyed
    per call id) would have `call_2`'s own start silently erase
    `call_1`'s, hiding a real mutation."""
    diff_pane = _FakeDiffPane()
    undo = UndoManager(repo_root=tmp_path)
    observer = TuiLoopObserver(
        conversation=_FakeConversation(),  # type: ignore[arg-type]
        status_bar=_FakeStatusBar(),  # type: ignore[arg-type]
        undo=undo,
        model_id="glm-5.2",
        mode_manager=ModeManager(),
        context_window=200_000,
        session_cap_usd=None,
        day_cap_usd=None,
        spent_day_usd_baseline=Decimal(0),
        diff_pane=diff_pane,  # type: ignore[arg-type]
    )
    call_1 = ToolCallEvent(id="call-1", name="edit_file", arguments_json="{}")
    call_2 = ToolCallEvent(id="call-2", name="edit_file", arguments_json="{}")
    result_1 = ToolResult(tool_call_id="call-1", content="ok")
    result_2 = ToolResult(tool_call_id="call-2", content="ok")

    observer.on_tool_call_started(call_1)
    undo.record(
        UndoEntry(turn_id=1, task_id="t-1", path="a.py", before="old-a", after="new-a")
    )
    observer.on_tool_call_started(call_2)
    observer.on_tool_call_finished(call_1, result_1)
    observer.on_tool_call_finished(call_2, result_2)

    assert diff_pane.calls == [("a.py", "old-a", "new-a")]
