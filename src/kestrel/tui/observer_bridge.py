"""Bridges a running task's `LoopObserver` callbacks onto the TUI's own
widgets, so submitting a task in the cockpit is visibly alive rather than
a silent wait for one final result.

`TuiLoopObserver` is constructed fresh for every submitted task and
handed to `build_task_deps` as that task's own `observer`. Every method
here is called synchronously, inline, on the same coroutine driving the
task -- exactly the contract `kestrel.agent.observer.LoopObserver`
documents -- so calling widget methods directly is safe: `KestrelApp.
run_worker(self._run_task(text))` schedules the coroutine as a Task on
the app's own asyncio event loop (Textual's default for a coroutine
function, not a separate OS thread), and the loop's observer hooks fire
from that identical loop.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from kestrel.agent.loop import LoopResult
from kestrel.cost.meter import TurnCost
from kestrel.managers.mode import ModeManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.repl import sanitize_terminal
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport
from kestrel.tui.status import StatusSnapshot

if TYPE_CHECKING:
    # Deferred to break the import cycle: `kestrel.tui.app.KestrelApp`
    # constructs a `TuiLoopObserver` per task, so a module-level import
    # here of the widgets it renders to would import `kestrel.tui.app`
    # right back into this module while it is still loading. `from
    # __future__ import annotations` (above) already makes every
    # annotation in this file a lazy string, so these names are only
    # ever needed for static type-checking, never at runtime.
    from kestrel.tui.app import (
        ArtifactPane,
        ConversationPane,
        DiffPane,
        StatusBar,
        ToolLogPane,
    )


class TuiLoopObserver:
    """Bridges `LoopObserver` calls onto the TUI's widgets.

    `tool_log`/`diff_pane`/`artifact_pane` default to `None` -- a
    harmless no-op for any hook they would otherwise drive, the same
    optional-collaborator-with-a-safe-default pattern `LoopDeps.session`/
    `LoopDeps.budget` already establish; a later change passes the real
    widgets once `KestrelApp` constructs one of these per task and wires
    the tool-call and verification hooks below up to them. `undo` is
    threaded through for that same later wiring -- this bridge does not
    yet read it itself.

    Running spend is tracked by this bridge itself (`turn_cost.usd`
    accumulated in `on_turn_finished`) rather than read back off the
    task's own `CostMeter`, so the bridge stays fully independent of
    exactly when, relative to `build_task_deps`, it itself gets built
    or attached to `LoopDeps.observer`.
    """

    def __init__(
        self,
        *,
        conversation: ConversationPane,
        status_bar: StatusBar,
        undo: UndoManager,
        model_id: str,
        mode_manager: ModeManager,
        context_window: int,
        session_cap_usd: Decimal | None,
        day_cap_usd: Decimal | None,
        spent_day_usd_baseline: Decimal,
        tool_log: ToolLogPane | None = None,
        diff_pane: DiffPane | None = None,
        artifact_pane: ArtifactPane | None = None,
    ) -> None:
        """Store every collaborator this task's own hooks will drive,
        and start this bridge's own running totals at zero/unset --
        `_context_used_tokens` stays `None` until the first turn bills,
        matching `StatusSnapshot`'s own "no turn has billed yet"
        convention.
        """
        self._conversation = conversation
        self._status_bar = status_bar
        self._undo = undo
        self._model_id = model_id
        self._mode_manager = mode_manager
        self._context_window = context_window
        self._session_cap_usd = session_cap_usd
        self._day_cap_usd = day_cap_usd
        self._spent_day_usd_baseline = spent_day_usd_baseline
        self._tool_log = tool_log
        self._diff_pane = diff_pane
        self._artifact_pane = artifact_pane
        self._session_usd = Decimal(0)
        self._context_used_tokens: int | None = None

    def _show_status(self, *, active_model_id: str) -> None:
        """Rebuild a `StatusSnapshot` from this bridge's own running
        totals and the current `mode_manager` state, and render it --
        the one status-refresh path every hook below that touches the
        status bar shares."""
        snapshot = StatusSnapshot(
            model_id=active_model_id,
            mode=self._mode_manager.mode,
            effort=self._mode_manager.effort(),
            context_used_tokens=self._context_used_tokens,
            context_window=self._context_window,
            session_usd=self._session_usd,
            session_cap_usd=self._session_cap_usd,
            day_usd=self._spent_day_usd_baseline + self._session_usd,
            day_cap_usd=self._day_cap_usd,
        )
        self._status_bar.show(snapshot)

    def on_turn_started(self, *, turn_id: int, active_model_id: str) -> None:
        """Refresh `status_bar` with `active_model_id` and this turn's
        not-yet-updated `session_usd`/`context_used_tokens` -- whatever
        the last `on_turn_finished` left them at."""
        self._show_status(active_model_id=active_model_id)

    def on_text_delta(self, text: str) -> None:
        """Append `text`, sanitized, to the conversation pane's own
        currently streaming line."""
        self._conversation.append_delta(sanitize_terminal(text))

    def on_tool_call_started(self, call: ToolCallEvent) -> None:
        """No-op -- a later change replaces this body once `tool_log`
        is wired to render live entries."""

    def on_tool_call_finished(self, call: ToolCallEvent, result: ToolResult) -> None:
        """No-op -- a later change replaces this body once `tool_log`
        is wired to render live entries."""

    def on_verification(self, report: VerificationReport) -> None:
        """No-op -- a later change replaces this body once a pane is
        wired to render verification results."""

    def on_turn_finished(
        self, *, turn_id: int, turn_cost: TurnCost, active_model_id: str
    ) -> None:
        """Fold `turn_cost` into this bridge's own running session
        total, record this turn's own billed input tokens as the
        current context-usage figure, and refresh `status_bar`."""
        self._session_usd += turn_cost.usd
        self._context_used_tokens = turn_cost.input_tokens
        self._show_status(active_model_id=active_model_id)

    def on_termination(self, result: LoopResult) -> None:
        """Flush any partial line `on_text_delta` has buffered but not
        yet written, then write a terse one-line summary -- termination
        reason, turn count, total cost -- to `conversation`. Mirrors
        `cli.py`'s own `_print_task_summary`'s content, not its exact
        multi-line CLI format; one line is enough here, since the full
        detail is already visible in the other panes."""
        self._conversation.flush_pending_line()
        self._conversation.write(
            f"\n[{result.reason.name}] {result.turns_used} turn(s) -- "
            f"${result.total_usd:.4f} total"
        )
