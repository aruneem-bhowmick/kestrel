"""Kestrel's Textual cockpit: `KestrelApp` and the panes it composes.

This module lays out and styles the interactive terminal interface --
a conversation view, an artifact viewer, a collapsible tool log, a diff
view, and a status bar -- in a restrained rust-and-slate default theme.
Submitting text in the task-input box drives a real `run_task` call
through `kestrel.task_setup.build_task_deps`: the conversation pane
streams the assistant's own text as it arrives, the tool log gains a
started/finished line for every tool call, the diff pane renders the
most recent `edit_file` mutation as a unified diff, the artifact pane
renders the task's own most recent `VerificationReport` as markdown, a
loading indicator shows for as long as any tool call is in flight, the
status bar updates live after every turn, and the task's own
termination prints a terse summary. Every pane that renders model- or
tool-sourced text routes it through `kestrel.repl.sanitize_terminal`
first, whether directly (the conversation pane) or via
`kestrel.tui.observer_bridge.TuiLoopObserver`. A `ctrl+p` command
palette (`kestrel.tui.commands.KestrelCommandProvider`) gives keyboard
access to model/mode switching, undo, a cost breakdown, and resuming a
prior task, alongside two informational entries covering approval and
knowledge-base status.
"""

from __future__ import annotations

import asyncio
import difflib
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Collapsible,
    Input,
    LoadingIndicator,
    Markdown,
    RichLog,
    Static,
)

from kestrel.agent.loop import resume_task, run_task
from kestrel.config import KestrelConfig
from kestrel.cost.meter import CostMeter, format_cost_line
from kestrel.kestrel_md import KestrelMd
from kestrel.managers.mode import Mode, ModeManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.registry.model import Registry
from kestrel.repl import sanitize_terminal
from kestrel.task_setup import TaskSetup, build_task_deps
from kestrel.tools.verify import VerificationReport, render_verification_markdown
from kestrel.tui.commands import KestrelCommandProvider
from kestrel.tui.observer_bridge import TuiLoopObserver
from kestrel.tui.status import StatusSnapshot, render_status_line

_MAX_TOOL_ARG_SUMMARY_CHARS: Final[int] = 120
_ARTIFACT_PLACEHOLDER: Final[str] = "_no artifact yet_"


class ConversationPane(RichLog):
    """Streams the active task's assistant text and turn/termination
    summaries. Append-only: `append_delta` grows the currently
    streaming line rather than re-rendering the whole log, so a long
    task never pays for a full-screen repaint per token."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward every keyword straight to `RichLog`; start this
        pane's own partial-line buffer empty."""
        super().__init__(**kwargs)
        self._pending_line = ""

    def append_delta(self, text: str) -> None:
        """Append `text` to the currently streaming line without a
        full-log re-render -- never calls `self.clear()` or otherwise
        replaces prior content.

        The installed `RichLog.write` always appends its argument as
        one or more brand-new lines rather than extending whatever was
        last written, so true sub-line incremental writes are not
        available here. Instead, `text` is folded into a small local
        buffer; every complete line the buffer accumulates (a `"\\n"`
        boundary reached) is flushed to the log immediately, and
        whatever remains after the last newline stays buffered until
        either the next delta completes it or `flush_pending_line`
        writes it out unfinished.
        """
        combined = self._pending_line + text
        *complete_lines, self._pending_line = combined.split("\n")
        for line in complete_lines:
            self.write(line)

    def flush_pending_line(self) -> None:
        """Write out whatever partial line `append_delta` has buffered
        but not yet flushed to a newline boundary, then clear the
        buffer -- called once a task's own text stream has definitively
        ended, so trailing content with no closing newline is never
        silently dropped."""
        if self._pending_line:
            self.write(self._pending_line)
            self._pending_line = ""


class ToolLogPane(RichLog):
    """Append-only log of tool calls and their outcomes, mounted inside
    a `Collapsible` by `KestrelApp.compose` -- the collapsible
    container, not this widget itself, is what makes the tool log
    collapsible. Empty until the first tool call of a submitted task
    writes its own started/finished lines here."""

    def append_started(self, call: ToolCallEvent) -> None:
        """Write a `"-> {name}({summary})"` line for a tool call that
        just started.

        `summary` is `call.arguments_json`, sanitized through
        `sanitize_terminal` and capped at `_MAX_TOOL_ARG_SUMMARY_CHARS`
        characters with a trailing `"..."` when longer, so a tool call
        carrying an enormous argument payload (a large `edit_file`
        replacement, say) never floods this pane with its full text.
        """
        summary = sanitize_terminal(call.arguments_json)
        if len(summary) > _MAX_TOOL_ARG_SUMMARY_CHARS:
            summary = summary[:_MAX_TOOL_ARG_SUMMARY_CHARS] + "..."
        self.write(f"-> {call.name}({summary})")

    def append_finished(self, call: ToolCallEvent, *, elapsed_s: float) -> None:
        """Write a `"<- {name} ({elapsed_s:.1f}s)"` line for a tool call
        that just finished, pairing with the `append_started` line the
        same call already wrote."""
        self.write(f"<- {call.name} ({elapsed_s:.1f}s)")


class ArtifactPane(Markdown):
    """Renders the task's most recently produced `VerificationReport` as
    markdown. Shows a placeholder message until the first `verify` tool
    call of a submitted task calls `show_report`."""

    can_focus = True

    def show_report(self, report: VerificationReport) -> None:
        """Render `report` via `render_verification_markdown`, sanitized
        against hostile terminal escape sequences the same way every
        other pane guards model- or tool-sourced text, and display it as
        this pane's entire content. Only the most recent report is ever
        shown; this pane keeps no history of earlier reports to browse
        back through, matching `DiffPane`'s own "most recent only"
        precedent."""
        self.update(sanitize_terminal(render_verification_markdown(report)))


class DiffPane(Static):
    """Renders the most recent file mutation as a syntax-highlighted
    unified diff. Shows a placeholder message until the first
    `edit_file` mutation of a submitted task calls `show_diff`."""

    can_focus = True

    def show_diff(self, path: str, before: str | None, after: str | None) -> None:
        """Render a unified diff of `before` -> `after` and display it.

        Either side may be `None` -- `before is None` means the
        mutation created `path`, `after is None` means it deleted
        `path` -- and is treated as empty content for the diff.
        The rendered diff text is sanitized through `sanitize_terminal`
        before display, since it embeds a file's own (untrusted)
        content, then wrapped as a `rich.syntax.Syntax` object so it
        renders with diff syntax highlighting. Only the most recent
        mutation is ever shown; this pane keeps no history of earlier
        diffs to browse back through.
        """
        diff_lines = difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
        diff_text = sanitize_terminal("".join(diff_lines))
        self.update(Syntax(diff_text, "diff"))


class StatusBar(Static):
    """One-line live status: model, mode and effort level,
    context-window usage percentage, and session/day spend against
    their caps."""

    def show(self, snapshot: StatusSnapshot) -> None:
        """Render `snapshot` via `render_status_line` and replace this
        widget's displayed text with the result."""
        self.update(render_status_line(snapshot))


class KestrelApp(App[None]):
    """Kestrel's Textual cockpit: the interactive terminal interface
    entered when `kestrel` is invoked with no subcommand. Submitting a
    task in `#task_input` drives it through the real tool-calling agent
    loop -- every submission runs the full `run_task`, never a plain
    chat turn -- with `kestrel.tui.observer_bridge.TuiLoopObserver`
    bridging that task's own progress onto the panes below live. A
    `ctrl+p` command palette, registered via `COMMANDS` below, gives
    keyboard access to model/mode switching, undo, a cost breakdown,
    and resuming a prior task."""

    CSS_PATH = "kestrel.tcss"
    COMMANDS = {*App.COMMANDS, KestrelCommandProvider}
    BINDINGS = [
        Binding("f1", "focus_conversation", "Conversation"),
        Binding("f2", "focus_tool_log", "Tool log"),
        Binding("f3", "focus_diff", "Diff"),
        Binding("f4", "focus_artifact", "Artifact"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        config: KestrelConfig,
        registry: Registry,
        model_id: str,
        kestrel_md: KestrelMd | None,
        repo_root: Path,
        mode_manager: ModeManager | None = None,
    ) -> None:
        """Store every collaborator a submitted task needs to build its
        own `LoopDeps` via `build_task_deps`. `mode_manager` defaults to
        a fresh `ModeManager()` -- a plain `ModeManager()` default
        argument would be shared, mutably, across every `KestrelApp`
        instance that left it unset, so `None` stands in as the
        per-call sentinel instead.
        """
        super().__init__()
        self.config = config
        self.registry = registry
        self.active_model_id = model_id
        self.kestrel_md = kestrel_md
        self.repo_root = repo_root
        self.mode_manager = mode_manager if mode_manager is not None else ModeManager()
        self._current_task_id: str | None = None
        self._last_completed_task_id: str | None = None
        self._last_meter: CostMeter | None = None

    def compose(self) -> ComposeResult:
        """Lay out the status bar, docked top, above a two-column body:
        the conversation pane and its task-input box on the left, and
        the artifact, tool-log, and diff panes stacked on the right.
        A `LoadingIndicator`, docked directly under the status bar,
        shows for as long as a submitted task has a tool call in
        flight (see `_run_task`) and is otherwise hidden."""
        yield StatusBar(id="status_bar")
        yield LoadingIndicator(id="loading_indicator")
        with Horizontal():
            with Vertical(id="left_column"):
                yield ConversationPane(id="conversation", markup=False, wrap=True)
                yield Input(placeholder="Describe a task...", id="task_input")
            with Vertical(id="right_column"):
                yield ArtifactPane(id="artifact")
                yield Collapsible(
                    ToolLogPane(id="tool_log"), title="Tool log", collapsed=False
                )
                yield DiffPane(id="diff")

    def on_mount(self) -> None:
        """Populate every pane with its first-party placeholder content,
        hide the loading indicator (no tool call is in flight yet), and
        show a real idle status line built from this app's own starting
        model and mode."""
        self.query_one("#conversation", ConversationPane).write("Kestrel ready.")
        self.query_one("#artifact", ArtifactPane).update(_ARTIFACT_PLACEHOLDER)
        self.query_one("#diff", DiffPane).update("no changes yet")
        self.query_one("#loading_indicator", LoadingIndicator).display = False
        self._show_idle_status()

    def _show_idle_status(self) -> None:
        """Render and show a fresh idle `StatusSnapshot` for the
        currently active model and mode -- no turn has necessarily
        billed under either one yet, so this always reports
        `context_used_tokens=None` and `session_usd=Decimal(0)` rather
        than carrying over whatever a prior task's own
        `TuiLoopObserver` last showed. Called once on mount, and again
        by `action_switch_model`/`action_set_mode` whenever a palette
        selection changes the active model or mode outside of a
        running task."""
        entry = self.registry.get(self.active_model_id)
        budget_config = self.config.managers.budget
        idle_snapshot = StatusSnapshot(
            model_id=self.active_model_id,
            mode=self.mode_manager.mode,
            effort=self.mode_manager.effort(),
            context_used_tokens=None,
            context_window=entry.context_window,
            session_usd=Decimal(0),
            session_cap_usd=budget_config.session_usd,
            day_usd=Decimal(0),
            day_cap_usd=budget_config.day_usd,
        )
        self.query_one("#status_bar", StatusBar).show(idle_snapshot)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Drive `event.value` through a full task once it is submitted
        from `#task_input`; ignores an empty submission.

        Also declines a submission while `_current_task_id` names a
        task still in flight: running a second agent loop concurrently
        would have both interleave writes into the same conversation
        pane and status bar, and both act on the same repo at once.
        The input's own text is left in place (not cleared) so the
        submission can simply be retried once the running task ends.
        """
        if event.input.id != "task_input":
            return
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return
        if self._current_task_id is not None:
            self.query_one("#conversation", ConversationPane).write(
                "\n[busy] a task is already running -- resubmit once it finishes\n"
            )
            return
        event.input.value = ""
        self.run_worker(self._run_task(text), exclusive=False)

    async def _run_task(self, text: str) -> None:
        """Run `text` as a brand new task: builds this task's own
        `TaskSetup` and observer via `_prepare_task_run`, then drives it
        to completion via `run_task`.

        `_current_task_id` is cleared in a `finally` block so it always
        reflects reality -- including when `run_task` raises -- and a
        later submission is accepted again only once this one has
        fully ended. `_last_meter` and `_last_completed_task_id` are set
        from the finished task in that same block, but only once
        `_prepare_task_run` has actually returned a `TaskSetup` -- an
        exception raised before that (e.g. an unknown `active_model_id`)
        leaves both exactly as `/cost` and `/undo` last found them,
        rather than clobbering them with a task that never really
        started.
        """
        task_id = str(uuid.uuid4())
        self._current_task_id = task_id
        setup: TaskSetup | None = None
        try:
            conversation = self.query_one("#conversation", ConversationPane)
            conversation.write(f"\n> {sanitize_terminal(text)}\n")
            setup = self._prepare_task_run(task_id)
            await run_task(text, setup.deps, task_id)
        finally:
            self._current_task_id = None
            if setup is not None:
                self._last_meter = setup.deps.meter
                self._last_completed_task_id = task_id

    async def _resume_task(self, task_id: str) -> None:
        """Resume the prior task named by `task_id`: builds a fresh
        `TaskSetup` and observer via `_prepare_task_run` exactly as
        `_run_task` does, then drives it to completion via
        `kestrel.agent.loop.resume_task` instead of `run_task` -- the
        loaded session's own journaled history stands in for the
        `text` argument a brand new task would otherwise need.

        `_current_task_id`, `_last_meter`, and `_last_completed_task_id`
        are managed identically to `_run_task`, including the same
        finally-block guard against a `setup` that never got built.
        """
        self._current_task_id = task_id
        setup: TaskSetup | None = None
        try:
            conversation = self.query_one("#conversation", ConversationPane)
            conversation.write(f"\n> resuming task {task_id}\n")
            setup = self._prepare_task_run(task_id)
            await resume_task(task_id, setup.deps)
        finally:
            self._current_task_id = None
            if setup is not None:
                self._last_meter = setup.deps.meter
                self._last_completed_task_id = task_id

    def _prepare_task_run(self, task_id: str) -> TaskSetup:
        """Build `task_id`'s own `TaskSetup` via `build_task_deps`, then
        swap in a fresh `TuiLoopObserver` bridging its progress onto the
        conversation pane, tool log, diff pane, artifact pane, loading
        indicator, and status bar -- the collaborator-building steps
        `_run_task` and `_resume_task` otherwise share verbatim,
        factored out here so a brand new task and a resumed one are
        bridged onto the cockpit identically.

        The observer is built only after `build_task_deps` returns so
        it can be seeded with `setup.spent_day_usd` -- the real
        same-day spend `build_task_deps` already computed from the
        repo's own session journals -- rather than always starting the
        status bar's day figure from zero. `LoopDeps.observer` is a
        plain mutable field for exactly this kind of late binding; the
        caller does not await either loop entry point until after it is
        set, so neither one ever sees the placeholder `NULL_OBSERVER`
        `build_task_deps` itself defaults to.

        `_set_inflight`, a small closure over this task's own
        `#loading_indicator`, is the observer's `on_inflight_change`
        hook: it shows the indicator once at least one tool call is in
        flight and hides it again the moment none are.

        The artifact pane is reset to its placeholder text before this
        task's observer is even built, so a prior task's own
        `VerificationReport` never lingers on screen once a task that
        may not itself call `verify` starts -- whether that task is
        brand new or a resumed one.
        """
        entry = self.registry.get(self.active_model_id)
        setup = build_task_deps(
            config=self.config,
            registry=self.registry,
            model_id=self.active_model_id,
            kestrel_md=self.kestrel_md,
            repo_root=self.repo_root,
            task_id=task_id,
        )
        loading_indicator = self.query_one("#loading_indicator", LoadingIndicator)

        def _set_inflight(count: int) -> None:
            """Show `loading_indicator` while `count` is positive, hide
            it once it drops back to zero."""
            loading_indicator.display = count > 0

        artifact_pane = self.query_one("#artifact", ArtifactPane)
        artifact_pane.update(_ARTIFACT_PLACEHOLDER)
        setup.deps.observer = TuiLoopObserver(
            conversation=self.query_one("#conversation", ConversationPane),
            status_bar=self.query_one("#status_bar", StatusBar),
            undo=setup.undo,
            model_id=self.active_model_id,
            mode_manager=self.mode_manager,
            context_window=entry.context_window,
            session_cap_usd=self.config.managers.budget.session_usd,
            day_cap_usd=self.config.managers.budget.day_usd,
            spent_day_usd_baseline=setup.spent_day_usd,
            on_inflight_change=_set_inflight,
            tool_log=self.query_one("#tool_log", ToolLogPane),
            diff_pane=self.query_one("#diff", DiffPane),
            artifact_pane=artifact_pane,
        )
        return setup

    def action_focus_conversation(self) -> None:
        """Move focus to the task-input box (F1)."""
        self.query_one("#task_input", Input).focus()

    def action_focus_tool_log(self) -> None:
        """Move focus to the tool-log pane (F2)."""
        self.query_one("#tool_log", ToolLogPane).focus()

    def action_focus_diff(self) -> None:
        """Move focus to the diff pane (F3)."""
        self.query_one("#diff", DiffPane).focus()

    def action_focus_artifact(self) -> None:
        """Move focus to the artifact pane (F4)."""
        self.query_one("#artifact", ArtifactPane).focus()

    def action_switch_model(self, model_id: str) -> None:
        """Switch the active model to `model_id` and refresh the idle
        status line to reflect it.

        No task is running when a palette selection fires this action,
        so `_show_idle_status` -- not a running task's own
        `TuiLoopObserver` -- is what makes the change visible; the next
        task submitted after this call is what actually sends a turn to
        the new model.
        """
        self.active_model_id = model_id
        self._show_idle_status()

    def action_set_mode(self, mode: Mode) -> None:
        """Switch the active PLAN/FAST mode and refresh the idle status
        line the same way `action_switch_model` does."""
        self.mode_manager.set_mode(mode)
        self._show_idle_status()

    def _reject_while_task_active(self, retry_hint: str) -> bool:
        """Notify and return `True` when `_current_task_id` names a task
        still in flight, so a palette action that would otherwise mutate
        shared session state can decline cleanly instead of racing the
        task already running. `retry_hint` names the verb the warning
        tells the user to retry (e.g. `"undo"`, `"switch"`, `"resume"`)
        once the running task finishes; returns `False`, notifying
        nothing, when no task is active.
        """
        if self._current_task_id is None:
            return False
        self.notify(
            f"a task is still running -- {retry_hint} once it finishes",
            severity="warning",
        )
        return True

    def action_undo_current_task(self) -> None:
        """Revert the most recently *finished* task's own journaled
        mutations, declining instead while a task is active (see
        `_reject_while_task_active`) or when no task has finished yet
        this session.

        Reverting a still-running task's own mutations would race that
        task's own tool calls, which may still be writing to the same
        paths a revert would touch -- `_current_task_id` names an
        active task, never a finished one, so undo only ever targets
        `_last_completed_task_id`, captured here and handed to
        `_undo_current_task` as a worker argument rather than read back
        off shared state once that worker actually starts running.
        Scheduled as a worker rather than performed inline, since
        reverting talks to the filesystem and every other
        filesystem-touching call in this codebase's own async call
        sites is likewise kept off the widget-handling coroutine.
        """
        if self._reject_while_task_active("undo"):
            return
        task_id = self._last_completed_task_id
        if task_id is None:
            self.notify("no task to undo yet", severity="warning")
            return
        self.run_worker(self._undo_current_task(task_id))

    async def _undo_current_task(self, task_id: str) -> None:
        """Revert `task_id`'s own mutations via a fresh `UndoManager`
        and write a one-line summary to the conversation pane naming
        how many were reverted.

        Takes `task_id` as a parameter, captured by
        `action_undo_current_task` before this worker was even
        scheduled, rather than reading `self._last_completed_task_id`
        again once running -- the task this call reverts is always the
        one the caller decided on, never whatever that attribute might
        have become (e.g. a later task finishing) by the time this
        coroutine's first await runs. `UndoManager.revert_task`
        performs real file I/O, so it runs via `asyncio.to_thread` off
        the event loop this coroutine itself runs on.
        """
        reverted = await asyncio.to_thread(
            UndoManager(repo_root=self.repo_root).revert_task, task_id
        )
        self.query_one("#conversation", ConversationPane).write(
            f"undo: reverted {len(reverted)} mutation(s) for task {task_id}"
        )

    def action_show_cost(self) -> None:
        """Write the most recently run task's own cost breakdown to the
        conversation pane: the session total, then one
        `format_cost_line` per recorded turn, reusing the REPL's own
        `/cost` rendering verbatim rather than reimplementing it.

        Before any task has run this session, `self._last_meter` is
        still `None`, so this writes a single "no turns recorded yet"
        line instead of a breakdown with nothing in it.
        """
        conversation = self.query_one("#conversation", ConversationPane)
        meter = self._last_meter
        if meter is None:
            conversation.write("no turns recorded yet")
            return
        conversation.write(f"session total: ${meter.session_usd:.4f}")
        for turn in meter.turns:
            conversation.write(format_cost_line(turn, meter.session_usd))

    def action_show_approve_info(self) -> None:
        """Tell the user approvals already happen automatically, rather
        than opening a queue that does not exist: every destructive
        action a running task proposes surfaces its own prompt the
        instant it comes up, so there is never a backlog to browse."""
        self.notify(
            "Approvals appear automatically as a modal when a "
            "destructive action is proposed."
        )

    def action_show_kb_info(self) -> None:
        """Tell the user the knowledge base is not available yet,
        rather than stubbing out storage or retrieval this codebase
        does not otherwise implement."""
        self.notify("Knowledge base is not available yet.")

    def list_resumable_task_ids(self) -> list[str]:
        """Every task id with a session journal on disk under this
        repo's `.kestrel/sessions/` directory, sorted for stable,
        deterministic ordering across repeated palette searches --
        `[]` when that directory does not exist yet, meaning no task in
        this repo has ever been journaled."""
        sessions_dir = self.repo_root / ".kestrel" / "sessions"
        if not sessions_dir.exists():
            return []
        return sorted(path.stem for path in sessions_dir.glob("*.jsonl"))

    def action_resume_task(self, task_id: str) -> None:
        """Resume the prior task named by `task_id` as a background
        worker, the same way `on_input_submitted` schedules a brand new
        one."""
        self.run_worker(self._resume_task(task_id))
