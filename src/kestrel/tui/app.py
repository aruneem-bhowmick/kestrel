"""Kestrel's Textual cockpit: `KestrelApp` and the panes it composes.

This module lays out and styles the interactive terminal interface --
a conversation view, an artifact viewer, a collapsible tool log, a diff
view, and a status bar -- in a restrained rust-and-slate default theme.
Submitting text in the task-input box drives a real `run_task` call
through `kestrel.task_setup.build_task_deps`: the conversation pane
streams the assistant's own text as it arrives, the status bar updates
live after every turn, and the task's own termination prints a terse
summary. Every pane that renders model- or tool-sourced text routes it
through `kestrel.repl.sanitize_terminal` first, whether directly (the
conversation pane) or via `kestrel.tui.observer_bridge.TuiLoopObserver`.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Input, Markdown, RichLog, Static

from kestrel.agent.loop import run_task
from kestrel.config import KestrelConfig
from kestrel.kestrel_md import KestrelMd
from kestrel.managers.mode import ModeManager
from kestrel.managers.undo import UndoManager
from kestrel.registry.model import Registry
from kestrel.repl import sanitize_terminal
from kestrel.task_setup import build_task_deps
from kestrel.tui.observer_bridge import TuiLoopObserver
from kestrel.tui.status import StatusSnapshot, render_status_line


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
    collapsible. Left empty (no placeholder text) until a later change
    starts writing real entries to it."""


class ArtifactPane(Markdown):
    """Renders the task's most recently produced artifact as markdown.
    Shows a placeholder message until a later change wires in real
    artifact content."""

    can_focus = True


class DiffPane(Static):
    """Renders the most recent file mutation as a syntax-highlighted
    unified diff. Shows a placeholder message until a later change
    computes a real diff to render."""

    can_focus = True


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
    bridging that task's own progress onto the panes below live."""

    CSS_PATH = "kestrel.tcss"
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

    def compose(self) -> ComposeResult:
        """Lay out the status bar, docked top, above a two-column body:
        the conversation pane and its task-input box on the left, and
        the artifact, tool-log, and diff panes stacked on the right."""
        yield StatusBar(id="status_bar")
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
        and show a real idle status line -- no turn has billed yet
        (`context_used_tokens=None`, `session_usd=Decimal(0)`), built
        from this app's own starting model and mode."""
        self.query_one("#conversation", ConversationPane).write("Kestrel ready.")
        self.query_one("#artifact", ArtifactPane).update("_no artifact yet_")
        self.query_one("#diff", DiffPane).update("no changes yet")

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
        `TaskSetup` via `build_task_deps`, bridging its progress onto
        the conversation pane and status bar through a fresh
        `TuiLoopObserver`, then drives it to completion via `run_task`.

        `spent_day_usd_baseline` starts at zero -- a fresh submission
        starts a task with no prior same-day history of its own to add;
        a later resume path seeds a real baseline instead.

        `_current_task_id` is cleared in a `finally` block so it always
        reflects reality -- including when `run_task` raises -- and a
        later submission is accepted again only once this one has
        fully ended.
        """
        task_id = str(uuid.uuid4())
        self._current_task_id = task_id
        try:
            conversation = self.query_one("#conversation", ConversationPane)
            conversation.write(f"\n> {sanitize_terminal(text)}\n")
            entry = self.registry.get(self.active_model_id)
            setup = build_task_deps(
                config=self.config,
                registry=self.registry,
                model_id=self.active_model_id,
                kestrel_md=self.kestrel_md,
                repo_root=self.repo_root,
                task_id=task_id,
                observer=TuiLoopObserver(
                    conversation=conversation,
                    status_bar=self.query_one("#status_bar", StatusBar),
                    undo=UndoManager(repo_root=self.repo_root),
                    model_id=self.active_model_id,
                    mode_manager=self.mode_manager,
                    context_window=entry.context_window,
                    session_cap_usd=self.config.managers.budget.session_usd,
                    day_cap_usd=self.config.managers.budget.day_usd,
                    spent_day_usd_baseline=Decimal(0),
                ),
            )
            await run_task(text, setup.deps, task_id)
        finally:
            self._current_task_id = None

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
