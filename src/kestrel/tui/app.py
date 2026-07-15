"""Kestrel's Textual cockpit: `KestrelApp` and the panes it composes.

This module lays out and styles the interactive terminal interface --
a conversation view, an artifact viewer, a collapsible tool log, a diff
view, and a status bar -- in a restrained rust-and-slate default theme.
Nothing here talks to a model, dispatches a tool, or computes a real
diff; every pane below is mounted with static placeholder content only,
and each will be filled in with live behavior by later changes without
touching this module's own layout code. Any pane that goes on to render
model- or tool-sourced text must route it through
`kestrel.repl.sanitize_terminal` first -- noted here so later work
inherits that rule, not applied yet since this module renders only its
own first-party literal strings.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Input, Markdown, RichLog, Static


class ConversationPane(RichLog):
    """Streams the active task's assistant text and turn/termination
    summaries. Append-only; a later change adds the incremental-write
    path that keeps it updated as a task runs. For now it is mounted
    with a single placeholder line."""


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


class DiffPane(Static):
    """Renders the most recent file mutation as a syntax-highlighted
    unified diff. Shows a placeholder message until a later change
    computes a real diff to render."""


class StatusBar(Static):
    """One-line live status: model, mode and effort level,
    context-window usage percentage, and session/day spend against
    their caps. Shows a placeholder line until a later change starts
    populating it with real session state."""


class KestrelApp(App[None]):
    """Kestrel's Textual cockpit: the interactive terminal interface
    entered when `kestrel` is invoked with no subcommand. Every widget
    it mounts here holds only static placeholder content -- later
    changes populate each pane with live behavior without touching this
    class's own layout code."""

    CSS_PATH = "kestrel.tcss"
    BINDINGS = [
        Binding("f1", "focus_conversation", "Conversation"),
        Binding("f2", "focus_tool_log", "Tool log"),
        Binding("f3", "focus_diff", "Diff"),
        Binding("f4", "focus_artifact", "Artifact"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

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
        """Populate every pane with its first-party placeholder content
        -- no model, tool, or artifact data exists yet at this stage."""
        self.query_one("#conversation", ConversationPane).write("Kestrel ready.")
        self.query_one("#artifact", ArtifactPane).update("_no artifact yet_")
        self.query_one("#diff", DiffPane).update("no changes yet")
        self.query_one("#status_bar", StatusBar).update("kestrel -- idle")

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
