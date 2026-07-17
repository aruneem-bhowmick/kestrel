"""Modal approval screen for the TUI's own destructive-action gate.

`kestrel.managers.approval.ApprovalManager.check` resolves any request
not already allowlisted or session-approved by calling an injectable
`decide_fn`; the CLI's own default, `_prompt_stdin`, blocks on a real
terminal `input()` call. Inside `KestrelApp`, the tool call proposing
that request runs via `asyncio.to_thread` (`agent/loop.py`'s own
`_drive`) -- the one call in this codebase that genuinely executes on a
background OS thread -- so `decide_fn` there cannot simply open a
Textual modal directly: a `ModalScreen` can only be pushed and awaited
on the app's own event loop, and calling into Textual from the wrong
thread would either crash outright or silently do nothing.

`ApprovalModal` is the on-screen prompt itself, rendering the exact
same `summary`/`detail` content `_prompt_stdin` already prints on the
CLI path. `make_tui_decide_fn` is the bridge: it builds a `decide_fn`
that, called from that background thread, schedules `ApprovalModal`
onto the app's own event loop and blocks the calling thread until the
modal is dismissed with a real decision.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from kestrel.managers.approval import ApprovalDecision, ApprovalRequest
from kestrel.repl import sanitize_terminal

if TYPE_CHECKING:
    # Deferred to break the import cycle: `kestrel.tui.app.KestrelApp`
    # calls `make_tui_decide_fn` to build each task's own `decide_fn`,
    # so a module-level import here of `KestrelApp` would import
    # `kestrel.tui.app` right back into this module while it is still
    # loading. `from __future__ import annotations` (above) already
    # makes every annotation in this file a lazy string, so this name
    # is only ever needed for static type-checking, never at runtime.
    from kestrel.tui.app import KestrelApp


class ApprovalModal(ModalScreen[ApprovalDecision]):
    """Shows one `ApprovalRequest`'s summary and exact detail, dismissing
    with the `ApprovalDecision` the user picks.

    Rendered content is `request.summary`/`request.detail`, each passed
    through `sanitize_terminal` and displayed verbatim -- the same
    content the CLI's own `_prompt_stdin` prints, so the modal never
    reformats, truncates, or otherwise obscures the exact command or
    diff being approved. Three ways to decide, all equivalent: the
    bound keys `y`/`a`/`n` (`escape` doubles for `n`), or clicking one
    of the three buttons `compose` yields.
    """

    BINDINGS = [
        Binding("y", "decide_once", "Approve once"),
        Binding("a", "decide_always", "Always"),
        Binding("n,escape", "decide_deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        """Store `request`; nothing is rendered until `compose` runs."""
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        """Lay out `request.summary` and `request.detail` (both
        sanitized) above a row of the three decision buttons, each
        keyed to the `ApprovalDecision` its own `id` names.

        Both `Static` widgets pass `markup=False`: `sanitize_terminal`
        strips terminal control bytes and ANSI escapes, but never
        touches Rich's own bracket markup syntax (`[bold]`,
        `[conceal]`, `:emoji:` shortcodes, ...), which is built from
        ordinary printable characters. Left enabled, a hostile
        `summary`/`detail` could still change what actually reaches
        the screen -- a `[conceal]...[/conceal]` span, for instance,
        renders as invisible text -- defeating the one guarantee this
        modal exists to make: showing the exact command or diff,
        unmodified. `#approval_detail` is wrapped in a `VerticalScroll`
        so a long command or diff scrolls in place instead of growing
        the dialog until the decision buttons are pushed off screen
        (bounded together with `kestrel.tcss`'s own sizing rules).
        """
        yield Vertical(
            Static(
                sanitize_terminal(self._request.summary),
                id="approval_summary",
                markup=False,
            ),
            VerticalScroll(
                Static(
                    sanitize_terminal(self._request.detail),
                    id="approval_detail",
                    markup=False,
                ),
                id="approval_detail_scroll",
            ),
            Horizontal(
                Button("Approve once (y)", id="once", variant="success"),
                Button("Always (a)", id="always", variant="warning"),
                Button("Deny (n)", id="deny", variant="error"),
                id="approval_buttons",
            ),
            id="approval_dialog",
        )

    def action_decide_once(self) -> None:
        """Dismiss with `"once"` (bound to `y`)."""
        self.dismiss("once")

    def action_decide_always(self) -> None:
        """Dismiss with `"always"` (bound to `a`)."""
        self.dismiss("always")

    def action_decide_deny(self) -> None:
        """Dismiss with `"deny"` (bound to `n`/`escape`)."""
        self.dismiss("deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the `ApprovalDecision` named by the pressed
        button's own `id` -- `"once"`/`"always"`/`"deny"`, matching
        `compose`'s three button ids exactly."""
        self.dismiss(cast(ApprovalDecision, event.button.id))


def make_tui_decide_fn(
    app: KestrelApp, loop: asyncio.AbstractEventLoop
) -> Callable[[ApprovalRequest], ApprovalDecision]:
    """Bridge `ApprovalManager.check`'s synchronous `decide_fn` contract
    onto `app`'s own asyncio event loop.

    The returned function is called from inside `_dispatch_tool_call`,
    itself run via `asyncio.to_thread` on a real background thread.
    Pushing a `ModalScreen` and awaiting its dismissal
    (`App.push_screen_wait`) is a Textual coroutine that can only run
    on the loop actually driving `app`, so the returned function
    schedules it there via `asyncio.run_coroutine_threadsafe` and
    blocks the calling thread on the resulting
    `concurrent.futures.Future` until a real decision comes back.

    `loop` must be the loop actually driving `app` -- callers capture
    it via `asyncio.get_running_loop()` on the coroutine that is about
    to launch the task's own worker, before that worker's background
    thread ever starts, so there is no window in which a decision could
    need `loop` before it has been captured.
    """

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        """Push `ApprovalModal(request)` onto `app`'s screen stack from
        the calling (background) thread and block until it is
        dismissed, returning the `ApprovalDecision` it was dismissed
        with."""
        future = asyncio.run_coroutine_threadsafe(
            app.push_screen_wait(ApprovalModal(request)), loop
        )
        return future.result()

    return decide
