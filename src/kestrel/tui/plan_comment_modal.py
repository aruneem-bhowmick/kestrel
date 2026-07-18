"""Modal screen for attaching a review comment to one plan line.

`PlanCommentModal` is `kestrel.tui.approval_modal.ApprovalModal`'s own
direct structural sibling: a `ModalScreen` composing `Static`/`Input`/
`Button` widgets, dismissing with a typed value, driven by both a bound
key and button presses. Where `ApprovalModal` asks the user to approve
or deny a proposed destructive action, `PlanCommentModal` asks for one
`kestrel.agent.plan.PlanComment` -- a 1-based plan-line number and free
text -- against the plan currently on screen.

There is no true text-selection widget backing "select a line": a plan
is rendered as plain `Markdown`, which offers no API for selecting a
span of its own rendered text. `kestrel.agent.plan.PlanLine.index`
stands in instead -- the user names a line by its printed number, the
same number `render_plan_markdown` already prints beside it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from kestrel.agent.plan import ImplementationPlan, PlanComment, render_plan_markdown
from kestrel.repl import sanitize_terminal


class PlanCommentModal(ModalScreen[PlanComment | None]):
    """Asks for one plan-line comment -- a 1-based line number and free
    text -- dismissing with a `PlanComment` naming that line's own
    captured text, or `None` on cancel.

    `escape` cancels; there is no bound key for submitting, since a
    comment's own free text is typed into an `Input`, so submission is
    always the "Add comment" button's own `on_button_pressed` path.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, plan: ImplementationPlan) -> None:
        """Store `plan`; nothing is rendered until `compose` runs."""
        super().__init__()
        self._plan = plan

    def compose(self) -> ComposeResult:
        """Lay out the plan's own rendered text above a line-number
        input, a comment input, and a row of "Add comment"/"Cancel"
        buttons.

        `#plan_comment_lines` renders `sanitize_terminal(render_plan_markdown
        (self._plan))` with `markup=False` -- the same hostile-content
        guard `ApprovalModal` applies to `request.summary`/`request.detail`,
        since a plan line's own text ultimately traces back to a model
        reply, not text this modal itself controls. It is wrapped in a
        focusable `#plan_comment_lines_scroll` (a `VerticalScroll`), the
        same `ApprovalModal.compose` already uses for its own
        `#approval_detail`, so a plan too long to fit the dialog scrolls
        via the keyboard instead of clipping lines the user would
        otherwise have no way to read -- let alone comment on by
        number.
        """
        yield Vertical(
            VerticalScroll(
                Static(
                    sanitize_terminal(render_plan_markdown(self._plan)),
                    id="plan_comment_lines",
                    markup=False,
                ),
                id="plan_comment_lines_scroll",
            ),
            Input(placeholder="line number", id="plan_comment_line_number"),
            Input(placeholder="comment", id="plan_comment_text"),
            Horizontal(
                Button("Add comment", id="submit", variant="success"),
                Button("Cancel", id="cancel", variant="error"),
                id="plan_comment_buttons",
            ),
            id="plan_comment_dialog",
        )

    def action_cancel(self) -> None:
        """Dismiss with `None` (bound to `escape`)."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with `None` for the `cancel` button; otherwise
        validate and submit via `_submit`."""
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def _submit(self) -> None:
        """Validate both inputs and dismiss with a `PlanComment`, or
        `notify` an error and leave the modal open for correction.

        Three ways this declines to dismiss, each surfaced as an
        error-severity notification rather than a raised exception,
        since a mistyped line number or an empty comment is an
        ordinary user-correctable mistake, not a bug: `#plan_comment
        _line_number` is not a valid integer; no `PlanLine` in
        `self._plan.lines` has that `index`; or `#plan_comment_text` is
        empty after stripping surrounding whitespace.

        `#plan_comment_text`'s own raw, unstripped value is what
        `PlanComment.comment` carries -- stripping is used only to
        decide whether the field is meaningfully empty. A comment
        deliberately typed with leading or trailing whitespace (quoting
        a snippet, say) reaches the model exactly as written, rather
        than silently losing that whitespace on the way there.
        """
        number_raw = self.query_one("#plan_comment_line_number", Input).value.strip()
        comment_raw = self.query_one("#plan_comment_text", Input).value
        try:
            index = int(number_raw)
        except ValueError:
            self.app.notify("line number must be an integer", severity="error")
            return
        line = next((pl for pl in self._plan.lines if pl.index == index), None)
        if line is None:
            self.app.notify(f"no plan line {index}", severity="error")
            return
        if not comment_raw.strip():
            self.app.notify("comment must not be empty", severity="error")
            return
        self.dismiss(
            PlanComment(line_index=line.index, line_text=line.text, comment=comment_raw)
        )
