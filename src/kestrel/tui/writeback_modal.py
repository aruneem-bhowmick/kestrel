"""Modal screen for approving a subset of the model's own proposed
durable learnings.

`WritebackModal` is `kestrel.tui.plan_comment_modal.PlanCommentModal`'s
own structural sibling: a `ModalScreen` composing `Checkbox`/`Button`
widgets, dismissing with a typed value on either a bound key or a
button press. Where `PlanCommentModal` asks for one new `PlanComment`,
this modal asks the user to whittle a model-proposed batch of learnings
down to the subset actually worth keeping -- every checkbox starts
checked (approve by default), the opposite of
`kestrel.tui.approval_modal.ApprovalModal`'s own deliberately
deny-by-default destructive-action gate, since a proposed learning is
non-destructive and the model already exercised its own judgment
proposing it in the first place; the human's role here is to catch a
bad one, not to approve each one from a neutral starting point.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox

from kestrel.kb.writeback import ProposedLearning
from kestrel.repl import sanitize_terminal


class WritebackModal(ModalScreen[tuple[ProposedLearning, ...]]):
    """Lists every `ProposedLearning` with a pre-checked checkbox and a
    Commit button, dismissing with the checked subset -- or `()` on
    Cancel/Escape, never `None`, so the caller (`KestrelApp._on_
    writeback_decision`) never has to special-case "modal cancelled"
    separately from "nothing was approved."
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, proposed: Sequence[ProposedLearning]) -> None:
        """Store `proposed`; nothing is rendered until `compose` runs."""
        super().__init__()
        self._proposed = tuple(proposed)
        self._checkboxes: list[Checkbox] = []

    def compose(self) -> ComposeResult:
        """Lay out one pre-checked `Checkbox` per proposed learning
        inside a focusable `VerticalScroll`, above a "Commit"/"Cancel"
        button row -- the same scrolling-body-over-fixed-buttons shape
        `ApprovalModal`/`PlanCommentModal` already use, so a long batch
        of proposals scrolls instead of pushing the buttons off screen.

        Each checkbox's own label is built from a literal `Content`
        wrapping `sanitize_terminal(learning.text)` plus its tags,
        never a bare `str`: `Checkbox` has no `markup=False`
        constructor argument the way `Static`/`RichLog` do, and a
        `str` label is otherwise parsed as Textual markup on the way
        in. `Content.from_text` -- what `Checkbox` calls internally --
        returns an already-built `Content` instance unchanged rather
        than re-parsing it, so constructing one directly here is what
        stands in for `markup=False`.
        """
        checkboxes = [
            Checkbox(_checkbox_label(learning), True, id=f"writeback_checkbox_{i}")
            for i, learning in enumerate(self._proposed)
        ]
        self._checkboxes = checkboxes
        yield Vertical(
            VerticalScroll(*checkboxes, id="writeback_checkboxes_scroll"),
            Horizontal(
                Button("Commit", id="commit", variant="success"),
                Button("Cancel", id="cancel", variant="error"),
                id="writeback_buttons",
            ),
            id="writeback_dialog",
        )

    def action_cancel(self) -> None:
        """Dismiss with `()` (bound to `escape`)."""
        self.dismiss(())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with `()` for the `cancel` button, regardless of any
        checkbox's own state; otherwise dismiss with the checked subset
        of `self._proposed`, read off `self._checkboxes` by index."""
        if event.button.id == "cancel":
            self.dismiss(())
            return
        approved = tuple(
            learning
            for learning, checkbox in zip(self._proposed, self._checkboxes)
            if checkbox.value
        )
        self.dismiss(approved)


def _checkbox_label(learning: ProposedLearning) -> Content:
    """`learning`'s own text, plus its tags in brackets when it has any,
    sanitized and wrapped as a literal `Content` -- never markup-parsed,
    and never split across lines regardless of what `learning.text`
    itself contains, since `Checkbox` only ever renders a label's own
    first line."""
    label = sanitize_terminal(learning.text)
    if learning.tags:
        tags = ", ".join(sanitize_terminal(tag) for tag in learning.tags)
        label += f" [{tags}]"
    return Content(label)
