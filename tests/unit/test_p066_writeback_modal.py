"""Unit tests for `kestrel.tui.writeback_modal.WritebackModal`: its
`compose()` renders one pre-checked checkbox per proposed learning plus
the documented `commit`/`cancel` buttons, unchecking one and pressing
Commit dismisses with the remaining approved subset, and Cancel (the
button or `escape`) always dismisses with `()`, regardless of any
checkbox's own state.

Mirrors `test_p051_plan_comment_modal.py`'s own structure: the
compose-time cases construct `WritebackModal` directly and read back
`compose()`'s own pre-mount `_pending_children` bookkeeping, without
ever mounting a real Textual app; the checkbox-interaction cases do
mount a real `KestrelApp` via `kestrel_app_factory`, since toggling a
`Checkbox`'s own `value` and reading it back afterward needs a
genuinely mounted widget tree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from textual.widgets import Button, Checkbox

from kestrel.kb.writeback import ProposedLearning
from kestrel.tui.app import KestrelApp
from kestrel.tui.writeback_modal import WritebackModal

pytestmark = [pytest.mark.p066, pytest.mark.unit, pytest.mark.ui]


def _learnings(*texts: str) -> tuple[ProposedLearning, ...]:
    """One `ProposedLearning` per `texts` entry, each with a fixed,
    otherwise-unused single tag."""
    return tuple(ProposedLearning(text=text, tags=("tag",)) for text in texts)


def _walk_pending(widget: object) -> Iterator[object]:
    """Yield `widget` and every descendant Textual will mount later,
    read from the compose-time `_pending_children` bookkeeping every
    container stores before it is ever actually mounted -- the only
    way to reach a `compose()` return value's nested widgets without
    mounting a real app."""
    yield widget
    for child in getattr(widget, "_pending_children", ()):
        yield from _walk_pending(child)


def _widgets_by_id(proposed: tuple[ProposedLearning, ...]) -> dict[str, object]:
    """Compose `WritebackModal(proposed)` and return every id-bearing
    widget it (recursively) yields, keyed by that widget's own id."""
    modal = WritebackModal(proposed)
    (root,) = modal.compose()
    return {
        widget.id: widget  # type: ignore[attr-defined]
        for widget in _walk_pending(root)
        if getattr(widget, "id", None) is not None
    }


def test_compose_renders_one_pre_checked_checkbox_per_proposal() -> None:
    """Given three proposed learnings, when the modal composes, then
    three checkboxes render, each pre-checked."""
    proposed = _learnings("first", "second", "third")
    widgets = _widgets_by_id(proposed)

    checkboxes = [widgets[f"writeback_checkbox_{i}"] for i in range(3)]
    for checkbox in checkboxes:
        assert isinstance(checkbox, Checkbox)
        assert checkbox.value is True


def test_compose_renders_the_documented_commit_and_cancel_buttons() -> None:
    """Given any proposal batch, when the modal composes, then both
    documented buttons are present."""
    widgets = _widgets_by_id(_learnings("only one"))

    assert isinstance(widgets["commit"], Button)
    assert isinstance(widgets["cancel"], Button)


def test_action_cancel_dismisses_with_the_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal, when `action_cancel` is invoked directly, then
    `dismiss` is called exactly once with `()`."""
    modal = WritebackModal(_learnings("a"))
    dismissed: list[tuple[ProposedLearning, ...]] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)

    modal.action_cancel()

    assert dismissed == [()]


def test_cancel_button_dismisses_with_the_empty_tuple_regardless_of_checkbox_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal whose own checkboxes were never touched (still
    default-checked), when the `cancel` button fires, then `dismiss`
    is called with `()` -- pressing Cancel never reads checkbox state
    at all."""
    modal = WritebackModal(_learnings("a", "b"))
    list(modal.compose())  # populates modal._checkboxes as a side effect
    dismissed: list[tuple[ProposedLearning, ...]] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)

    modal.on_button_pressed(Button.Pressed(Button("Cancel", id="cancel")))

    assert dismissed == [()]


async def test_unchecking_one_and_committing_dismisses_with_the_remaining_subset(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal mounted against a real app with three proposals,
    when the middle checkbox is unchecked and `commit` is pressed,
    then `dismiss` is called exactly once with the first and third
    proposals, in their original order -- the unchecked middle one is
    excluded."""
    proposed = _learnings("keep first", "drop this", "keep third")
    async with kestrel_app_factory().run_test() as pilot:
        modal = WritebackModal(proposed)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#writeback_checkbox_1", Checkbox).value = False
        dismissed: list[tuple[ProposedLearning, ...]] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)

        modal.on_button_pressed(Button.Pressed(Button("Commit", id="commit")))

        assert dismissed == [(proposed[0], proposed[2])]


async def test_committing_with_every_checkbox_still_checked_dismisses_with_all(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal mounted against a real app, when `commit` is
    pressed without unchecking anything, then `dismiss` is called with
    every proposed learning, in order -- the default-checked state
    approves the whole batch."""
    proposed = _learnings("first", "second")
    async with kestrel_app_factory().run_test() as pilot:
        modal = WritebackModal(proposed)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        dismissed: list[tuple[ProposedLearning, ...]] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)

        modal.on_button_pressed(Button.Pressed(Button("Commit", id="commit")))

        assert dismissed == [proposed]


async def test_unchecking_every_checkbox_and_committing_dismisses_with_the_empty_tuple(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal mounted against a real app, when every checkbox is
    unchecked and `commit` is pressed, then `dismiss` is called with
    `()` -- indistinguishable, by design, from a Cancel."""
    proposed = _learnings("first", "second")
    async with kestrel_app_factory().run_test() as pilot:
        modal = WritebackModal(proposed)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#writeback_checkbox_0", Checkbox).value = False
        modal.query_one("#writeback_checkbox_1", Checkbox).value = False
        dismissed: list[tuple[ProposedLearning, ...]] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)

        modal.on_button_pressed(Button.Pressed(Button("Commit", id="commit")))

        assert dismissed == [()]
