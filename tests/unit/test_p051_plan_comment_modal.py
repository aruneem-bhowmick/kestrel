"""Unit tests for `kestrel.tui.plan_comment_modal.PlanCommentModal`: its
`compose()` renders the plan's own markdown and the documented
input/button ids, submitting with a valid line number and non-empty
comment dismisses with the matching `PlanComment`, and each invalid
input -- a non-integer line number, an out-of-range line number, and an
empty comment -- notifies an error and leaves the modal open (`dismiss`
is never called). `action_cancel` and the `cancel` button both dismiss
with `None`.

Mirrors `test_p042_approval_modal.py`'s own structure: the compose-time
and cancel-path cases construct `PlanCommentModal` directly and read
back `compose()`'s own pre-mount `_pending_children` bookkeeping or spy
on `dismiss`, without ever mounting a real Textual app. The submission
cases (valid and invalid) do mount a real `KestrelApp` via
`kestrel_app_factory`, since `_submit`'s own `query_one` and
`self.app.notify` calls both require a genuinely mounted widget tree.

Also covers one red-team case: a plan line carrying the injection
corpus's `ansi_escape_laden_payload`, the same hostile-content shape
`test_p042_approval_modal.py` already exercises against
`ApprovalModal`, must render inertly in `#plan_comment_lines`; and one
UI case, mirroring `test_p042_approval_modal.py`'s own long-detail
scenario, mounting a real app to prove a plan too long to fit the
dialog scrolls via a focusable container instead of clipping lines the
user has no way to read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from kestrel.agent.plan import (
    ImplementationPlan,
    PlanComment,
    PlanLine,
    render_plan_markdown,
)
from kestrel.repl import sanitize_terminal
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tui.app import KestrelApp
from kestrel.tui.plan_comment_modal import PlanCommentModal

pytestmark = [pytest.mark.p051, pytest.mark.unit, pytest.mark.sanity]

_TASK_ID = "task-p051"
_HOSTILE_CASE_ID = "ansi_escape_laden_payload"


def _plan(*, lines: tuple[PlanLine, ...]) -> ImplementationPlan:
    """A minimal `ImplementationPlan` carrying `lines` verbatim;
    `raw_text` is never read by `render_plan_markdown`, so it is left
    empty."""
    return ImplementationPlan(task_id=_TASK_ID, raw_text="", lines=lines)


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


def _walk_pending(widget: object) -> Iterator[object]:
    """Yield `widget` and every descendant Textual will mount later,
    read from the compose-time `_pending_children` bookkeeping every
    container stores before it is ever actually mounted -- the only
    way to reach a `compose()` return value's nested widgets without
    mounting a real app."""
    yield widget
    for child in getattr(widget, "_pending_children", ()):
        yield from _walk_pending(child)


def _widgets_by_id(plan: ImplementationPlan) -> dict[str, object]:
    """Compose `PlanCommentModal(plan)` and return every id-bearing
    widget it (recursively) yields, keyed by that widget's own id."""
    modal = PlanCommentModal(plan)
    (root,) = modal.compose()
    return {
        widget.id: widget  # type: ignore[attr-defined]
        for widget in _walk_pending(root)
        if getattr(widget, "id", None) is not None
    }


def _default_plan() -> ImplementationPlan:
    """A plan with two ordinary lines, standing in for a real PLAN-mode
    reply's own parsed content across most cases here."""
    return _plan(
        lines=(
            PlanLine(index=1, text="Add an authentication middleware module."),
            PlanLine(index=2, text="Wire it into the request pipeline."),
        )
    )


def test_compose_renders_the_plan_into_the_documented_id() -> None:
    """Given a plan, when the modal composes, then
    `#plan_comment_lines` holds `sanitize_terminal(render_plan_markdown
    (plan))`."""
    plan = _default_plan()
    widgets = _widgets_by_id(plan)

    lines_widget = widgets["plan_comment_lines"]
    assert isinstance(lines_widget, Static)
    assert lines_widget.content == sanitize_terminal(render_plan_markdown(plan))


def test_compose_renders_the_documented_input_and_button_ids() -> None:
    """Given a plan, when the modal composes, then both documented
    inputs and both documented buttons are present."""
    widgets = _widgets_by_id(_default_plan())

    assert isinstance(widgets["plan_comment_line_number"], Input)
    assert isinstance(widgets["plan_comment_text"], Input)
    assert isinstance(widgets["submit"], Button)
    assert isinstance(widgets["cancel"], Button)


def test_action_cancel_dismisses_with_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a modal, when `action_cancel` is invoked directly, then
    `dismiss` is called exactly once with `None`."""
    modal = PlanCommentModal(_default_plan())
    dismissed: list[PlanComment | None] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)

    modal.action_cancel()

    assert dismissed == [None]


def test_cancel_button_dismisses_with_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a modal, when a `Button.Pressed` event for the `cancel`
    button fires, then `dismiss` is called exactly once with `None`."""
    modal = PlanCommentModal(_default_plan())
    dismissed: list[PlanComment | None] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)
    button = Button("Cancel", id="cancel")

    modal.on_button_pressed(Button.Pressed(button))

    assert dismissed == [None]


async def test_valid_submission_dismisses_with_the_matching_plan_comment(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal mounted against a real app, when both inputs carry
    a valid line number and a non-empty comment and the `submit`
    button is pressed, then `dismiss` is called exactly once with a
    `PlanComment` naming that line's own index and text."""
    plan = _default_plan()
    async with kestrel_app_factory().run_test() as pilot:
        modal = PlanCommentModal(plan)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#plan_comment_line_number", Input).value = "2"
        modal.query_one("#plan_comment_text", Input).value = "use Alembic instead"
        dismissed: list[PlanComment | None] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)

        modal.on_button_pressed(Button.Pressed(Button("Add comment", id="submit")))

        assert dismissed == [
            PlanComment(
                line_index=2,
                line_text="Wire it into the request pipeline.",
                comment="use Alembic instead",
            )
        ]


async def test_valid_submission_preserves_the_comments_raw_unstripped_text(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a comment typed with deliberate leading and trailing
    whitespace, when it is submitted, then `PlanComment.comment` carries
    that whitespace verbatim -- only the emptiness check strips it, not
    the value that actually reaches the model."""
    plan = _default_plan()
    async with kestrel_app_factory().run_test() as pilot:
        modal = PlanCommentModal(plan)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#plan_comment_line_number", Input).value = "1"
        modal.query_one("#plan_comment_text", Input).value = "  quote: use Alembic  "
        dismissed: list[PlanComment | None] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)

        modal.on_button_pressed(Button.Pressed(Button("Add comment", id="submit")))

        assert dismissed == [
            PlanComment(
                line_index=1,
                line_text="Add an authentication middleware module.",
                comment="  quote: use Alembic  ",
            )
        ]


@pytest.mark.parametrize(
    ("line_number", "comment", "expected_fragment"),
    [
        ("not-a-number", "a real comment", "must be an integer"),
        ("99", "a real comment", "no plan line 99"),
        ("1", "   ", "must not be empty"),
    ],
    ids=["non_integer_line_number", "out_of_range_line_number", "empty_comment"],
)
async def test_invalid_submission_notifies_an_error_and_never_dismisses(
    line_number: str,
    comment: str,
    expected_fragment: str,
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal mounted against a real app, when `submit` is
    pressed with `line_number`/`comment` carrying one of three invalid
    shapes, then an error-severity notification carrying
    `expected_fragment` fires exactly once and `dismiss` is never
    called."""
    plan = _default_plan()
    async with kestrel_app_factory().run_test() as pilot:
        modal = PlanCommentModal(plan)
        await pilot.app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#plan_comment_line_number", Input).value = line_number
        modal.query_one("#plan_comment_text", Input).value = comment
        dismissed: list[PlanComment | None] = []
        monkeypatch.setattr(modal, "dismiss", dismissed.append)
        notifications: list[tuple[str, str]] = []

        def _spy_notify(
            message: str, *, severity: str = "information", **_: object
        ) -> None:
            """Record `message`/`severity` verbatim instead of the real
            `App.notify`, which shows a toast this test does not need
            to render."""
            notifications.append((message, severity))

        monkeypatch.setattr(pilot.app, "notify", _spy_notify)

        modal.on_button_pressed(Button.Pressed(Button("Add comment", id="submit")))

        assert dismissed == []
        assert len(notifications) == 1
        message, severity = notifications[0]
        assert severity == "error"
        assert expected_fragment in message


@pytest.mark.redteam
def test_hostile_plan_line_renders_inertly_in_the_documented_static() -> None:
    """Given a plan whose one line carries the injection corpus's
    `ansi_escape_laden_payload` verbatim, when the modal composes, then
    `#plan_comment_lines` still matches `sanitize_terminal(
    render_plan_markdown(plan))` exactly, and no raw ANSI/CSI/OSC escape
    byte survives into the rendered text."""
    case = _find_case(_HOSTILE_CASE_ID)
    plan = _plan(lines=(PlanLine(index=1, text=case.payload),))

    widgets = _widgets_by_id(plan)
    lines_widget = widgets["plan_comment_lines"]
    assert isinstance(lines_widget, Static)
    content = lines_widget.content
    assert isinstance(content, str)

    assert content == sanitize_terminal(render_plan_markdown(plan))
    assert "\x1b" not in content
    assert "\x9b" not in content
    assert "\x07" not in content


@pytest.mark.ui
async def test_a_long_plan_scrolls_inside_a_focusable_container(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given a plan with far more lines than fit the dialog's own
    capped height, when the modal is pushed onto a running app, then
    `#plan_comment_lines_scroll` is a focusable `VerticalScroll` whose
    content genuinely overflows it -- so every line, not just the ones
    that happen to fit on screen, is reachable by scrolling -- and
    `#plan_comment_buttons` still lands within the screen's own visible
    rows rather than being pushed off the bottom."""
    plan = _plan(
        lines=tuple(
            PlanLine(index=i, text=f"Step {i}: do the thing.") for i in range(1, 201)
        )
    )

    async with kestrel_app_factory().run_test(size=(80, 24)) as pilot:
        await pilot.app.push_screen(PlanCommentModal(plan))
        await pilot.pause()
        await pilot.pause()

        scroll = pilot.app.screen.query_one(
            "#plan_comment_lines_scroll", VerticalScroll
        )
        assert scroll.can_focus
        assert scroll.max_scroll_y > 0

        buttons = pilot.app.screen.query_one("#plan_comment_buttons")
        screen_height = pilot.app.screen.size.height
        assert buttons.region.y + buttons.region.height <= screen_height
