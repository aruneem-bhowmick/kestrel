"""Unit tests for `KestrelApp.action_comment_on_plan`'s two new guards:
it declines while a task is active (see `_reject_while_task_active`,
the same guard `action_switch_model`/`action_set_mode`/`action_resume_task`
already carry) and while the cockpit's own mode is not `"plan"` -- since
`_pending_plan_comments` is only ever drained by a `"plan"`-mode
resubmission, queuing a comment while some other mode is active would
leave it stranded until mode switches back, then surface unexpectedly
against whatever plan happens to be current by then.

The pre-existing "no plan yet" decline and the happy-path modal-push
are both covered here too, proving neither regressed.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from kestrel.agent.plan import ImplementationPlan, PlanLine
from kestrel.tui.app import KestrelApp
from kestrel.tui.plan_comment_modal import PlanCommentModal

pytestmark = [pytest.mark.p051, pytest.mark.unit, pytest.mark.sanity]

_TASK_ID = "task-p051-guard"


def _plan() -> ImplementationPlan:
    """A minimal, single-line plan standing in for a real parsed
    PLAN-mode reply."""
    return ImplementationPlan(
        task_id=_TASK_ID,
        raw_text="1. Add auth middleware.",
        lines=(PlanLine(index=1, text="Add auth middleware."),),
    )


def _spy_notify(app: KestrelApp) -> list[tuple[str, str]]:
    """Replace `app.notify` with a spy recording every call's own
    message/severity, returning the list it appends to."""
    notifications: list[tuple[str, str]] = []
    app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
        notifications.append((message, severity))
    )
    return notifications


def _spy_push_screen(app: KestrelApp) -> list[object]:
    """Replace `app.push_screen` with a spy recording every screen it
    was asked to push, returning the list it appends to."""
    pushed: list[object] = []
    app.push_screen = lambda screen, *_a, **_kw: pushed.append(screen)  # type: ignore[method-assign]
    return pushed


async def test_declines_while_a_task_is_active(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given a task currently running and a plan on screen, when `c` is
    pressed, then no modal is pushed and a busy warning naming
    "comment" fires instead."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        app.mode_manager.set_mode("plan")
        app._last_plan = _plan()
        app._current_task_id = "running-task"
        notifications = _spy_notify(app)
        pushed = _spy_push_screen(app)

        app.action_comment_on_plan()

        assert pushed == []
        assert notifications == [
            ("a task is still running -- comment once it finishes", "warning")
        ]


async def test_declines_outside_plan_mode(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given the cockpit's mode is `"fast"` and a plan is still on
    screen from an earlier PLAN-mode task, when `c` is pressed, then no
    modal is pushed and a mode-specific warning fires instead."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        assert app.mode_manager.mode == "fast"
        app._last_plan = _plan()
        notifications = _spy_notify(app)
        pushed = _spy_push_screen(app)

        app.action_comment_on_plan()

        assert pushed == []
        assert notifications == [
            ("comments are only available in PLAN mode", "warning")
        ]


async def test_declines_when_no_plan_exists_yet(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given PLAN mode but no plan produced yet this session, when `c`
    is pressed, then no modal is pushed and the pre-existing "no plan"
    warning fires -- unchanged by the new guards."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        app.mode_manager.set_mode("plan")
        notifications = _spy_notify(app)
        pushed = _spy_push_screen(app)

        app.action_comment_on_plan()

        assert pushed == []
        assert notifications == [("no plan to comment on yet", "warning")]


async def test_opens_the_modal_when_idle_in_plan_mode_with_a_plan(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given PLAN mode, no task active, and a plan on screen, when `c`
    is pressed, then `PlanCommentModal` is pushed against that exact
    plan and nothing is notified."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        app.mode_manager.set_mode("plan")
        plan = _plan()
        app._last_plan = plan
        notifications = _spy_notify(app)
        pushed = _spy_push_screen(app)

        app.action_comment_on_plan()

        assert notifications == []
        assert len(pushed) == 1
        modal = pushed[0]
        assert isinstance(modal, PlanCommentModal)
        assert modal._plan is plan
