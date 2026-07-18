"""Unit test for a `KestrelApp.on_input_submitted` correctness fix: a
revising submission reserves `_current_task_id` synchronously inside
the handler itself, before `run_worker` ever schedules `_revise_plan`'s
own coroutine -- the same fix `action_resume_task` already carries for
a resumed task, since `run_worker` merely schedules a coroutine onto
the event loop rather than running it immediately, leaving a window in
which a second submission could otherwise race it.

Bypasses the real agent loop and mock server entirely: stubs
`KestrelApp._revise_plan` and drives `on_input_submitted` directly.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual.widgets import Input

from kestrel.agent.plan import PlanComment
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p051, pytest.mark.unit, pytest.mark.sanity]

_PLAN_TASK_ID = "plan-task-p051-wiring"


def _queue_one_comment(app: KestrelApp) -> None:
    """Put `app` into the exact state `on_input_submitted`'s own
    `revising` condition requires: PLAN mode, a tracked plan task, and
    one queued comment against it."""
    app.mode_manager.set_mode("plan")
    app._plan_task_id = _PLAN_TASK_ID
    app._pending_plan_comments = [
        PlanComment(line_index=1, line_text="do the thing", comment="use Alembic")
    ]


async def test_revising_submission_reserves_current_task_id_before_scheduling(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a revision is pending, when `on_input_submitted` is
    awaited directly -- bypassing Textual's own key-dispatch pump, so
    `_revise_plan`'s scheduled coroutine gets no chance to run before
    this assertion -- then `_current_task_id` already names the plan's
    own task the instant the handler returns.

    This is what proves the reservation happens synchronously inside
    the handler itself rather than as `_revise_plan`'s own first
    statement: `on_input_submitted` has no `await` before it calls
    `run_worker`, so a direct `await app.on_input_submitted(event)`
    runs the whole handler body -- including the reservation -- in one
    uninterrupted stretch, with no opportunity for the scheduled worker
    to run first."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        _queue_one_comment(app)

        ran: list[None] = []

        async def _fake_revise_plan() -> None:
            """Record that the worker eventually ran; irrelevant to
            this test's own assertion, which checks state before this
            coroutine is ever scheduled to run."""
            ran.append(None)

        monkeypatch.setattr(app, "_revise_plan", _fake_revise_plan)
        task_input = app.query_one("#task_input", Input)

        await app.on_input_submitted(Input.Submitted(input=task_input, value=""))

        assert app._current_task_id == _PLAN_TASK_ID

        await pilot.app.workers.wait_for_complete()
        assert ran == [None]
