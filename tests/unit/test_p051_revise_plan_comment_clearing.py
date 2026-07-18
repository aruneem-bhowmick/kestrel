"""Unit test for a `KestrelApp._revise_plan` correctness fix:
`_pending_plan_comments` is cleared the instant `revise_plan` itself
returns, before `_show_plan_from_result` ever touches the reply -- so
an already-submitted batch of comments is never left queued (and later
resubmitted a second time) just because the revised reply itself
failed to parse into a plan.

Bypasses the real agent loop and mock server: stubs the module-level
`revise_plan` function `_revise_plan` calls, returning a hand-built
`LoopResult` engineered to raise `PlanError`.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.agent.plan import PlanComment
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p051, pytest.mark.unit, pytest.mark.sanity]

_PLAN_TASK_ID = "plan-task-p051-comment-clearing"


def _queue_one_comment(app: KestrelApp) -> None:
    """Put `app` into the exact state `on_input_submitted`'s own
    `revising` condition requires: PLAN mode, a tracked plan task, and
    one queued comment against it."""
    app.mode_manager.set_mode("plan")
    app._plan_task_id = _PLAN_TASK_ID
    app._pending_plan_comments = [
        PlanComment(line_index=1, line_text="do the thing", comment="use Alembic")
    ]


async def test_pending_comments_clear_even_when_the_revised_reply_is_a_plan_error(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `revise_plan` returns a `LoopResult` that
    `extract_plan_from_result` cannot parse (ended `TURN_CAP` mid
    tool-call, no plain final reply), when `_revise_plan` runs, then
    `_pending_plan_comments` is empty afterward regardless -- the
    queued comments were already sent to the model the instant
    `revise_plan` returned, so leaving them queued past that point
    would resend the exact same batch on the next resubmission."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        _queue_one_comment(app)

        async def _fake_revise_plan(
            task_id: str, deps: object, comments: object
        ) -> LoopResult:
            """Stand in for a real revision turn that ends without a
            plain assistant reply to parse."""
            return LoopResult(
                reason=TerminationReason.TURN_CAP,
                turns_used=1,
                total_usd=Decimal("0"),
                history=(),
            )

        monkeypatch.setattr("kestrel.tui.app.revise_plan", _fake_revise_plan)
        notifications: list[tuple[str, str]] = []

        def _spy_notify(
            message: str, *, severity: str = "information", **_: object
        ) -> None:
            notifications.append((message, severity))

        monkeypatch.setattr(app, "notify", _spy_notify)

        await app._revise_plan()

        assert app._pending_plan_comments == []
        assert app._plan_task_id == _PLAN_TASK_ID
        assert app._last_plan is None
        assert len(notifications) == 1
        assert notifications[0][1] == "warning"
