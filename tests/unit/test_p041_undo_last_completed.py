"""Tests for `/undo`'s corrected target: the most recently *finished*
task, not whatever `_current_task_id` happens to name.

`_current_task_id` is `None` both before any task has ever run and
immediately after one finishes -- the second case is exactly when a
user would want to undo it, so `action_undo_current_task` must not
treat the two as identical. These cases drive the action methods
directly against a mounted `KestrelApp`, simulating "a task already
finished" or "a task is still running" by writing to `_current_task_id`
and `_last_completed_task_id` rather than running a real task, keeping
this suite fast and network-free -- the same pattern
`test_p038_task_submission_guard.py` already established for
`_current_task_id`.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import KestrelConfig
from kestrel.managers.undo import UndoEntry, UndoManager
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import ConversationPane, KestrelApp

pytestmark = [pytest.mark.p041, pytest.mark.ui]


def _app(tmp_path: Path) -> KestrelApp:
    """A minimally configured `KestrelApp` rooted at `tmp_path`."""
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return KestrelApp(
        config=KestrelConfig(),
        registry=Registry(models={"glm-5.2": entry}, source=None),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )


@pytest.mark.sanity
async def test_undo_reverts_the_last_completed_task_not_an_active_one(
    tmp_path: Path,
) -> None:
    """Given a finished task with one journaled mutation, and no task
    currently active, when `/undo` runs, then it reverts that finished
    task's own mutation via `_last_completed_task_id` -- proving undo
    no longer requires `_current_task_id` (which is `None` for any
    finished task by definition) to be set."""
    target = tmp_path / "greeting.txt"
    target.write_text("after", encoding="utf-8")
    UndoManager(repo_root=tmp_path).record(
        UndoEntry(
            turn_id=0,
            task_id="finished-task",
            path="greeting.txt",
            before="before",
            after="after",
        )
    )

    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        pilot.app._last_completed_task_id = "finished-task"

        pilot.app.action_undo_current_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert target.read_text(encoding="utf-8") == "before"
        conversation = pilot.app.query_one("#conversation", ConversationPane)
        lines = [strip.text for strip in conversation.lines]
        assert any(
            "reverted 1 mutation(s) for task finished-task" in line for line in lines
        )


async def test_undo_declines_with_a_busy_warning_while_a_task_is_active(
    tmp_path: Path,
) -> None:
    """Given a task currently running, and a different, already
    finished task with its own mutation available, when `/undo` runs,
    then it declines with a busy warning and touches nothing -- undo
    while a task is active would race that task's own tool calls."""
    target = tmp_path / "greeting.txt"
    target.write_text("after", encoding="utf-8")
    UndoManager(repo_root=tmp_path).record(
        UndoEntry(
            turn_id=0,
            task_id="finished-task",
            path="greeting.txt",
            before="before",
            after="after",
        )
    )

    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        pilot.app._current_task_id = "running-task"
        pilot.app._last_completed_task_id = "finished-task"
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )

        pilot.app.action_undo_current_task()
        await pilot.pause()

        assert notified == [
            ("a task is still running -- undo once it finishes", "warning")
        ]
        assert target.read_text(encoding="utf-8") == "after"


async def test_undo_still_warns_when_nothing_has_finished_yet(tmp_path: Path) -> None:
    """Given no task has ever run this session, when `/undo` runs, then
    it warns that there is nothing to undo -- unchanged from before the
    `_last_completed_task_id` split."""
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )

        pilot.app.action_undo_current_task()
        await pilot.pause()

        assert notified == [("no task to undo yet", "warning")]
