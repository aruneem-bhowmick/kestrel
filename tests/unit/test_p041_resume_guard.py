"""Tests for `action_resume_task`'s synchronous `_current_task_id`
reservation: declining while another task is active, reserving before
`run_worker` is even called (so a second selection made before the
first worker actually starts still sees the reservation), and rolling
the reservation back if `run_worker` itself fails to schedule the
worker.

`_resume_task` itself is stubbed out via `monkeypatch` in every case
that would otherwise schedule it, the same pattern
`test_p038_task_submission_guard.py` uses for `_run_task` -- these
cases are about the reservation around the worker, not the resumed
task's own agent-loop behavior, so nothing here needs a real session
journal, a mock model server, or the network.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

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
async def test_resume_task_declines_when_a_task_is_already_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a task already active, when `/resume` is selected for a
    different task id, then it declines with a busy warning, never
    schedules `_resume_task`, and leaves `_current_task_id` naming the
    original task."""
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        pilot.app._current_task_id = "already-running"
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )
        calls: list[str] = []

        async def _fake_resume_task(task_id: str) -> None:
            """Record the id instead of driving a real resume."""
            calls.append(task_id)

        monkeypatch.setattr(pilot.app, "_resume_task", _fake_resume_task)

        pilot.app.action_resume_task("other-task")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert calls == []
        assert pilot.app._current_task_id == "already-running"
        assert notified == [
            ("a task is still running -- resume once it finishes", "warning")
        ]


async def test_resume_task_reserves_current_task_id_before_the_worker_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no task is active, when `/resume` is selected, then
    `_current_task_id` already names the resumed task immediately after
    `action_resume_task` returns -- before `_resume_task`'s own
    coroutine has had any chance to run, since `run_worker` only
    schedules it -- and `_resume_task` itself still runs once the event
    loop gets a turn."""
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        assert pilot.app._current_task_id is None
        calls: list[str] = []

        async def _fake_resume_task(task_id: str) -> None:
            """Record the id instead of driving a real resume."""
            calls.append(task_id)

        monkeypatch.setattr(pilot.app, "_resume_task", _fake_resume_task)

        pilot.app.action_resume_task("task-xyz")

        assert pilot.app._current_task_id == "task-xyz"
        assert calls == []

        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert calls == ["task-xyz"]


async def test_resume_task_rolls_back_the_reservation_when_run_worker_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `run_worker` itself raises before scheduling the worker,
    when `/resume` is selected, then the exception propagates and the
    synchronous `_current_task_id` reservation is rolled back to `None`
    rather than left permanently set, which would otherwise block every
    later submission and resume attempt for the rest of the session."""
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)

        def _boom(work: object, *args: object, **kwargs: object) -> None:
            """Raise before scheduling `work`, closing the coroutine
            `run_worker` would otherwise have consumed so it is not
            left dangling ("coroutine was never awaited") -- the same
            cleanup a real failed scheduling attempt would need."""
            work.close()  # type: ignore[attr-defined]
            raise RuntimeError("worker startup failed")

        monkeypatch.setattr(pilot.app, "run_worker", _boom)

        with pytest.raises(RuntimeError, match="worker startup failed"):
            pilot.app.action_resume_task("task-xyz")

        assert pilot.app._current_task_id is None
