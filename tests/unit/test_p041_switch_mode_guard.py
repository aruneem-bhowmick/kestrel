"""Tests for `action_switch_model`/`action_set_mode` declining while a
task is active, rather than mutating `active_model_id`/`mode_manager`
(and the status bar) underneath a task whose own `TuiLoopObserver`
reads both live on every turn refresh.

Simulates "a task is active" by writing `_current_task_id` directly,
the same pattern `test_p038_task_submission_guard.py` and
`test_p041_undo_last_completed.py` already use, rather than running a
real task -- keeping this suite fast and network-free.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p041, pytest.mark.ui]


def _model_entry(model_id: str) -> ModelEntry:
    """A minimal, cheap OpenRouter-routed `ModelEntry` for `model_id`."""
    return ModelEntry(
        id=model_id,
        backend="openrouter",
        provider_model=f"z-ai/{model_id}",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )


def _app(tmp_path: Path, *model_ids: str) -> KestrelApp:
    """A `KestrelApp` rooted at `tmp_path`, registered with one entry
    per id in `model_ids` and starting active on the first one."""
    registry = Registry(
        models={model_id: _model_entry(model_id) for model_id in model_ids},
        source=None,
    )
    return KestrelApp(
        config=KestrelConfig(),
        registry=registry,
        model_id=model_ids[0],
        kestrel_md=None,
        repo_root=tmp_path,
    )


@pytest.mark.sanity
async def test_switch_model_declines_while_a_task_is_active(tmp_path: Path) -> None:
    """Given a task currently running, when `/model` is selected for a
    different id, then `active_model_id` is left untouched and a busy
    warning fires instead."""
    app = _app(tmp_path, "glm-5.2", "glm-5.2-mini")
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        pilot.app._current_task_id = "running-task"
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )

        pilot.app.action_switch_model("glm-5.2-mini")

        assert pilot.app.active_model_id == "glm-5.2"
        assert notified == [
            ("a task is still running -- switch once it finishes", "warning")
        ]


@pytest.mark.sanity
async def test_set_mode_declines_while_a_task_is_active(tmp_path: Path) -> None:
    """Given a task currently running, when `/mode plan` is selected,
    then `mode_manager.mode` is left untouched and a busy warning fires
    instead."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)
        pilot.app._current_task_id = "running-task"
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )

        pilot.app.action_set_mode("plan")

        assert pilot.app.mode_manager.mode == "fast"
        assert notified == [
            ("a task is still running -- switch once it finishes", "warning")
        ]


async def test_switch_model_and_set_mode_still_work_while_idle(tmp_path: Path) -> None:
    """Given no task is active, when `/model` and `/mode plan` are
    selected, then both apply immediately -- the busy guard changes
    nothing about the idle path."""
    app = _app(tmp_path, "glm-5.2", "glm-5.2-mini")
    async with app.run_test() as pilot:
        assert isinstance(pilot.app, KestrelApp)

        pilot.app.action_switch_model("glm-5.2-mini")
        pilot.app.action_set_mode("plan")

        assert pilot.app.active_model_id == "glm-5.2-mini"
        assert pilot.app.mode_manager.mode == "plan"
