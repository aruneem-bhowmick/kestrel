"""Unit tests for `build_task_deps`'s `mode_manager` parameter: a
PLAN-mode manager narrows effort, tools, and verification the same way
regardless of what the caller itself asked for, a FAST-mode manager
changes only the effort level, and leaving `mode_manager` unset (the
default) reproduces `test_p038_task_setup.py`'s own "defaults" shape
byte-for-byte -- proving this parameter's default is a genuine no-op,
not merely an untested claim in its docstring.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopLimits
from kestrel.agent.observer import NULL_OBSERVER
from kestrel.config import KestrelConfig
from kestrel.managers.budget import BudgetLimits, BudgetStatus
from kestrel.managers.mode import ModeManager
from kestrel.managers.session import SessionManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry
from kestrel.task_setup import TaskSetup, build_task_deps

pytestmark = [pytest.mark.p049, pytest.mark.unit, pytest.mark.sanity]

_MODEL_ID = "glm-5.2"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry, standing in for
    whatever registry a real caller would have already loaded."""
    entry = ModelEntry(
        id=_MODEL_ID,
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
    return Registry(models={_MODEL_ID: entry}, source=None)


def test_plan_mode_manager_narrows_effort_tools_and_verification(
    tmp_path: Path,
) -> None:
    """Given a `ModeManager(mode="plan")` and `require_verification=True`
    explicitly passed, when a task's deps are built, then `deps.effort`
    is `"max"`, `deps.available_tools` is exactly `{"read_file",
    "search"}`, and `deps.require_verification` is forced `False` --
    PLAN mode overrides the caller's own verification request, since a
    PLAN-mode task is never offered `verify`."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        require_verification=True,
        mode_manager=ModeManager(mode="plan"),
    )

    assert setup.deps.effort == "max"
    assert setup.deps.available_tools == frozenset({"read_file", "search"})
    assert setup.deps.require_verification is False


def test_fast_mode_manager_sets_high_effort_and_leaves_tools_and_verification(
    tmp_path: Path,
) -> None:
    """Given a `ModeManager(mode="fast")`, when a task's deps are built,
    then `deps.effort` is `"high"`, `deps.available_tools` is `None`
    (every tool remains offered), and `deps.require_verification` is
    exactly whatever the caller passed."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        require_verification=True,
        mode_manager=ModeManager(mode="fast"),
    )

    assert setup.deps.effort == "high"
    assert setup.deps.available_tools is None
    assert setup.deps.require_verification is True


def test_fast_mode_manager_leaves_require_verification_false_when_unset(
    tmp_path: Path,
) -> None:
    """Given a `ModeManager(mode="fast")` and `require_verification` left
    at its own default, when a task's deps are built, then
    `deps.require_verification` is `False`, exactly as passed."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        mode_manager=ModeManager(mode="fast"),
    )

    assert setup.deps.require_verification is False


@pytest.mark.sanity
def test_no_mode_manager_reproduces_the_p038_defaults_shape(tmp_path: Path) -> None:
    """Given `mode_manager` left at its default (`None`), when a task's
    deps are built, then the returned bundle matches
    `test_p038_task_setup.py`'s own "defaults" assertions exactly --
    a regression pin proving this parameter's default never changes an
    existing caller's behavior."""
    registry = _registry()
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=registry,
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert isinstance(setup, TaskSetup)
    assert setup.deps.require_verification is False
    assert setup.deps.observer is NULL_OBSERVER
    assert setup.deps.model_id == _MODEL_ID
    assert setup.deps.repo_root == tmp_path
    assert setup.deps.kestrel_md is None
    assert setup.deps.limits == LoopLimits()
    assert setup.deps.effort == "high"
    assert setup.deps.available_tools is None
    assert isinstance(setup.deps.client, LiteLLMClient)
    assert isinstance(setup.deps.session, SessionManager)
    assert setup.deps.meter is setup.meter
    assert setup.deps.undo is setup.undo
    assert setup.deps.meter.turns == ()
    assert setup.budget_limits == BudgetLimits()
    assert setup.spent_day_usd == Decimal(0)
    assert setup.spent_month_usd == Decimal(0)

    event = setup.deps.budget.check(  # type: ignore[union-attr]
        spent_session=Decimal("999999"),
        spent_day=Decimal("999999"),
        spent_month=Decimal("999999"),
    )
    assert event.status is BudgetStatus.OK
