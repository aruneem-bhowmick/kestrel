"""Unit test for `KestrelApp._show_plan_from_result`'s failure path: a
task that did not end on a plain assistant message surfaces as a
warning notification instead of crashing the worker, and never touches
this session's own plan state -- the one branch
`test_p050_tui_plan_submission.py`'s own end-to-end scenario never
takes, since its own scripted task always ends on a plain reply.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p050, pytest.mark.unit, pytest.mark.sanity]

_TASK_ID = "task-p050-error"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry, standing in for
    whatever registry a real `KestrelApp` would have already loaded."""
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
    return Registry(models={"glm-5.2": entry}, source=None)


async def test_a_turn_cap_result_notifies_a_warning_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """Given a `LoopResult` that ended `TURN_CAP` mid tool-call -- no
    plain assistant message to parse a plan from -- when
    `_show_plan_from_result` runs, then `notify` is called once with
    `severity="warning"`, and `_last_plan`/`_plan_task_id` are left
    exactly as they started, since `extract_plan_from_result`'s own
    `PlanError` fires before either is ever set."""
    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )
    notifications: list[tuple[str, str]] = []

    def _spy_notify(
        message: str, *, severity: str = "information", **_: object
    ) -> None:
        """Record `message`/`severity` verbatim instead of the real
        `App.notify`, which needs a mounted app to actually display
        anything."""
        notifications.append((message, severity))

    app.notify = _spy_notify  # type: ignore[method-assign]
    result = LoopResult(
        reason=TerminationReason.TURN_CAP,
        turns_used=1,
        total_usd=Decimal("0"),
        history=(),
    )

    await app._show_plan_from_result(result, _TASK_ID)

    assert len(notifications) == 1
    message, severity = notifications[0]
    assert severity == "warning"
    assert _TASK_ID in message
    assert app._last_plan is None
    assert app._plan_task_id is None
