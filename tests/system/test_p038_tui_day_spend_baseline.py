"""System test: the status bar's day-spend figure seeds from a repo's
own real session history, not zero.

`TuiLoopObserver.spent_day_usd_baseline` used to be hardcoded to zero
on every submission, so the status bar's `day` segment only ever showed
the current task's own spend -- silently dropping whatever a different
task had already spent earlier the same day, and drifting from the
same-day total `LoopDeps.budget` itself checks against. This suite
seeds a prior task's own session journal directly, submits a new task
through the cockpit, and confirms the shown day figure is the prior
spend plus the new task's own running total, not the new total alone.

No tool calls are scripted here, so this suite needs no sandbox and
carries no `bwrap` skip, unlike
`tests/system/test_p038_tui_conversation_stream.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.cost.meter import TurnCost
from kestrel.managers.session import SessionManager, TurnRecord
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp, StatusBar

pytestmark = [pytest.mark.p038, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_MODEL_ID = "glm-5.2"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassette's own `model` field."""
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


def _seed_prior_spend(repo_root: Path, *, usd: Decimal) -> None:
    """Journal one turn of spend for a different task, timestamped now,
    so `aggregate_historical_spend`'s own day window picks it up."""
    session = SessionManager(repo_root=repo_root, task_id="prior-task")
    session.record_turn(
        TurnRecord(
            turn_id=1,
            task_id="prior-task",
            timestamp=time.time(),
            message_deltas=(),
            turn_cost=TurnCost(
                model_id=_MODEL_ID,
                input_tokens=1_000,
                output_tokens=200,
                cached_tokens=0,
                usd=usd,
            ),
            verification=None,
        )
    )


async def test_status_bar_day_spend_seeds_from_the_historical_baseline(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a repo whose session journal already recorded spend for a
    different task earlier today, when a new task is submitted and
    completes, then the status bar's day figure is that prior spend
    plus the new task's own total -- not the new task's total alone."""
    _seed_prior_spend(tmp_path, usd=Decimal("1.5000"))

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(_DONE_CASSETTE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "say hello"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        status_bar = pilot.app.query_one("#status_bar", StatusBar)
        final_status = str(status_bar.render())

        # done_no_more_tools.sse bills 70 input + 5 output tokens against
        # this registry's own rates: (70 * 0.60 + 5 * 2.20) / 1e6 =
        # 0.000053, so the day total is 1.5000 + 0.000053 = 1.500053,
        # which the status line's own :.4f formatting rounds to 1.5001.
        assert "day $1.5001" in final_status
