"""Unit tests for `kestrel.tools.registry.schemas_for`: the name-filtered
view over the registered tool schemas that a per-task tool allowlist is
built from, plus a cost-regression pin proving the agent loop's own new
`effort`/`available_tools` fields change nothing when left at their
defaults.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import (
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.read_file import READ_FILE_SCHEMA
from kestrel.tools.registry import all_schemas, schemas_for
from kestrel.tools.search import SEARCH_SCHEMA

pytestmark = [pytest.mark.p045, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


@pytest.mark.sanity
def test_schemas_for_none_matches_all_schemas() -> None:
    """Given `None`, when `schemas_for` is called, then it returns
    exactly `all_schemas()` -- the unfiltered case delegates to that
    function rather than reimplementing its own copy of the fixed
    order."""
    assert schemas_for(None) == all_schemas()


@pytest.mark.sanity
def test_schemas_for_a_name_subset_returns_only_those_schemas_in_fixed_order() -> None:
    """Given a subset of registered tool names, when `schemas_for` is
    called, then it returns exactly those schemas, in `_TOOLS`'s own
    fixed order -- `read_file` before `search` -- rather than the order
    the names were passed in."""
    result = schemas_for(frozenset({"search", "read_file"}))

    assert result == (READ_FILE_SCHEMA, SEARCH_SCHEMA)


@pytest.mark.sanity
def test_schemas_for_an_empty_set_returns_no_schemas() -> None:
    """Given an empty `frozenset`, when `schemas_for` is called, then it
    returns an empty tuple rather than falling back to every tool."""
    assert schemas_for(frozenset()) == ()


def _registry() -> Registry:
    """Build a single-entry `Registry` at the same rates the packaged
    default registry ships, matching `test_p022_loop.py`'s own
    cost-regression fixture so this test's pinned total is directly
    comparable to that one."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend=_BACKEND,
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


@dataclass
class _ScriptedTurn:
    """One scripted `.complete()` call's outcome -- the events to yield."""

    events: tuple[StreamEvent, ...] = ()


@dataclass
class _ScriptedLoopClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order, ignoring every argument but the count."""

    turns: Sequence[_ScriptedTurn]
    call_count: int = field(default=0, init=False)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted turn's events."""
        turn = self.turns[self.call_count]
        self.call_count += 1
        for event in turn.events:
            yield event


@pytest.mark.cost_regression
async def test_new_fields_add_no_extra_cost(tmp_path: Path) -> None:
    """Given the exact two-turn scripted task `test_p022_loop.py`'s own
    `test_two_turn_task_cost_band` pins, but built with `effort` and
    `available_tools` explicitly set to their defaults, when the task
    runs, then it prices identically to that pinned band -- proving the
    two new `LoopDeps` fields change nothing about a turn's cost when
    left unset."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _ScriptedTurn(
                events=(
                    ToolCallEvent(
                        id="call-1",
                        name="read_file",
                        arguments_json=json.dumps({"path": "greet.py"}),
                    ),
                    UsageEvent(input_tokens=1000, output_tokens=50, cached_tokens=0),
                    StopEvent(reason="tool_use"),
                )
            ),
            _ScriptedTurn(
                events=(
                    TextDelta(text="done"),
                    UsageEvent(input_tokens=1200, output_tokens=20, cached_tokens=0),
                    StopEvent(reason="end_turn"),
                )
            ),
        ]
    )
    deps = LoopDeps(
        client=client,
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        effort="high",
        available_tools=None,
    )

    result = await run_task("read greet.py", deps, task_id="t-p045-cost")

    # turn 1: (1000 * 0.60 + 50 * 2.20) / 1e6 = 0.000710
    # turn 2: (1200 * 0.60 + 20 * 2.20) / 1e6 = 0.000764
    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.total_usd == Decimal("0.001474")
