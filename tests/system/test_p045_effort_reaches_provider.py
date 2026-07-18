"""System test: a task's own `LoopDeps.effort` setting genuinely reaches
every outgoing model call `run_task` makes, rather than the effort level
that was hardcoded before `LoopDeps.effort` existed.

Uses a scripted fake `ProviderClient` recording its own `effort`
argument on every call, matching `test_p022_loop.py`'s own scripted-
client pattern -- what is under test here is the value the loop's own
`_drain_think` passes through, not a real backend's request body (see
`test_p044_effort_openrouter.py`/`test_p044_effort_zai.py` for that).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
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
from kestrel.tools.registry import ToolResult

pytestmark = [pytest.mark.p045, pytest.mark.system]

_MODEL_ID = "glm-5.2"


def _registry() -> Registry:
    """A single-entry `Registry`, matching this suite's siblings."""
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


@dataclass
class _ScriptedTurn:
    """One scripted `.complete()` call's outcome -- the events to yield."""

    events: tuple[StreamEvent, ...] = ()


@dataclass
class _EffortRecordingClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order, recording the `effort` argument it was actually called with
    on every single call."""

    turns: Sequence[_ScriptedTurn]
    recorded_efforts: list[Effort] = field(default_factory=list)
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
        """Record `effort`, then replay the next scripted turn's events."""
        self.recorded_efforts.append(effort)
        turn = self.turns[self.call_count]
        self.call_count += 1
        for event in turn.events:
            yield event


async def test_effort_max_reaches_every_outgoing_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `LoopDeps.effort="max"` and a two-turn scripted task (a
    tool call, then a stop), when the task runs, then every one of the
    client's `.complete()` calls is made with `effort="max"` -- not the
    `"high"` value every turn was hardcoded to before this field
    existed."""

    def _fake_dispatch(
        event: object, *, repo_root: Path, **context: object
    ) -> ToolResult:
        call_id = getattr(event, "id", "unknown")
        return ToolResult(tool_call_id=call_id, content="ran it")

    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)

    client = _EffortRecordingClient(
        turns=[
            _ScriptedTurn(
                events=(
                    ToolCallEvent(
                        id="call-1", name="read_file", arguments_json=json.dumps({})
                    ),
                    UsageEvent(input_tokens=100, output_tokens=20, cached_tokens=0),
                    StopEvent(reason="tool_use"),
                )
            ),
            _ScriptedTurn(
                events=(
                    TextDelta(text="done"),
                    UsageEvent(input_tokens=100, output_tokens=20, cached_tokens=0),
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
        effort="max",
    )

    result = await run_task("do something", deps, task_id="sys-p045-effort")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 2
    assert client.recorded_efforts == ["max", "max"]
