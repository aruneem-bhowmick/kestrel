"""Unit tests for the agent loop's own compaction wiring: a pre-check
that folds `history` via `kestrel.agent.compaction.compact_history`
once the most recently recorded turn's own input tokens cross 70% of
the active model's context window, before the next real Think call is
ever made.

Reuses `test_p022_loop.py`'s scripted-`ProviderClient` pattern and
`test_p029_loop_session.py`'s monkeypatched-`dispatch` pattern rather
than a live mock server, since what is under test here is the loop's
own control flow around a fold, not a real model or tool call --
`tests/system/test_p032_compaction_scripted.py` covers that instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.compaction as compaction_module
import kestrel.agent.loop as loop_module
from kestrel.agent.loop import LoopDeps, LoopLimits, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.budget import BudgetLimits, BudgetManager
from kestrel.managers.session import SessionManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.errors import ContextOverflowError
from kestrel.provider.events import (
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.registry import ToolResult

pytestmark = [pytest.mark.p032, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry(*, context_window: int) -> Registry:
    """A single-entry `Registry` with round, easy-to-hand-verify rates
    and a caller-chosen `context_window` small enough for this suite's
    own scripted token counts to cross the compaction threshold."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend=_BACKEND,
        provider_model=f"z-ai/{_MODEL_ID}",
        api_key_env="OPENROUTER_API_KEY",
        context_window=context_window,
        max_output=16_384,
        usd_per_mtok_input=Decimal("1.00"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={_MODEL_ID: entry}, source=None)


@dataclass
class _ScriptedTurn:
    """One scripted `.complete()` call's outcome: the events to yield,
    or -- when `raises` is set instead -- an exception to raise without
    yielding anything, standing in for a failure mid-turn."""

    events: tuple[StreamEvent, ...] = ()
    raises: Exception | None = None


@dataclass
class _ScriptedLoopClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order, recording every call's own `messages` -- a compaction call is,
    from this client's perspective, just another `.complete()` call, so
    this suite tells them apart by position and by the messages a call
    actually carried."""

    turns: Sequence[_ScriptedTurn]
    call_count: int = field(default=0, init=False)
    received_messages: list[list[Message]] = field(default_factory=list, init=False)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted turn, recording the messages it was
        actually called with."""
        turn = self.turns[self.call_count]
        self.received_messages.append(list(messages))
        self.call_count += 1
        if turn.raises is not None:
            raise turn.raises
        for event in turn.events:
            yield event


def _tool_turn(
    call_id: str, *, input_tokens: int, output_tokens: int = 0
) -> _ScriptedTurn:
    """A turn that requests exactly one tool call, priced at
    `input_tokens`/`output_tokens` -- used here to build up `history`
    past `compact_history`'s own default kept-tail size."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(id=call_id, name="read_file", arguments_json="{}"),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _stop_turn(
    text: str, *, input_tokens: int, output_tokens: int = 0
) -> _ScriptedTurn:
    """A turn that answers with plain text and requests no tool calls --
    the same shape a real compaction summary reply or a natural task
    completion both take at the client level."""
    return _ScriptedTurn(
        events=(
            TextDelta(text=text),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="end_turn"),
        )
    )


def _fake_dispatch(
    event: ToolCallEvent, *, repo_root: Path, **context: object
) -> ToolResult:
    """Stand in for `kestrel.agent.loop.dispatch`: succeeds unconditionally
    without running any real tool."""
    return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    registry: Registry,
    *,
    budget: BudgetManager | None = None,
    session: SessionManager | None = None,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle from real managers and a fresh
    `CostMeter`, scoped to `repo_root`, for one test's `run_task` call."""
    return LoopDeps(
        client=client,
        registry=registry,
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=LoopLimits(max_total_tokens=100_000_000),
        budget=budget,
        session=session,
    )


async def test_compaction_fires_before_the_think_call_once_the_kept_tail_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two tool-calling turns building `history` past the default
    four-message kept tail, the second of which crosses 70% of a small
    test `context_window`, when the task runs, then the pre-check ahead
    of the third turn calls the client once more with a compaction-
    shaped request (the compaction system prompt leading the rest of
    the message) before that third turn's own real Think call, and the
    task still completes normally afterward."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn("carry-forward summary", input_tokens=50, output_tokens=10),
            _stop_turn("done", input_tokens=20),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(context_window=100))

    result = await run_task("do it", deps, task_id="t-compact-fires")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 4
    assert result.turns_used == 3

    compaction_call_messages = client.received_messages[2]
    assert compaction_call_messages[0] == {
        "role": "system",
        "content": compaction_module._COMPACTION_SYSTEM_PROMPT,
    }
    final_think_messages = client.received_messages[3]
    assert any(
        message.get("content") == "carry-forward summary"
        for message in final_think_messages
    )


@pytest.mark.sanity
async def test_compaction_usage_is_priced_but_never_counted_as_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the same two-turns-then-fold scenario, when the task
    completes, then the compaction call's own usage is recorded in
    `deps.meter.turns` alongside every real turn's, but
    `LoopResult.turns_used` counts only the three real Think-phase
    turns -- never the compaction call in between."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn("carry-forward summary", input_tokens=50, output_tokens=10),
            _stop_turn("done", input_tokens=20),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(context_window=100))

    result = await run_task("do it", deps, task_id="t-compact-not-a-turn")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    assert len(deps.meter.turns) == 4
    assert deps.meter.turns[2].input_tokens == 50
    assert deps.meter.turns[2].output_tokens == 10


async def test_a_compaction_call_that_crosses_the_hard_budget_cap_halts_before_the_next_think_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a hard session cap that only the compaction call's own
    priced cost pushes past, when the task runs, then it ends
    `BUDGET_HALT` right after the compaction call -- a fourth scripted
    turn, if the loop wrongly attempted one, would raise `IndexError`
    rather than being silently skipped."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn("carry-forward summary", input_tokens=50, output_tokens=10),
        ]
    )
    # Turns 1+2 alone cost $0.00013; only adding the compaction call's own
    # $0.00005 crosses this $0.00015 hard cap.
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("0.00015")))
    deps = _build_deps(client, tmp_path, _registry(context_window=100), budget=budget)

    result = await run_task("do it", deps, task_id="t-compact-hard-halt")

    assert result.reason == TerminationReason.BUDGET_HALT
    assert result.turns_used == 2
    assert client.call_count == 3
    assert len(deps.meter.turns) == 3


@pytest.mark.cost_regression
async def test_compaction_cost_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a fixed, hand-computed scenario (two tool turns, a
    compaction fold, and a closing turn, all priced at a round
    $1.00/Mtok input rate with no output or cache pricing), when the
    task completes, then its total priced cost is pinned exactly to the
    sum of all four calls' own input-token cost -- catching a change
    that silently omits the compaction call's own usage from the total,
    or double-counts it."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn("carry-forward summary", input_tokens=50, output_tokens=10),
            _stop_turn("done", input_tokens=100),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(context_window=100))

    result = await run_task("do it", deps, task_id="t-compact-cost-band")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.total_usd == Decimal("0.000280")


async def test_a_context_overflow_during_the_compaction_call_ends_the_task_that_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `ContextOverflowError` raised by the client during the
    compaction call itself -- the older tail alone was too large to
    summarize -- when the task runs, then it ends `CONTEXT_OVERFLOW`
    right there rather than letting the exception escape `run_task` or
    reaching a fourth, never-scripted Think call."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _ScriptedTurn(
                raises=ContextOverflowError(
                    "too big", model_id=_MODEL_ID, backend=_BACKEND
                )
            ),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(context_window=100))

    result = await run_task("do it", deps, task_id="t-compact-overflow")

    assert result.reason == TerminationReason.CONTEXT_OVERFLOW
    assert result.turns_used == 2
    assert client.call_count == 3


async def test_a_compaction_call_that_crosses_the_token_cap_ends_the_task_that_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `max_total_tokens` cap that only the compaction call's
    own token usage pushes past, when the task runs, then it ends
    `TOKEN_CAP` right after the compaction call -- a fourth scripted
    turn, if the loop wrongly attempted one, would raise `IndexError`
    rather than being silently skipped."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn("carry-forward summary", input_tokens=50, output_tokens=10),
        ]
    )
    # Turns 1+2 alone total 130 tokens; only adding the compaction call's
    # own 60 (50 input + 10 output) crosses this 150-token cap.
    deps = _build_deps(client, tmp_path, _registry(context_window=100))
    deps.limits = LoopLimits(max_total_tokens=150)

    result = await run_task("do it", deps, task_id="t-compact-token-cap")

    assert result.reason == TerminationReason.TOKEN_CAP
    assert result.turns_used == 2
    assert client.call_count == 3
