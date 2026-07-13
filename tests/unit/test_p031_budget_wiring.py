"""Unit tests for the agent loop's own budget wiring: a `BudgetManager`
configured on `LoopDeps` degrades a task to a `"cheap"`-tagged registry
entry once its soft threshold trips, halts it outright once its hard
threshold trips, and never touches a task that leaves `budget` unset.

Reuses `test_p022_loop.py`'s scripted-`ProviderClient` pattern and
`test_p029_loop_session.py`'s monkeypatched-`dispatch` pattern rather
than a live mock server, since what is under test here is the loop's
own accounting and model-routing decisions, not a real model or tool
call -- that is `test_p031_budget_halt_resume.py`'s own system-level job.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.loop import (
    LoopDeps,
    LoopLimits,
    TerminationReason,
    resume_task,
    run_task,
)
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.budget import BudgetLimits, BudgetManager
from kestrel.managers.session import SessionManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import (
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from kestrel.registry.model import ModelEntry, Registry, Tag
from kestrel.tools.registry import ToolResult

pytestmark = [pytest.mark.p031, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_CHEAP_MODEL_ID = "glm-5.2-air"
_BACKEND = "openrouter"


def _entry(
    model_id: str,
    *,
    tags: frozenset[Tag] = frozenset(),
    usd_per_mtok_input: Decimal = Decimal("1.00"),
    usd_per_mtok_output: Decimal = Decimal("0"),
    usd_per_mtok_cached: Decimal = Decimal("0"),
) -> ModelEntry:
    """Build one registry entry with round, easy-to-hand-verify rates by
    default -- individual tests override a rate where the scenario
    itself needs the original and cheap entries priced differently."""
    return ModelEntry(
        id=model_id,
        backend=_BACKEND,
        provider_model=f"z-ai/{model_id}",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=usd_per_mtok_input,
        usd_per_mtok_output=usd_per_mtok_output,
        usd_per_mtok_cached=usd_per_mtok_cached,
        supports_tools=True,
        supports_cache=True,
        tags=tags,
    )


def _registry_with_cheap_entry(
    *, cheap_usd_per_mtok_input: Decimal = Decimal("1.00")
) -> Registry:
    """A two-entry `Registry`: the original model plus a `"cheap"`-tagged
    one to degrade to. `cheap_usd_per_mtok_input` defaults to the same
    rate as the original, so a scenario that only cares about *which*
    model a turn was sent to isn't also asserting on price; a scenario
    that cares about price too (the cost-regression case) overrides it.
    """
    return Registry(
        models={
            _MODEL_ID: _entry(_MODEL_ID, tags=frozenset({"planner", "executor"})),
            _CHEAP_MODEL_ID: _entry(
                _CHEAP_MODEL_ID,
                tags=frozenset({"cheap"}),
                usd_per_mtok_input=cheap_usd_per_mtok_input,
            ),
        },
        source=None,
    )


def _registry_without_cheap_entry() -> Registry:
    """A single-entry `Registry` with no `"cheap"`-tagged route at all."""
    return Registry(
        models={_MODEL_ID: _entry(_MODEL_ID, tags=frozenset({"planner", "executor"}))},
        source=None,
    )


@dataclass
class _ScriptedTurn:
    """One scripted `.complete()` call's outcome -- the events to yield."""

    events: tuple[StreamEvent, ...] = ()


@dataclass
class _ScriptedLoopClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order, and records every call's own `model_id` so a test can assert
    which registry entry a given turn actually targeted. Calling
    `.complete()` more times than there are scripted turns is a
    test-authoring error and raises `IndexError` -- every case here
    scripts exactly as many turns as the loop should take, so running
    out of script means the loop kept going past where it should have
    stopped.
    """

    turns: Sequence[_ScriptedTurn]
    call_count: int = field(default=0, init=False)
    requested_model_ids: list[str] = field(default_factory=list, init=False)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted turn, recording the `model_id` it
        was actually called with."""
        turn = self.turns[self.call_count]
        self.requested_model_ids.append(model_id)
        self.call_count += 1
        for event in turn.events:
            yield event


def _tool_turn(
    call_id: str, *, input_tokens: int, output_tokens: int = 0
) -> _ScriptedTurn:
    """A turn that requests exactly one tool call, priced at
    `input_tokens`/`output_tokens` -- the "keep going" shape a budget
    check needs to reach a later turn."""
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
    the natural-completion shape."""
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
    without running any real tool, so this suite can drive multi-turn
    tool-calling tasks without depending on the sandbox or the
    registry's real executors."""
    return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")


_UNBOUNDED_TOKENS = LoopLimits(max_total_tokens=100_000_000)


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    registry: Registry,
    *,
    budget: BudgetManager | None = None,
    limits: LoopLimits = _UNBOUNDED_TOKENS,
    session: SessionManager | None = None,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle from real managers and a fresh
    `CostMeter`, scoped to `repo_root`, for one test's `run_task` or
    `resume_task` call.

    `limits` defaults to a token cap far above anything this suite's own
    scripted turns spend -- the scenarios here script large token counts
    to hit clean dollar amounts, not to exercise `TOKEN_CAP`, which has
    its own dedicated coverage elsewhere."""
    return LoopDeps(
        client=client,
        registry=registry,
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=limits,
        budget=budget,
        session=session,
    )


async def test_soft_cap_degrades_the_very_next_turn_to_the_cheap_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a session cap crossed into SOFT on turn 2, with a
    `"cheap"`-tagged entry present in the registry, when the task keeps
    running, then turn 3's Think call targets the cheap entry's own
    `model_id`, and turn 3's own priced `TurnCost.model_id` matches it
    too."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=500_000),
            _tool_turn("call-2", input_tokens=500_000),
            _stop_turn("done", input_tokens=100),
        ]
    )
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("1.20")))
    deps = _build_deps(client, tmp_path, _registry_with_cheap_entry(), budget=budget)

    result = await run_task("do it", deps, task_id="t-soft-degrade")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 3
    assert client.requested_model_ids == [_MODEL_ID, _MODEL_ID, _CHEAP_MODEL_ID]
    assert deps.meter.turns[2].model_id == _CHEAP_MODEL_ID


async def test_soft_cap_with_no_cheap_entry_keeps_running_on_the_original_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Given the same soft-cap crossing but a registry with no
    `"cheap"`-tagged entry at all, when the task keeps running, then it
    continues on the original model without raising, and a warning
    names the missing degrade target."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=500_000),
            _tool_turn("call-2", input_tokens=500_000),
            _stop_turn("done", input_tokens=100),
        ]
    )
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("1.20")))
    deps = _build_deps(client, tmp_path, _registry_without_cheap_entry(), budget=budget)

    with caplog.at_level("WARNING", logger="kestrel.agent"):
        result = await run_task("do it", deps, task_id="t-soft-no-cheap")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 3
    assert client.requested_model_ids == [_MODEL_ID, _MODEL_ID, _MODEL_ID]
    assert "no 'cheap'-tagged registry entry" in caplog.text


@pytest.mark.sanity
async def test_hard_cap_halts_after_the_tripping_turn_with_no_third_think_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given cumulative cost crossing the HARD threshold on turn 2, when
    the task runs, then it ends `BUDGET_HALT` after exactly two turns --
    a third scripted turn, if the loop wrongly attempted one, would
    raise `IndexError` rather than being silently skipped."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=500_000),
            _tool_turn("call-2", input_tokens=500_000),
        ]
    )
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("0.90")))
    deps = _build_deps(client, tmp_path, _registry_with_cheap_entry(), budget=budget)

    result = await run_task("do it", deps, task_id="t-hard-halt")

    assert result.reason == TerminationReason.BUDGET_HALT
    assert result.turns_used == 2
    assert client.call_count == 2


@pytest.mark.sanity
@pytest.mark.regression
async def test_budget_none_never_runs_any_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `deps.budget` left at its default of `None`, when a
    multi-turn task runs with spend that would otherwise cross both
    thresholds many times over, then it completes exactly as every
    budget-unaware caller from an earlier phase would have -- no
    degrade, no halt, and no exception."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=10_000_000),
            _stop_turn("done", input_tokens=10_000_000),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry_with_cheap_entry())
    assert deps.budget is None

    result = await run_task("do it", deps, task_id="t-no-budget")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert client.requested_model_ids == [_MODEL_ID, _MODEL_ID]


async def test_soft_then_hard_degrades_once_on_turn_two_then_halts_on_turn_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task that crosses SOFT on turn 1 and later HARD on turn 3,
    when it runs, then it degrades exactly once (turn 2 onward targets
    the cheap entry) and halts on turn 3, in that order -- never
    attempting a second degrade and never attempting a fourth turn."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=1_300_000),
            _tool_turn("call-2", input_tokens=1_000_000),
            _tool_turn("call-3", input_tokens=2_000_000),
        ]
    )
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("1.50")))
    registry = _registry_with_cheap_entry(cheap_usd_per_mtok_input=Decimal("0.10"))
    deps = _build_deps(client, tmp_path, registry, budget=budget)

    result = await run_task("do it", deps, task_id="t-soft-then-hard")

    assert result.reason == TerminationReason.BUDGET_HALT
    assert result.turns_used == 3
    assert client.call_count == 3
    assert client.requested_model_ids == [_MODEL_ID, _CHEAP_MODEL_ID, _CHEAP_MODEL_ID]


async def test_resume_after_a_budget_halt_completes_once_the_cap_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task halted `BUDGET_HALT` at turn 2, with its journal
    persisted via a real `SessionManager`, when a second, independent
    `LoopDeps` resumes it with a raised session cap, then the task
    proceeds to `TASK_COMPLETE` on turn 3, continuing the turn counter
    rather than resetting it, with every turn's messages -- from both
    the original and the resumed run -- present in the final history."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    registry = _registry_with_cheap_entry()

    first_client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=500_000),
            _tool_turn("call-2", input_tokens=500_000),
        ]
    )
    first_budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("0.90")))
    session = SessionManager(repo_root=tmp_path, task_id="t-budget-resume")
    first_deps = _build_deps(
        first_client, tmp_path, registry, budget=first_budget, session=session
    )

    first_result = await run_task("do it", first_deps, task_id="t-budget-resume")
    assert first_result.reason == TerminationReason.BUDGET_HALT
    assert first_result.turns_used == 2

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done", input_tokens=100)])
    raised_budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("100")))
    second_session = SessionManager(repo_root=tmp_path, task_id="t-budget-resume")
    second_deps = _build_deps(
        second_client, tmp_path, registry, budget=raised_budget, session=second_session
    )

    resumed_result = await resume_task("t-budget-resume", second_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 3
    assert second_client.call_count == 1
    assert len(second_session.records) == 3
    assert resumed_result.history[0] == {"role": "user", "content": "do it"}
    assert resumed_result.history[-1] == {"role": "assistant", "content": "done"}
    tool_messages = [m for m in resumed_result.history if m["role"] == "tool"]
    assert len(tool_messages) == 2


@pytest.mark.cost_regression
async def test_degraded_task_cost_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a registry whose original and cheap entries carry different,
    realistic rates, when a task degrades partway through, then its
    final `total_usd` is exactly the sum of the original entry's own
    rate for the turns before the degrade and the cheap entry's own rate
    for the turn after it -- pinned exactly, to catch a change that
    silently prices a degraded turn at the wrong entry's rates."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    registry = Registry(
        models={
            _MODEL_ID: _entry(
                _MODEL_ID,
                tags=frozenset({"planner", "executor"}),
                usd_per_mtok_input=Decimal("0.60"),
                usd_per_mtok_output=Decimal("2.20"),
                usd_per_mtok_cached=Decimal("0.11"),
            ),
            _CHEAP_MODEL_ID: _entry(
                _CHEAP_MODEL_ID,
                tags=frozenset({"cheap"}),
                usd_per_mtok_input=Decimal("0.10"),
                usd_per_mtok_output=Decimal("0.40"),
                usd_per_mtok_cached=Decimal("0.02"),
            ),
        },
        source=None,
    )
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=1000, output_tokens=200),
            _tool_turn("call-2", input_tokens=1000, output_tokens=200),
            _stop_turn("done", input_tokens=1000, output_tokens=200),
        ]
    )
    budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("0.0025")))
    deps = _build_deps(client, tmp_path, registry, budget=budget)

    result = await run_task("do it", deps, task_id="t-cost-band")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.requested_model_ids == [_MODEL_ID, _MODEL_ID, _CHEAP_MODEL_ID]
    assert result.total_usd == Decimal("0.002260")


async def test_resume_after_budget_halt_preserves_degraded_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task that crosses SOFT on turn 1, degrades, and then halts
    on turn 2 (BUDGET_HALT), when it is resumed with a raised cap, then
    it remains on the cheap model for turn 3 instead of resetting to the
    original model."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    registry = _registry_with_cheap_entry()

    first_client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=1_300_000),
            _tool_turn("call-2", input_tokens=1_000_000),
        ]
    )
    first_budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("1.50")))
    session = SessionManager(repo_root=tmp_path, task_id="t-budget-resume-degraded")
    first_deps = _build_deps(
        first_client, tmp_path, registry, budget=first_budget, session=session
    )

    first_result = await run_task(
        "do it", first_deps, task_id="t-budget-resume-degraded"
    )
    assert first_result.reason == TerminationReason.BUDGET_HALT
    assert first_result.turns_used == 2
    assert first_client.requested_model_ids == [_MODEL_ID, _CHEAP_MODEL_ID]

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done", input_tokens=100)])
    raised_budget = BudgetManager(limits=BudgetLimits(session_usd=Decimal("100")))
    second_session = SessionManager(
        repo_root=tmp_path, task_id="t-budget-resume-degraded"
    )
    second_deps = _build_deps(
        second_client, tmp_path, registry, budget=raised_budget, session=second_session
    )

    resumed_result = await resume_task("t-budget-resume-degraded", second_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 3
    assert second_client.requested_model_ids == [_CHEAP_MODEL_ID]
