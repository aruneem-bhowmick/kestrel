"""Unit tests for the agent loop's own session-journaling wiring:
`LoopDeps.session` receiving one `TurnRecord` per real turn with the
right message-delta slice, and `resume_task`'s reconstruction and
continuation of a prior journal's turn count, cost meter, and
verification state.

Reuses `test_p022_loop.py`'s scripted-`ProviderClient` pattern and
`test_p026_verification_gate.py`'s monkeypatched-`dispatch` pattern
rather than a live mock server, since what is under test here is the
loop's own bookkeeping around a session, not a real model or tool call
-- that is `test_p029_session_persist_resume.py`'s own system-level job.
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
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport

pytestmark = [pytest.mark.p029, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry() -> Registry:
    """Build a single-entry `Registry`, matching this suite's siblings."""
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


def _stop_turn(
    text: str, *, input_tokens: int = 100, output_tokens: int = 20
) -> _ScriptedTurn:
    """A turn that answers with plain text and requests no tool calls."""
    return _ScriptedTurn(
        events=(
            TextDelta(text=text),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="end_turn"),
        )
    )


def _tool_turn(
    call_id: str, name: str, *, input_tokens: int = 60, output_tokens: int = 15
) -> _ScriptedTurn:
    """A turn that requests exactly one tool call named `name`."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(id=call_id, name=name, arguments_json="{}"),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _fake_dispatch(
    event: ToolCallEvent, *, repo_root: Path, **context: object
) -> ToolResult:
    """Stand in for `kestrel.agent.loop.dispatch`: succeeds unconditionally
    without running any real tool, so this suite can drive multi-turn
    tool-calling tasks without depending on the sandbox or the registry's
    real executors."""
    return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")


def _fake_verify_dispatch(
    report: VerificationReport,
) -> object:
    """Stand in for `dispatch`, appending `report` to whatever
    `report_sink` the loop threaded through -- exactly the contract the
    real `verify` executor honors."""

    def _dispatch(
        event: ToolCallEvent, *, repo_root: Path, **context: object
    ) -> ToolResult:
        sink = context.get("report_sink")
        if isinstance(sink, list):
            sink.append(report)
        return ToolResult(tool_call_id=event.id, content="verify: stub")

    return _dispatch


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    *,
    limits: LoopLimits = LoopLimits(),
    session: SessionManager | None = None,
    require_verification: bool = False,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle from real managers and a fresh
    `CostMeter`, scoped to `repo_root`, for one test's `run_task` or
    `resume_task` call."""
    return LoopDeps(
        client=client,
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=limits,
        session=session,
        require_verification=require_verification,
    )


@pytest.mark.sanity
async def test_run_task_journals_the_seed_message_and_the_turns_own_output(
    tmp_path: Path,
) -> None:
    """Given a single-turn task with `deps.session` set, when it
    completes, then exactly one `TurnRecord` is journaled, and its
    `message_deltas` covers both the seeded user message (never
    previously persisted) and this turn's own assistant reply."""
    client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-1")
    deps = _build_deps(client, tmp_path, session=session)

    result = await run_task("say hi", deps, task_id="t-loop-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert len(session.records) == 1
    record = session.records[0]
    assert record.turn_id == 1
    assert record.task_id == "t-loop-1"
    assert record.message_deltas == (
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "done"},
    )
    assert record.turn_cost == deps.meter.turns[0]
    assert record.verification is None


async def test_a_second_turns_deltas_do_not_repeat_the_first_turns_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a two-turn task (a tool call, then a stop), when both
    complete, then two `TurnRecord`s are journaled, and the second one's
    `message_deltas` holds only its own new message -- not a repeat of
    the first turn's, which the first record already covers."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[_tool_turn("call-1", "read_file"), _stop_turn("done")]
    )
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-2")
    deps = _build_deps(client, tmp_path, session=session)

    result = await run_task("do it", deps, task_id="t-loop-2")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert len(session.records) == 2
    first, second = session.records
    assert first.turn_id == 1
    assert first.message_deltas[0] == {"role": "user", "content": "do it"}
    assert second.turn_id == 2
    assert second.message_deltas == ({"role": "assistant", "content": "done"},)


async def test_a_self_critique_declined_turn_is_still_journaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `self_critique_fn` that declines the first turn's proposed
    tool call, when the task runs, then the declined turn is still
    journaled, carrying the synthetic skip explanation rather than a
    real tool result."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    client = _ScriptedLoopClient(
        turns=[_tool_turn("call-1", "read_file"), _stop_turn("done")]
    )
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-3")
    decisions = iter([False, True])
    deps = _build_deps(client, tmp_path, session=session)
    deps.self_critique_fn = lambda proposal, history: next(decisions)

    result = await run_task("do it", deps, task_id="t-loop-3")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert len(session.records) == 2
    declined = session.records[0]
    tool_messages = [m for m in declined.message_deltas if m["role"] == "tool"]
    assert tool_messages[0]["content"] == loop_module._SELF_CRITIQUE_SKIP_CONTENT


async def test_verification_reports_are_captured_on_the_turn_that_produced_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `require_verification=True` and a scripted `verify` call
    whose (faked) dispatch appends a passing report, when the task runs,
    then both the turn that produced the report and the following
    completing turn carry it in their own `TurnRecord.verification`."""
    report = VerificationReport(task_id="t-loop-4", turn_id=1, commands=(), passed=True)
    monkeypatch.setattr(loop_module, "dispatch", _fake_verify_dispatch(report))
    client = _ScriptedLoopClient(
        turns=[_tool_turn("call-1", "verify"), _stop_turn("done")]
    )
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-4")
    deps = _build_deps(client, tmp_path, session=session, require_verification=True)

    result = await run_task("fix it", deps, task_id="t-loop-4")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert len(session.records) == 2
    assert session.records[0].verification == report
    assert session.records[1].verification == report


async def test_resume_task_continues_the_turn_counter_and_reseeds_the_meter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task halted at `TURN_CAP` after one journaled turn, when a
    second, independent `LoopDeps` resumes it via `resume_task`, then the
    turn counter continues from where the journal left off, the meter is
    re-seeded with the original turn's own cost, and the final history
    contains every message from both runs, in order."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    first_client = _ScriptedLoopClient(turns=[_tool_turn("call-1", "read_file")])
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-5")
    first_deps = _build_deps(
        first_client, tmp_path, limits=LoopLimits(max_turns=1), session=session
    )

    first_result = await run_task("do it", first_deps, task_id="t-loop-5")
    assert first_result.reason == TerminationReason.TURN_CAP
    assert first_result.turns_used == 1

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    second_session = SessionManager(repo_root=tmp_path, task_id="t-loop-5")
    second_deps = _build_deps(
        second_client, tmp_path, limits=LoopLimits(max_turns=10), session=second_session
    )

    resumed_result = await resume_task("t-loop-5", second_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 2
    assert second_client.call_count == 1
    assert len(second_deps.meter.turns) == 2
    assert second_deps.meter.turns[0] == first_deps.meter.turns[0]
    assert resumed_result.history[0] == {"role": "user", "content": "do it"}
    assert resumed_result.history[-1] == {"role": "assistant", "content": "done"}
    assert len(second_session.records) == 2


async def test_resume_task_seeds_verification_reports_from_the_last_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a halted task whose journal's last record carries a passing
    `VerificationReport`, when resumed with `require_verification=True`,
    then `deps.verification_reports` is pre-populated with that report,
    letting the very next no-tool-calls turn complete the task."""
    report = VerificationReport(task_id="t-loop-6", turn_id=1, commands=(), passed=True)
    monkeypatch.setattr(loop_module, "dispatch", _fake_verify_dispatch(report))
    first_client = _ScriptedLoopClient(turns=[_tool_turn("call-1", "verify")])
    session = SessionManager(repo_root=tmp_path, task_id="t-loop-6")
    first_deps = _build_deps(
        first_client,
        tmp_path,
        limits=LoopLimits(max_turns=1),
        session=session,
        require_verification=True,
    )
    await run_task("fix it", first_deps, task_id="t-loop-6")

    second_client = _ScriptedLoopClient(turns=[_stop_turn("now done")])
    second_session = SessionManager(repo_root=tmp_path, task_id="t-loop-6")
    second_deps = _build_deps(
        second_client,
        tmp_path,
        limits=LoopLimits(max_turns=10),
        require_verification=True,
        session=second_session,
    )

    resumed_result = await resume_task("t-loop-6", second_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert second_deps.verification_reports == [report]
    assert len(second_session.records) == 2


@pytest.mark.sanity
async def test_resume_task_on_an_unknown_task_id_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Given no session journal at all for a task id, when `resume_task`
    is called, then `FileNotFoundError` propagates unchanged rather than
    being swallowed."""
    deps = _build_deps(_ScriptedLoopClient(turns=[]), tmp_path)

    with pytest.raises(FileNotFoundError):
        await resume_task("no-such-task", deps)
