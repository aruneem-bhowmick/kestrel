"""Unit tests for the agent loop's opt-in verification gate:
`LoopDeps.require_verification` withholding `TASK_COMPLETE` from a
no-tool-calls turn until the most recent recorded `VerificationReport`
actually passed.

Reuses `test_p022_loop.py`'s scripted-`ProviderClient` pattern; the
`verify` tool call itself is stubbed out via a monkeypatched
`kestrel.agent.loop.dispatch` (exactly like that suite's own
keyboard-interrupt test) rather than run for real, since what's under
test here is the loop's own gating logic around a report, not the
sandboxed tool behind it -- that belongs to `test_p025_verify.py` and
its own integration/system counterparts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.loop import LoopDeps, LoopLimits, TerminationReason, run_task
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
from kestrel.tools.verify import VerificationReport

pytestmark = [pytest.mark.p026, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry() -> Registry:
    """Build a single-entry `Registry` at the same rates
    `test_p022_loop.py` pins its own cost-regression cases against, so a
    band asserted here is comparable to that suite's."""
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
    order, ignoring every argument but the count. Running out of script
    is a test-authoring error and raises `IndexError`."""

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


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    *,
    limits: LoopLimits = LoopLimits(),
    require_verification: bool = False,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle from real managers and a fresh
    `CostMeter`, scoped to `repo_root`, for one test's `run_task` call."""
    return LoopDeps(
        client=client,
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=limits,
        require_verification=require_verification,
    )


def _stop_turn(
    text: str, *, input_tokens: int = 100, output_tokens: int = 20
) -> _ScriptedTurn:
    """A turn that answers with plain text and requests no tool calls --
    the natural-completion shape, gated or not depending on
    `require_verification`."""
    return _ScriptedTurn(
        events=(
            TextDelta(text=text),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="end_turn"),
        )
    )


def _verify_turn(
    call_id: str, *, input_tokens: int = 60, output_tokens: int = 15
) -> _ScriptedTurn:
    """A turn that requests exactly one `verify` tool call."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(id=call_id, name="verify", arguments_json="{}"),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _fake_report(*, passed: bool, turn_id: int = 1) -> VerificationReport:
    """A `VerificationReport` built directly rather than through a real
    `run_verification` call -- only its `passed` flag matters to the
    gate under test here."""
    return VerificationReport(
        task_id="stub-task", turn_id=turn_id, commands=(), passed=passed
    )


def _fake_dispatch_appending_reports(
    reports: Sequence[VerificationReport],
) -> Callable[..., ToolResult]:
    """Stand in for `kestrel.agent.loop.dispatch`: pops the next
    scripted report off `reports` and appends it to whatever
    `report_sink` the loop threaded through -- exactly the contract the
    real `verify` executor honors -- without running a real sandboxed
    command. Any tool name other than `verify` is a test-authoring
    error."""
    queue = list(reports)

    def _dispatch(
        event: ToolCallEvent, *, repo_root: Path, **context: object
    ) -> ToolResult:
        assert event.name == "verify", f"unexpected tool call {event.name!r}"
        report_sink = context.get("report_sink")
        report = queue.pop(0)
        if isinstance(report_sink, list):
            report_sink.append(report)
        return ToolResult(tool_call_id=event.id, content="verify: stub")

    return _dispatch


def _tool_messages(history: tuple[Message, ...]) -> list[Message]:
    """Every tool-role message in `history`, in order."""
    return [message for message in history if message["role"] == "tool"]


@pytest.mark.sanity
@pytest.mark.regression
async def test_default_require_verification_false_completes_after_one_turn_unchanged(
    tmp_path: Path,
) -> None:
    """Given `require_verification` left at its default `False`, when a
    turn stops with no tool calls, then the task still ends
    TASK_COMPLETE after exactly one turn -- byte-identical to the
    pre-gate behavior `test_p022_loop.py` already pins, proving the
    gate is a strict opt-in with no effect on an unset caller."""
    client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    deps = _build_deps(client, tmp_path)

    result = await run_task("say hi", deps, task_id="t-p026-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 1
    assert client.call_count == 1


@pytest.mark.sanity
async def test_required_but_never_verified_never_completes_and_trips_turn_cap(
    tmp_path: Path,
) -> None:
    """Given `require_verification=True` and a script that stops with no
    tool calls on every turn without ever calling `verify`, when the
    task runs, then it never reaches TASK_COMPLETE -- it keeps
    re-entering Think with the nudge folded into history each time,
    until `max_turns` trips TURN_CAP."""
    client = _ScriptedLoopClient(
        turns=[_stop_turn("done?"), _stop_turn("done?"), _stop_turn("done?")]
    )
    deps = _build_deps(
        client, tmp_path, limits=LoopLimits(max_turns=3), require_verification=True
    )

    result = await run_task("say hi", deps, task_id="t-p026-2")

    assert result.reason == TerminationReason.TURN_CAP
    assert result.turns_used == 3
    assert client.call_count == 3
    tool_messages = _tool_messages(result.history)
    assert len(tool_messages) == 3
    assert all(
        message["content"] == loop_module._VERIFICATION_REQUIRED_CONTENT
        for message in tool_messages
    )


async def test_failing_verification_report_still_withholds_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `require_verification=True` and a scripted `verify` call
    whose (faked) dispatch appends a `passed=False` report, followed by
    a no-tool-calls turn, when the task runs, then that turn still does
    not complete the task -- it is nudged and the loop keeps going
    instead."""
    client = _ScriptedLoopClient(
        turns=[_verify_turn("call-1"), _stop_turn("still not done?")]
    )
    monkeypatch.setattr(
        loop_module,
        "dispatch",
        _fake_dispatch_appending_reports([_fake_report(passed=False)]),
    )
    deps = _build_deps(
        client, tmp_path, limits=LoopLimits(max_turns=2), require_verification=True
    )

    result = await run_task("fix it", deps, task_id="t-p026-3")

    assert result.reason == TerminationReason.TURN_CAP
    assert result.turns_used == 2
    assert client.call_count == 2
    assert deps.verification_reports == [_fake_report(passed=False)]


async def test_passing_verification_report_lets_the_next_stop_turn_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `require_verification=True` and a scripted `verify` call
    whose (faked) dispatch appends a `passed=True` report, followed by a
    no-tool-calls turn, when the task runs, then that following turn
    DOES end the task TASK_COMPLETE."""
    client = _ScriptedLoopClient(
        turns=[_verify_turn("call-1"), _stop_turn("now it's done")]
    )
    monkeypatch.setattr(
        loop_module,
        "dispatch",
        _fake_dispatch_appending_reports([_fake_report(passed=True)]),
    )
    deps = _build_deps(client, tmp_path, require_verification=True)

    result = await run_task("fix it", deps, task_id="t-p026-4")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert client.call_count == 2


async def test_gate_reads_only_the_most_recent_of_several_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two recorded verification reports -- the first failing, the
    second passing -- when a no-tool-calls turn follows, then the task
    still completes: `_has_passing_verification` reads only the last
    entry in `verification_reports`, not whether every entry passed."""
    client = _ScriptedLoopClient(
        turns=[
            _verify_turn("call-1"),
            _verify_turn("call-2"),
            _stop_turn("now it's done"),
        ]
    )
    monkeypatch.setattr(
        loop_module,
        "dispatch",
        _fake_dispatch_appending_reports(
            [
                _fake_report(passed=False, turn_id=1),
                _fake_report(passed=True, turn_id=2),
            ]
        ),
    )
    deps = _build_deps(client, tmp_path, require_verification=True)

    result = await run_task("fix it", deps, task_id="t-p026-5")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    assert len(deps.verification_reports) == 2
    assert deps.verification_reports[0].passed is False
    assert deps.verification_reports[-1].passed is True


async def test_nudge_turn_is_still_subject_to_the_token_cap(tmp_path: Path) -> None:
    """Given `require_verification=True` and a first turn whose own
    usage already crosses `max_total_tokens`, when the task runs, then
    it ends TOKEN_CAP right after the nudge turn's usage is folded in --
    the gate's own nudge path is subject to the token cap exactly like
    the self-critique-skip path already is, never granting a
    verification-required task unbounded turns to reach a passing
    `verify` call."""
    client = _ScriptedLoopClient(
        turns=[_stop_turn("not done yet", input_tokens=900, output_tokens=200)]
    )
    deps = _build_deps(
        client,
        tmp_path,
        limits=LoopLimits(max_total_tokens=1000),
        require_verification=True,
    )

    result = await run_task("fix it", deps, task_id="t-p026-6")

    assert result.reason == TerminationReason.TOKEN_CAP
    assert result.turns_used == 1
    assert client.call_count == 1


@pytest.mark.cost_regression
async def test_nudge_turn_cost_band(tmp_path: Path) -> None:
    """The nudge-message turn's own usage is priced and folded into
    `deps.meter` exactly like any other turn's -- pinned to a Decimal
    band so a change that silently adds extra unbudgeted turns to every
    verification-required task is caught here rather than discovered as
    a cost surprise later."""
    client = _ScriptedLoopClient(
        turns=[_stop_turn("not done yet", input_tokens=1000, output_tokens=50)]
    )
    deps = _build_deps(
        client, tmp_path, limits=LoopLimits(max_turns=1), require_verification=True
    )

    result = await run_task("fix it", deps, task_id="t-p026-cost")

    # one nudged turn: (1000 * 0.60 + 50 * 2.20) / 1e6 = 0.000710
    assert result.reason == TerminationReason.TURN_CAP
    assert result.turns_used == 1
    assert result.total_usd == Decimal("0.000710")
