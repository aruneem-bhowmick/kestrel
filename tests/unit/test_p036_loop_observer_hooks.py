"""Unit tests for `LoopDeps.observer`'s wiring into `_drive`: every one
of `LoopObserver`'s seven hooks fires at the point its own contract
promises, in the order a real task actually reaches them.

Reuses `test_p022_loop.py`'s scripted-`ProviderClient` pattern, plus a
`RecordingObserver` test double that captures every call it receives,
in arrival order, so each case here asserts against the exact call
sequence a real `run_task` invocation produced -- not just that a hook
fired at all, but where.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.loop import (
    LoopDeps,
    LoopLimits,
    LoopResult,
    TerminationReason,
    run_task,
)
from kestrel.cost.meter import CostMeter, TurnCost
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

pytestmark = [pytest.mark.p036, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry(*, context_window: int = 200_000) -> Registry:
    """A single-entry `Registry` at the same rates `test_p022_loop.py`
    pins its own cost-regression case against, with a caller-chosen
    `context_window` -- small enough, for the compaction case here, to
    cross the fold threshold on this suite's own scripted token counts."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend=_BACKEND,
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=context_window,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={_MODEL_ID: entry}, source=None)


@dataclass
class RecordingObserver:
    """A `LoopObserver` recording every call it receives, in arrival
    order, as `(method_name, payload)` pairs. `payload` is a `dict` of
    keyword arguments for a keyword-only method, or the positional
    argument(s) themselves otherwise -- whichever shape a test can
    assert against directly."""

    calls: list[tuple[str, object]] = field(default_factory=list)

    def on_turn_started(self, *, turn_id: int, active_model_id: str) -> None:
        """Record this call."""
        self.calls.append(
            (
                "on_turn_started",
                {"turn_id": turn_id, "active_model_id": active_model_id},
            )
        )

    def on_text_delta(self, text: str) -> None:
        """Record this call."""
        self.calls.append(("on_text_delta", text))

    def on_tool_call_started(self, call: ToolCallEvent) -> None:
        """Record this call."""
        self.calls.append(("on_tool_call_started", call))

    def on_tool_call_finished(self, call: ToolCallEvent, result: ToolResult) -> None:
        """Record this call."""
        self.calls.append(("on_tool_call_finished", (call, result)))

    def on_verification(self, report: VerificationReport) -> None:
        """Record this call."""
        self.calls.append(("on_verification", report))

    def on_turn_finished(
        self, *, turn_id: int, turn_cost: TurnCost, active_model_id: str
    ) -> None:
        """Record this call."""
        self.calls.append(
            (
                "on_turn_finished",
                {
                    "turn_id": turn_id,
                    "turn_cost": turn_cost,
                    "active_model_id": active_model_id,
                },
            )
        )

    def on_termination(self, result: LoopResult) -> None:
        """Record this call."""
        self.calls.append(("on_termination", result))

    def names(self) -> list[str]:
        """Just the method names, in arrival order -- the shape a test
        asserts against when only call sequencing matters, not payload
        detail."""
        return [name for name, _ in self.calls]


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


def _stop_turn(
    chunks: Sequence[str], *, input_tokens: int = 100, output_tokens: int = 20
) -> _ScriptedTurn:
    """A turn that streams `chunks` as separate `TextDelta` events, in
    order, then stops with no tool calls -- the natural-completion
    shape, letting a test assert `on_text_delta` fires once per chunk
    rather than once per turn."""
    return _ScriptedTurn(
        events=(
            *(TextDelta(text=chunk) for chunk in chunks),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="end_turn"),
        )
    )


def _tool_turn(
    calls: Sequence[tuple[str, str, str]],
    *,
    input_tokens: int = 50,
    output_tokens: int = 10,
) -> _ScriptedTurn:
    """A turn that requests every `(call_id, name, arguments_json)` in
    `calls`, in order, then stops for tool dispatch."""
    return _ScriptedTurn(
        events=(
            *(
                ToolCallEvent(id=call_id, name=name, arguments_json=arguments_json)
                for call_id, name, arguments_json in calls
            ),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _fake_dispatch(
    event: ToolCallEvent, *, repo_root: Path, **context: object
) -> ToolResult:
    """Stand in for `kestrel.agent.loop.dispatch`: succeeds
    unconditionally without running any real tool or touching
    `report_sink`."""
    return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")


def _fake_dispatch_with_reports(
    reports: Sequence[VerificationReport],
) -> Callable[..., ToolResult]:
    """Stand in for `kestrel.agent.loop.dispatch`: a `verify` call pops
    the next scripted report off `reports` and appends it to whatever
    `report_sink` the loop threaded through, exactly like the real
    `verify` executor's own contract; every other tool name succeeds
    unconditionally with no report appended."""
    queue = list(reports)

    def _dispatch(
        event: ToolCallEvent, *, repo_root: Path, **context: object
    ) -> ToolResult:
        """Replay the next scripted report for a `verify` call, or
        succeed plainly for any other tool."""
        if event.name == "verify":
            report_sink = context.get("report_sink")
            report = queue.pop(0)
            if isinstance(report_sink, list):
                report_sink.append(report)
        return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")

    return _dispatch


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    registry: Registry,
    *,
    observer: RecordingObserver,
    limits: LoopLimits = LoopLimits(),
    self_critique_fn: Callable[[str, list[Message]], bool] | None = None,
    require_verification: bool = False,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle wired to `observer`, from real
    managers and a fresh `CostMeter`, for one test's `run_task` call."""
    kwargs: dict[str, object] = {}
    if self_critique_fn is not None:
        kwargs["self_critique_fn"] = self_critique_fn
    return LoopDeps(
        client=client,
        registry=registry,
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=limits,
        require_verification=require_verification,
        observer=observer,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.sanity
async def test_one_turn_no_tool_calls_fires_started_deltas_finished_then_termination(
    tmp_path: Path,
) -> None:
    """Given a one-turn, no-tool-calls task streaming two text chunks,
    when the task runs, then the observer sees exactly:
    `on_turn_started`, `on_text_delta` once per chunk in order,
    `on_turn_finished`, `on_termination` -- and nothing else."""
    observer = RecordingObserver()
    client = _ScriptedLoopClient(turns=[_stop_turn(["Hello ", "world"])])
    deps = _build_deps(client, tmp_path, _registry(), observer=observer)

    result = await run_task("say hi", deps, task_id="t-p036-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert observer.names() == [
        "on_turn_started",
        "on_text_delta",
        "on_text_delta",
        "on_turn_finished",
        "on_termination",
    ]
    assert observer.calls[0][1] == {"turn_id": 1, "active_model_id": _MODEL_ID}
    assert observer.calls[1][1] == "Hello "
    assert observer.calls[2][1] == "world"
    turn_finished_payload = observer.calls[3][1]
    assert isinstance(turn_finished_payload, dict)
    assert turn_finished_payload["turn_id"] == 1
    assert turn_finished_payload["active_model_id"] == _MODEL_ID
    termination_result = observer.calls[4][1]
    assert isinstance(termination_result, LoopResult)
    assert termination_result.reason == TerminationReason.TASK_COMPLETE


@pytest.mark.sanity
async def test_two_tool_calls_in_one_turn_never_interleave_their_started_finished_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given one turn requesting two tool calls, when the task runs,
    then each call's own `on_tool_call_finished` immediately follows
    its own `on_tool_call_started` -- the first call's pair never has
    the second call's `on_tool_call_started` land in between."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    observer = RecordingObserver()
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn([("call-a", "read_file", "{}"), ("call-b", "execute", "{}")]),
            _stop_turn(["done"]),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(), observer=observer)

    result = await run_task("run two tools", deps, task_id="t-p036-2")

    assert result.reason == TerminationReason.TASK_COMPLETE
    tool_events = [
        entry for entry in observer.calls if entry[0].startswith("on_tool_call")
    ]
    assert [name for name, _ in tool_events] == [
        "on_tool_call_started",
        "on_tool_call_finished",
        "on_tool_call_started",
        "on_tool_call_finished",
    ]
    first_started = tool_events[0][1]
    first_finished_call, _ = tool_events[1][1]
    second_started = tool_events[2][1]
    second_finished_call, _ = tool_events[3][1]
    assert first_started.id == first_finished_call.id == "call-a"
    assert second_started.id == second_finished_call.id == "call-b"


async def test_a_passing_verify_call_fires_on_verification_once_between_its_pair_and_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given one turn requesting `verify` (which records a passing
    report) followed by a second, unrelated tool call, when the task
    runs, then `on_verification` fires exactly once, positioned between
    the `verify` call's own `on_tool_call_finished` and the second
    call's `on_tool_call_started` -- the second call never triggers it."""
    report = VerificationReport(task_id="t-p036-3", turn_id=1, commands=(), passed=True)
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch_with_reports([report]))
    observer = RecordingObserver()
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn([("call-1", "verify", "{}"), ("call-2", "read_file", "{}")]),
            _stop_turn(["done"]),
        ]
    )
    deps = _build_deps(client, tmp_path, _registry(), observer=observer)

    result = await run_task("verify then read", deps, task_id="t-p036-3")

    assert result.reason == TerminationReason.TASK_COMPLETE
    relevant = [
        entry
        for entry in observer.calls
        if entry[0]
        in ("on_tool_call_started", "on_tool_call_finished", "on_verification")
    ]
    assert [name for name, _ in relevant] == [
        "on_tool_call_started",
        "on_tool_call_finished",
        "on_verification",
        "on_tool_call_started",
        "on_tool_call_finished",
    ]
    assert relevant[2][1] is report
    assert observer.names().count("on_verification") == 1


async def test_self_critique_skip_path_finishes_its_turn_without_ending_the_task(
    tmp_path: Path,
) -> None:
    """Given a `self_critique_fn` that declines the first turn's
    proposed tool call and approves the second, when the task runs,
    then the declined turn still fires its own `on_turn_finished` --
    but the task's single `on_termination` fires only once, at the very
    end, after the second turn completes it. The declined turn's tool
    call is never dispatched, so neither `on_tool_call_started` nor
    `on_tool_call_finished` fires for it."""
    observer = RecordingObserver()
    client = _ScriptedLoopClient(
        turns=[_tool_turn([("call-1", "read_file", "{}")]), _stop_turn(["done"])]
    )
    decisions = iter([False, True])

    def self_critique_fn(proposal: str, history: list[Message]) -> bool:
        """Pop the next scripted decision: decline, then approve."""
        return next(decisions)

    deps = _build_deps(
        client,
        tmp_path,
        _registry(),
        observer=observer,
        self_critique_fn=self_critique_fn,
    )

    result = await run_task("read greet.py", deps, task_id="t-p036-4")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    names = observer.names()
    assert names.count("on_tool_call_started") == 0
    assert names.count("on_turn_finished") == 2
    assert names.count("on_termination") == 1
    assert names[-1] == "on_termination"
    turn_ids = [
        payload["turn_id"]
        for name, payload in observer.calls
        if name == "on_turn_finished"
    ]
    assert turn_ids == [1, 2]


async def test_require_verification_nudge_turns_finish_without_terminating_until_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `require_verification=True` and a script that nudges once,
    then calls a passing `verify`, then stops again, when the task
    runs, then all three turns fire their own `on_turn_finished`, but
    `on_termination` fires only once -- as the very last call -- once
    the third turn's stop is finally allowed to complete the task."""
    report = VerificationReport(task_id="t-p036-5", turn_id=2, commands=(), passed=True)
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch_with_reports([report]))
    observer = RecordingObserver()
    client = _ScriptedLoopClient(
        turns=[
            _stop_turn(["not done yet"]),
            _tool_turn([("call-1", "verify", "{}")]),
            _stop_turn(["now it's done"]),
        ]
    )
    deps = _build_deps(
        client, tmp_path, _registry(), observer=observer, require_verification=True
    )

    result = await run_task("fix it", deps, task_id="t-p036-5")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    names = observer.names()
    assert names.count("on_turn_finished") == 3
    assert names.count("on_termination") == 1
    assert names[-1] == "on_termination"


async def test_compaction_fold_finishes_the_shared_turn_id_before_the_real_turn_starts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two tool-calling turns that build history past the
    compaction threshold, when the task runs, then the resulting fold's
    own `on_turn_finished` -- sharing its `turn_id` with the real turn
    that follows -- fires before that real turn's own `on_turn_started`
    for that same id, which in turn fires before that real turn's own,
    second `on_turn_finished`: two `on_turn_finished` calls and one
    `on_turn_started` call share one `turn_id`."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    observer = RecordingObserver()
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn([("call-1", "read_file", "{}")], input_tokens=50),
            _tool_turn([("call-2", "read_file", "{}")], input_tokens=80),
            _stop_turn(["carry-forward summary"], input_tokens=50, output_tokens=10),
            _stop_turn(["done"], input_tokens=20),
        ]
    )
    deps = _build_deps(
        client,
        tmp_path,
        _registry(context_window=100),
        observer=observer,
        limits=LoopLimits(max_total_tokens=100_000_000),
    )

    result = await run_task("do it", deps, task_id="t-p036-6")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    finished_id_3 = [
        index
        for index, (name, payload) in enumerate(observer.calls)
        if name == "on_turn_finished" and payload["turn_id"] == 3
    ]
    started_id_3 = [
        index
        for index, (name, payload) in enumerate(observer.calls)
        if name == "on_turn_started" and payload["turn_id"] == 3
    ]
    assert len(finished_id_3) == 2
    assert len(started_id_3) == 1
    assert finished_id_3[0] < started_id_3[0] < finished_id_3[1]


@pytest.mark.cost_regression
async def test_observer_wiring_adds_no_extra_cost(tmp_path: Path) -> None:
    """A task left at `LoopDeps.observer`'s `NullLoopObserver` default
    prices out to the exact same Decimal total as
    `test_p022_loop.py::test_two_turn_task_cost_band`'s own pinned
    band -- proving the observer amendment adds zero extra model calls
    or bookkeeping capable of shifting what a task is billed."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn(
                [("call-1", "read_file", json.dumps({"path": "greet.py"}))],
                input_tokens=1000,
                output_tokens=50,
            ),
            _stop_turn(["done"], input_tokens=1200, output_tokens=20),
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
    )

    result = await run_task("read greet.py", deps, task_id="t-p036-cost")

    assert result.total_usd == Decimal("0.001474")
