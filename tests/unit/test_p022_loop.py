"""Unit tests for the tool-calling agent loop: every termination
predicate, the tool-call round trip through the shared registry, the
self-critique skip path, and the approval/keyboard-interrupt escape
hatches.

Every case drives `run_task` against a small scripted `ProviderClient`
that replays one full turn's event sequence per call -- distinct from
`test_p020_retry.py`'s fake, which scripts retry *attempts* for a
single call -- and against real `read_file` dispatch through
`kestrel.tools.registry`, so the tool round trip is proven against the
actual dispatcher rather than a stand-in for it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, LoopLimits, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
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

pytestmark = [pytest.mark.p022, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry() -> Registry:
    """Build a single-entry `Registry` at the same rates the packaged
    default registry ships, so a cost-regression assertion pins against
    a real, documented price rather than an arbitrary test-only one."""
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
    """One scripted `.complete()` call's outcome: the events to yield,
    or -- when `raises` is set instead -- an exception to raise without
    yielding anything, standing in for a failure mid-turn."""

    events: tuple[StreamEvent, ...] = ()
    raises: Exception | None = None


@dataclass
class _ScriptedLoopClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order. Calling `.complete()` more times than there are scripted
    turns is a test-authoring error and raises `IndexError` --
    every case here scripts exactly as many turns as the loop should
    actually take, so running out of script means the loop kept going
    past where it should have stopped.
    """

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
        """Replay the next scripted turn, ignoring every argument but the count."""
        turn = self.turns[self.call_count]
        self.call_count += 1
        if turn.raises is not None:
            raise turn.raises
        for event in turn.events:
            yield event


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    *,
    limits: LoopLimits = LoopLimits(),
    self_critique_fn: Callable[[str, list[Message]], bool] | None = None,
    approval: ApprovalManager | None = None,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle from real managers and a fresh
    `CostMeter`, scoped to `repo_root`, for one test's `run_task` call."""
    kwargs: dict[str, object] = {}
    if self_critique_fn is not None:
        kwargs["self_critique_fn"] = self_critique_fn
    return LoopDeps(
        client=client,
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=approval if approval is not None else ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        limits=limits,
        **kwargs,  # type: ignore[arg-type]
    )


def _stop_turn(
    text: str, *, input_tokens: int = 100, output_tokens: int = 20
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


def _read_file_turn(
    call_id: str, path: str, *, input_tokens: int = 100, output_tokens: int = 20
) -> _ScriptedTurn:
    """A turn that requests exactly one `read_file` tool call for `path`."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(
                id=call_id, name="read_file", arguments_json=json.dumps({"path": path})
            ),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _tool_messages(history: tuple[Message, ...]) -> list[Message]:
    """Every tool-role message in `history`, in order."""
    return [message for message in history if message["role"] == "tool"]


@pytest.mark.sanity
async def test_zero_tool_calls_on_first_turn_completes_after_one_turn(
    tmp_path: Path,
) -> None:
    """Given a script whose first turn answers with plain text and no
    tool calls, when the task runs, then it ends TASK_COMPLETE after
    exactly one turn."""
    client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    deps = _build_deps(client, tmp_path)

    result = await run_task("say hi", deps, task_id="t-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 1
    assert client.call_count == 1


@pytest.mark.sanity
async def test_one_tool_call_then_stop_completes_after_two_turns_with_result_folded_in(
    tmp_path: Path,
) -> None:
    """Given a script that calls `read_file` once and then stops, when
    the task runs, then it ends TASK_COMPLETE after two turns, and the
    tool's result is folded into history between the two model calls."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[_read_file_turn("call-1", "greet.py"), _stop_turn("done")]
    )
    deps = _build_deps(client, tmp_path)

    result = await run_task("read greet.py", deps, task_id="t-2")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    tool_messages = _tool_messages(result.history)
    assert len(tool_messages) == 1
    assert "print('hi')" in tool_messages[0]["content"]


@pytest.mark.cost_regression
async def test_two_turn_task_cost_band(tmp_path: Path) -> None:
    """The exact two-turn scripted task above, priced at the default
    registry's rates, costs a pinned Decimal total -- any drift in
    either the pricing formula or this loop's own usage bookkeeping
    fails this test.
    """
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _read_file_turn("call-1", "greet.py", input_tokens=1000, output_tokens=50),
            _stop_turn("done", input_tokens=1200, output_tokens=20),
        ]
    )
    deps = _build_deps(client, tmp_path)

    result = await run_task("read greet.py", deps, task_id="t-2-cost")

    # turn 1: (1000 * 0.60 + 50 * 2.20) / 1e6 = 0.000710
    # turn 2: (1200 * 0.60 + 20 * 2.20) / 1e6 = 0.000764
    assert result.total_usd == Decimal("0.001474")


async def test_turn_cap_stops_at_exactly_max_turns_without_a_further_think_call(
    tmp_path: Path,
) -> None:
    """Given `max_turns=2` and a script that never stops on its own,
    when the task runs, then it ends TURN_CAP with `turns_used == 2`
    and never attempts a third model call."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _read_file_turn("call-1", "greet.py"),
            _read_file_turn("call-2", "greet.py"),
        ]
    )
    deps = _build_deps(client, tmp_path, limits=LoopLimits(max_turns=2))

    result = await run_task("keep reading forever", deps, task_id="t-3")

    assert result.reason == TerminationReason.TURN_CAP
    assert result.turns_used == 2
    assert client.call_count == 2


async def test_token_cap_stops_on_the_turn_that_crosses_it_not_one_turn_late(
    tmp_path: Path,
) -> None:
    """Given `max_total_tokens=1000` and a first turn whose own usage
    already crosses it, when the task runs, then it ends TOKEN_CAP
    immediately after that turn's usage is folded in -- never attempting
    a second model call the cap should have prevented."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _read_file_turn("call-1", "greet.py", input_tokens=900, output_tokens=200),
        ]
    )
    deps = _build_deps(client, tmp_path, limits=LoopLimits(max_total_tokens=1000))

    result = await run_task("read greet.py", deps, task_id="t-4")

    assert result.reason == TerminationReason.TOKEN_CAP
    assert result.turns_used == 1
    assert client.call_count == 1


async def test_wall_clock_cap_stops_between_turns(tmp_path: Path) -> None:
    """Given an injected clock that reports elapsed time past
    `max_wall_clock_s` at the start of a second turn, when the task
    runs, then it ends WALL_CLOCK_CAP without attempting that turn."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(turns=[_read_file_turn("call-1", "greet.py")])
    deps = _build_deps(client, tmp_path, limits=LoopLimits(max_wall_clock_s=100.0))
    clock_values = iter([0.0, 0.0, 150.0])

    def clock_fn() -> float:
        """Pop the next scripted timestamp."""
        return next(clock_values)

    result = await run_task("read greet.py", deps, task_id="t-5", clock_fn=clock_fn)

    assert result.reason == TerminationReason.WALL_CLOCK_CAP
    assert result.turns_used == 1
    assert client.call_count == 1


async def test_context_overflow_mid_think_ends_the_task_without_raising(
    tmp_path: Path,
) -> None:
    """Given a `ContextOverflowError` raised by the client during a
    turn, when the task runs, then it ends CONTEXT_OVERFLOW rather than
    letting the exception escape `run_task`."""
    client = _ScriptedLoopClient(
        turns=[
            _ScriptedTurn(
                raises=ContextOverflowError(
                    "too big", model_id=_MODEL_ID, backend=_BACKEND
                )
            )
        ]
    )
    deps = _build_deps(client, tmp_path)

    result = await run_task("do something huge", deps, task_id="t-6")

    assert result.reason == TerminationReason.CONTEXT_OVERFLOW
    assert result.turns_used == 1


async def test_self_critique_declining_skips_act_and_continues_to_a_real_next_turn(
    tmp_path: Path,
) -> None:
    """Given a `self_critique_fn` that declines the first turn's
    proposed tool call and approves the second turn's, when the task
    runs, then the declined turn's tool is never dispatched, a synthetic
    explanation is folded into history in its place, and the loop
    reaches a real second turn that completes the task."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[_read_file_turn("call-1", "greet.py"), _stop_turn("done")]
    )
    decisions = iter([False, True])

    def self_critique_fn(proposal: str, history: list[Message]) -> bool:
        """Pop the next scripted decision: decline, then approve."""
        return next(decisions)

    deps = _build_deps(client, tmp_path, self_critique_fn=self_critique_fn)

    result = await run_task("read greet.py", deps, task_id="t-7")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    tool_messages = _tool_messages(result.history)
    assert len(tool_messages) == 1
    assert "print('hi')" not in tool_messages[0]["content"]


async def test_unregistered_tool_name_returns_a_framed_error_and_the_loop_continues(
    tmp_path: Path,
) -> None:
    """Given a turn that requests a tool name the registry does not
    know, when the task runs, then the dispatcher's own framed error
    result is folded into history and the loop continues to a real next
    turn rather than crashing."""
    client = _ScriptedLoopClient(
        turns=[
            _ScriptedTurn(
                events=(
                    ToolCallEvent(id="call-1", name="frobnicate", arguments_json="{}"),
                    UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=0),
                    StopEvent(reason="tool_use"),
                )
            ),
            _stop_turn("done"),
        ]
    )
    deps = _build_deps(client, tmp_path)

    result = await run_task("call a made-up tool", deps, task_id="t-8")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    tool_messages = _tool_messages(result.history)
    assert "frobnicate" in tool_messages[0]["content"]


async def test_keyboard_interrupt_during_tool_execution_returns_user_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `KeyboardInterrupt` raised while a tool call is being
    dispatched, when the task runs, then it ends USER_STOP with the
    turns and cost accumulated up to that point, rather than the
    exception escaping `run_task`."""
    import kestrel.agent.loop as loop_module

    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(turns=[_read_file_turn("call-1", "greet.py")])
    deps = _build_deps(client, tmp_path)

    def _raise_interrupt(*_args: object, **_kwargs: object) -> None:
        """Stand in for `dispatch`, simulating an interrupt mid-tool-call."""
        raise KeyboardInterrupt

    monkeypatch.setattr(loop_module, "dispatch", _raise_interrupt)

    result = await run_task("read greet.py", deps, task_id="t-9")

    assert result.reason == TerminationReason.USER_STOP
    assert result.turns_used == 1
    assert result.total_usd == Decimal("0")


async def test_two_tool_calls_in_one_turn_dispatch_in_order_before_the_next_think_call(
    tmp_path: Path,
) -> None:
    """Given one turn that requests two `read_file` calls, when the
    task runs, then both are dispatched in the order the model requested
    them, and both results are present in history before the next model
    call is made."""
    (tmp_path / "a.py").write_text("A_CONTENT", encoding="utf-8")
    (tmp_path / "b.py").write_text("B_CONTENT", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _ScriptedTurn(
                events=(
                    ToolCallEvent(
                        id="call-a",
                        name="read_file",
                        arguments_json=json.dumps({"path": "a.py"}),
                    ),
                    ToolCallEvent(
                        id="call-b",
                        name="read_file",
                        arguments_json=json.dumps({"path": "b.py"}),
                    ),
                    UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=0),
                    StopEvent(reason="tool_use"),
                )
            ),
            _stop_turn("done"),
        ]
    )
    deps = _build_deps(client, tmp_path)

    result = await run_task("read both files", deps, task_id="t-10")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 2
    tool_messages = _tool_messages(result.history)
    assert len(tool_messages) == 2
    assert "A_CONTENT" in tool_messages[0]["content"]
    assert "B_CONTENT" in tool_messages[1]["content"]


async def test_denied_approval_becomes_a_framed_refusal_instead_of_crashing_the_loop(
    tmp_path: Path,
) -> None:
    """Given a turn that requests an `execute` call classified as
    destructive, and an `ApprovalManager` that denies it, when the task
    runs, then the resulting `ApprovalDenied` is caught and turned into
    a framed refusal result rather than escaping `run_task`, and the
    loop continues to a real next turn."""
    client = _ScriptedLoopClient(
        turns=[
            _ScriptedTurn(
                events=(
                    ToolCallEvent(
                        id="call-1",
                        name="execute",
                        arguments_json=json.dumps({"cmd": ["rm", "-rf", "build"]}),
                    ),
                    UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=0),
                    StopEvent(reason="tool_use"),
                )
            ),
            _stop_turn("done"),
        ]
    )
    approval = ApprovalManager(decide_fn=lambda _request: "deny")
    deps = _build_deps(client, tmp_path, approval=approval)

    result = await run_task("clean the build directory", deps, task_id="t-11")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    tool_messages = _tool_messages(result.history)
    assert "Delete: rm -rf build" in tool_messages[0]["content"]


async def test_token_cap_applies_even_when_self_critique_declines_the_turn(
    tmp_path: Path,
) -> None:
    """Given a `self_critique_fn` that declines every turn, and a token
    cap the first (declined) turn's own usage already crosses, when the
    task runs, then it ends TOKEN_CAP right after that turn's usage is
    recorded -- the cap is enforced on the decline path exactly as it is
    on the ordinary path, so a repeatedly-declined task cannot spend
    tokens forever."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _read_file_turn("call-1", "greet.py", input_tokens=900, output_tokens=200)
        ]
    )
    deps = _build_deps(
        client,
        tmp_path,
        limits=LoopLimits(max_total_tokens=1000),
        self_critique_fn=lambda proposal, history: False,
    )

    result = await run_task("read greet.py", deps, task_id="t-12")

    assert result.reason == TerminationReason.TOKEN_CAP
    assert result.turns_used == 1
    assert client.call_count == 1


async def test_task_complete_on_last_turn_honors_terminal_action_over_turn_cap(
    tmp_path: Path,
) -> None:
    """Given `max_turns=2` and a script that calls a tool on turn 1 and
    then produces no tool calls (TASK_COMPLETE) on turn 2, when the task
    runs, then it ends TASK_COMPLETE rather than being overwritten by
    TURN_CAP on that last allowed turn."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    client = _ScriptedLoopClient(
        turns=[
            _read_file_turn("call-1", "greet.py"),
            _stop_turn("done")
        ]
    )
    deps = _build_deps(client, tmp_path, limits=LoopLimits(max_turns=2))

    result = await run_task("read greet.py", deps, task_id="t-13")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
