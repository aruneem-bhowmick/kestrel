"""Unit tests for `kestrel.agent.loop.resume_task`'s `inject_message`
parameter: a new user-role message folded into a prior task's loaded
history right before the resumed drive begins, and the resulting
ability to resume a task that already reached `TASK_COMPLETE` rather
than only one a cap or crash halted mid-run.

Reuses `test_p029_loop_session.py`'s scripted-`ProviderClient` and
monkeypatched-`dispatch` patterns, since what is under test here is
`resume_task`'s own history seeding, not a real model or tool call.
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

pytestmark = [pytest.mark.p045, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


def _registry() -> Registry:
    """A single-entry `Registry`, matching this suite's siblings."""
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
    """A turn that requests exactly one tool call named `name` -- used to
    halt a task at `TURN_CAP` rather than let it complete naturally, so
    a later `resume_task` call has real journaled state to continue."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(id=call_id, name=name, arguments_json="{}"),
            UsageEvent(
                input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
            ),
            StopEvent(reason="tool_use"),
        )
    )


def _fake_dispatch(event: object, *, repo_root: Path, **context: object) -> ToolResult:
    """Stand in for `kestrel.agent.loop.dispatch`: succeeds
    unconditionally without running any real tool."""
    name = getattr(event, "name", "unknown")
    call_id = getattr(event, "id", "unknown")
    return ToolResult(tool_call_id=call_id, content=f"ran {name}")


def _build_deps(
    client: _ScriptedLoopClient,
    repo_root: Path,
    *,
    limits: LoopLimits = LoopLimits(),
    session: SessionManager | None = None,
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle scoped to `repo_root`, matching this
    suite's siblings."""
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
    )


async def test_inject_message_is_folded_in_right_after_the_loaded_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task halted at TURN_CAP after one journaled turn, when it
    is resumed with `inject_message` set, then the resumed drive's
    history holds the injected message as the very next entry after the
    loaded history, ahead of the new turn's own reply, and the final
    result's history contains it exactly once."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    first_client = _ScriptedLoopClient(turns=[_tool_turn("call-1", "read_file")])
    session = SessionManager(repo_root=tmp_path, task_id="t-inject-1")
    first_deps = _build_deps(
        first_client, tmp_path, limits=LoopLimits(max_turns=1), session=session
    )
    first_result = await run_task("do the first part", first_deps, task_id="t-inject-1")
    assert first_result.reason == TerminationReason.TURN_CAP

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    second_session = SessionManager(repo_root=tmp_path, task_id="t-inject-1")
    second_deps = _build_deps(
        second_client,
        tmp_path,
        limits=LoopLimits(max_turns=10),
        session=second_session,
    )

    resumed_result = await resume_task(
        "t-inject-1", second_deps, inject_message="now also do the second part"
    )

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    loaded_len = len(first_result.history)
    assert resumed_result.history[loaded_len] == {
        "role": "user",
        "content": "now also do the second part",
    }
    assert resumed_result.history[loaded_len + 1] == {
        "role": "assistant",
        "content": "done",
    }
    assert (
        sum(
            1
            for message in resumed_result.history
            if message == {"role": "user", "content": "now also do the second part"}
        )
        == 1
    )


async def test_inject_message_is_journaled_as_the_resumed_turns_own_input_not_the_priors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the same halted-then-resumed task, when the resumed turn
    completes, then the injected message is captured only in the
    *resumed* session's own new `TurnRecord` -- the original session's
    already-written journal is untouched by a later call's own
    `inject_message`."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    first_client = _ScriptedLoopClient(turns=[_tool_turn("call-1", "read_file")])
    session = SessionManager(repo_root=tmp_path, task_id="t-inject-2")
    first_deps = _build_deps(
        first_client, tmp_path, limits=LoopLimits(max_turns=1), session=session
    )
    await run_task("do the first part", first_deps, task_id="t-inject-2")
    original_record_count = len(session.records)

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    second_session = SessionManager(repo_root=tmp_path, task_id="t-inject-2")
    second_deps = _build_deps(
        second_client,
        tmp_path,
        limits=LoopLimits(max_turns=10),
        session=second_session,
    )

    await resume_task(
        "t-inject-2", second_deps, inject_message="steer the task differently"
    )

    assert len(session.records) == original_record_count
    new_record = second_session.records[-1]
    assert new_record.message_deltas[0] == {
        "role": "user",
        "content": "steer the task differently",
    }


async def test_resume_task_continues_a_task_that_already_reached_task_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task that already ended TASK_COMPLETE (not one halted by
    a cap or a crash), when `resume_task` is called against it with a
    new `inject_message`, then it drives a genuine further turn and
    completes again -- `resume_task` places no precondition on the
    loaded state's prior termination reason."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    first_client = _ScriptedLoopClient(turns=[_stop_turn("first answer")])
    session = SessionManager(repo_root=tmp_path, task_id="t-inject-3")
    first_deps = _build_deps(first_client, tmp_path, session=session)
    first_result = await run_task(
        "answer the question", first_deps, task_id="t-inject-3"
    )
    assert first_result.reason == TerminationReason.TASK_COMPLETE

    second_client = _ScriptedLoopClient(turns=[_stop_turn("second answer")])
    second_session = SessionManager(repo_root=tmp_path, task_id="t-inject-3")
    second_deps = _build_deps(second_client, tmp_path, session=second_session)

    resumed_result = await resume_task(
        "t-inject-3", second_deps, inject_message="one more thing"
    )

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 2
    assert resumed_result.history[-1] == {
        "role": "assistant",
        "content": "second answer",
    }


@pytest.mark.sanity
async def test_inject_message_left_unset_preserves_prior_resume_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a halted task resumed without passing `inject_message` at
    all, when it completes, then its history holds only the messages
    the loaded session and the new turn itself produced -- no extra
    message appears, matching every caller written before this
    parameter existed."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    first_client = _ScriptedLoopClient(turns=[_stop_turn("partial")])
    session = SessionManager(repo_root=tmp_path, task_id="t-inject-4")
    first_deps = _build_deps(
        first_client, tmp_path, limits=LoopLimits(max_turns=1), session=session
    )
    first_result = await run_task("do it", first_deps, task_id="t-inject-4")

    second_client = _ScriptedLoopClient(turns=[_stop_turn("done")])
    second_deps = _build_deps(second_client, tmp_path, limits=LoopLimits(max_turns=10))

    resumed_result = await resume_task("t-inject-4", second_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert len(resumed_result.history) == len(first_result.history) + 1
    assert resumed_result.history[-1] == {"role": "assistant", "content": "done"}


async def test_compaction_on_the_first_resumed_turn_journals_the_injected_message_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a task halted with enough recorded input tokens to trigger
    compaction the instant it resumes, and resumed with `inject_message`
    set, when compaction folds the loaded history (injected message
    included) before the first new turn ever runs, then the injected
    message is captured exactly once across the resumed session's own
    journal -- by the compaction fold's own record, which already
    covers the whole post-fold history -- rather than a second time by
    the real turn that follows it."""
    monkeypatch.setattr(loop_module, "dispatch", _fake_dispatch)
    # Only the second turn's own recorded input tokens are large: a
    # mid-task compaction check reads the *most recently recorded*
    # turn's tokens, so keeping the first turn small means compaction
    # stays quiet until TURN_CAP has already stopped this call, priming
    # it to fire the instant the resumed call's own first pre-check runs.
    first_client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", "read_file"),
            _tool_turn("call-2", "read_file", input_tokens=150_000),
        ]
    )
    session = SessionManager(repo_root=tmp_path, task_id="t-inject-compact")
    first_deps = _build_deps(
        first_client, tmp_path, limits=LoopLimits(max_turns=2), session=session
    )
    first_result = await run_task(
        "do the first part", first_deps, task_id="t-inject-compact"
    )
    assert first_result.reason == TerminationReason.TURN_CAP

    injected = "now also do the second part"
    second_client = _ScriptedLoopClient(
        turns=[_stop_turn("carry-forward summary"), _stop_turn("done")]
    )
    second_session = SessionManager(repo_root=tmp_path, task_id="t-inject-compact")
    second_deps = _build_deps(
        second_client,
        tmp_path,
        limits=LoopLimits(max_turns=10),
        session=second_session,
    )

    resumed_result = await resume_task(
        "t-inject-compact", second_deps, inject_message=injected
    )

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    # Two client calls: the compaction summary, then the real turn --
    # proof compaction actually fired rather than taking its no-op path.
    assert second_client.call_count == 2
    occurrences = sum(
        1
        for record in second_session.records
        for message in record.message_deltas
        if message == {"role": "user", "content": injected}
    )
    assert occurrences == 1
