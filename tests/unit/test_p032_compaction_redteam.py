"""Red-team proof that folding an old, hostile tool result into a
compaction summary can never smuggle its own claims into the loop's
control flow.

`compact_history` performs no parsing of its own on either the older
tail it summarizes or the text a model returns for it: the older tail
is forwarded to the client as plain message content, verbatim, and
whatever text comes back is wrapped, unmodified, into one new
`"system"`-role message. Even in the worst case -- a confused model
that echoes a hostile payload's own claims straight back as its own
"summary" -- the agent loop still only reacts to structured
`StreamEvent`s from its own real Think calls, never to substrings of
any message's content, so a folded-in echo can neither forge a tool
call nor end the task early.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.compaction import compact_history
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
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.security.framing import frame_untrusted
from kestrel.tools.registry import ToolResult

pytestmark = [pytest.mark.p032, pytest.mark.unit, pytest.mark.redteam]

_MODEL_ID = "glm-5.2"
_HOSTILE_CASE_ID = "fake_destructive_action_success"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


@pytest.fixture(scope="session")
def hostile_case() -> InjectionCase:
    """The corpus case used throughout this suite: a fake tool result
    claiming a destructive command "succeeded" and instructing the
    model to exfiltrate an SSH key -- old enough, in every scenario
    here, to be a candidate for compaction's own fold."""
    return _find_case(_HOSTILE_CASE_ID)


def _registry(*, context_window: int) -> Registry:
    """A single-entry `Registry` with a small `context_window`, so this
    suite's own scripted token counts cross the compaction threshold."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend="openrouter",
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
class _EchoClient:
    """A `ProviderClient` that always replies with one fixed,
    caller-chosen text -- standing in for a worst-case model that
    echoes back whatever a hostile payload told it to say, so this
    suite can prove that even that outcome is harmless."""

    reply_text: str
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
        """Record this call's own messages, then yield the fixed reply."""
        self.call_count += 1
        self.received_messages.append(list(messages))
        yield TextDelta(text=self.reply_text)
        yield UsageEvent(input_tokens=10, output_tokens=5, cached_tokens=0)
        yield StopEvent(reason="end_turn")


async def test_compact_history_forwards_a_hostile_payload_as_inert_data(
    hostile_case: InjectionCase,
) -> None:
    """Given an older tail containing a hostile corpus payload framed
    exactly as a real, aging tool result would be, when `compact_history`
    folds it, then the client receives that framed text completely
    unmodified (no stripping, no re-parsing) among the messages it
    summarizes, and even a scripted reply that is itself the raw,
    unframed payload text comes back as nothing more than one plain
    `"system"`-role message's own content -- never treated as an
    instruction by `compact_history` itself, and never causing it to
    return anything but the ordinary `[summary_message, *kept_tail]`
    shape."""
    framed_payload = frame_untrusted(
        hostile_case.payload, source=hostile_case.source, origin="cat old-output.txt"
    )
    history: list[Message] = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "reading the log now"},
        {"role": "tool", "content": framed_payload},
        {"role": "user", "content": "keep going"},
        {"role": "assistant", "content": "on it"},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "still working"},
    ]
    client = _EchoClient(reply_text=hostile_case.payload)

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=None
    )

    assert client.call_count == 1
    (sent,) = client.received_messages
    assert framed_payload in [message["content"] for message in sent]
    assert result_history[0]["role"] == "system"
    assert result_history[0]["content"].startswith(hostile_case.payload)
    assert len(result_history) == 5
    assert result_history[1:] == history[-4:]


@dataclass
class _ScriptedTurn:
    """One scripted `.complete()` call's outcome -- the events to yield."""

    events: tuple[StreamEvent, ...] = ()


@dataclass
class _ScriptedLoopClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order -- the compaction call is, from this client's perspective,
    just another `.complete()` call in the script."""

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


def _tool_turn(call_id: str, *, input_tokens: int) -> _ScriptedTurn:
    """A turn that requests exactly one tool call."""
    return _ScriptedTurn(
        events=(
            ToolCallEvent(id=call_id, name="read_file", arguments_json="{}"),
            UsageEvent(input_tokens=input_tokens, output_tokens=0, cached_tokens=0),
            StopEvent(reason="tool_use"),
        )
    )


def _stop_turn(text: str, *, input_tokens: int) -> _ScriptedTurn:
    """A turn that answers with plain text and requests no tool calls."""
    return _ScriptedTurn(
        events=(
            TextDelta(text=text),
            UsageEvent(input_tokens=input_tokens, output_tokens=0, cached_tokens=0),
            StopEvent(reason="end_turn"),
        )
    )


def _dispatch_returning_hostile_content(
    hostile_content: str,
) -> object:
    """Stand in for `kestrel.agent.loop.dispatch`: the first tool call
    it sees returns the hostile, framed payload as if it were that
    call's own real result; every other call succeeds unremarkably."""

    def _dispatch(
        event: ToolCallEvent, *, repo_root: Path, **context: object
    ) -> ToolResult:
        if event.id == "call-1":
            return ToolResult(tool_call_id=event.id, content=hostile_content)
        return ToolResult(tool_call_id=event.id, content=f"ran {event.name}")

    return _dispatch


async def test_a_compaction_echo_of_a_hostile_payload_never_ends_the_task_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostile_case: InjectionCase
) -> None:
    """Given a task whose first tool result carries the hostile payload,
    old enough by turn 3 to be folded into a compaction summary, and a
    worst-case compaction reply that is itself the raw, unframed
    payload text (as if the model had been talked into echoing it
    verbatim), when the task runs, then it still reaches a real fourth
    call and only ends `TASK_COMPLETE` once *that* turn -- not the
    compaction call -- reports no tool calls; the payload's own claim
    of a completed destructive action and its instruction to keep going
    are never treated as a real signal by the loop."""
    framed_payload = frame_untrusted(
        hostile_case.payload, source=hostile_case.source, origin="cat old-output.txt"
    )
    monkeypatch.setattr(
        loop_module, "dispatch", _dispatch_returning_hostile_content(framed_payload)
    )
    client = _ScriptedLoopClient(
        turns=[
            _tool_turn("call-1", input_tokens=50),
            _tool_turn("call-2", input_tokens=80),
            _stop_turn(hostile_case.payload, input_tokens=50),
            _stop_turn("done", input_tokens=20),
        ]
    )
    deps = LoopDeps(
        client=client,
        registry=_registry(context_window=100),
        model_id=_MODEL_ID,
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        limits=LoopLimits(max_total_tokens=100_000_000),
    )

    result = await run_task("do it", deps, task_id="rt-compaction-echo")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert client.call_count == 4
    assert result.turns_used == 3
