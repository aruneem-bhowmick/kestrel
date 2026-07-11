"""Red-team proof that a hostile payload arriving as the model's own
assistant text -- as if it were a prior tool result being echoed back
into the conversation -- can never be mistaken by the agent loop for a
real tool call or a real termination signal.

The loop only ever reacts to structured `StreamEvent` objects: a
`ToolCallEvent` triggers dispatch, and the *absence* of any
`ToolCallEvent` in a turn is what ends a task `TASK_COMPLETE`. Neither
decision ever inspects the text content of a `TextDelta` for meaning,
so no string embedded in that text -- however it is phrased -- can
forge a tool call the loop did not actually receive as a structured
event, or talk the loop into stopping on a turn it would not otherwise
have stopped on.
"""

from __future__ import annotations

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
from kestrel.provider.events import StopEvent, StreamEvent, TextDelta, UsageEvent
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.security.framing import frame_untrusted

pytestmark = [pytest.mark.p022, pytest.mark.unit, pytest.mark.redteam]

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
def hostile_echo_case() -> InjectionCase:
    """The corpus case used to prove a hostile payload, echoed back as
    plain assistant text, never derails the loop's own control flow."""
    return _find_case(_HOSTILE_CASE_ID)


def _registry() -> Registry:
    """A single-entry `Registry` sufficient to price one turn."""
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
class _SingleTurnClient:
    """A `ProviderClient` that yields one fixed event sequence for its
    only expected call, raising `IndexError` if called again -- this
    test's whole point is that the loop never asks for a second turn
    over a hostile text reply that never actually requested a tool."""

    events: tuple[StreamEvent, ...]
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
        """Yield the one scripted sequence, refusing a second call."""
        if self.call_count >= 1:
            raise IndexError("the loop asked for a second turn it should not have")
        self.call_count += 1
        for event in self.events:
            yield event


async def test_hostile_text_reply_never_becomes_a_tool_call_or_termination_signal(
    tmp_path: Path, hostile_echo_case: InjectionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a turn whose assistant text is a hostile corpus payload
    rendered exactly as a real tool result would be -- as if the model
    were quoting a prior tool's framed output back into its own reply --
    and no real `ToolCallEvent` in that turn, when the task runs, then
    it ends TASK_COMPLETE (the only correct reading of "no tool call
    this turn") without the tool dispatcher ever being invoked, proving
    the payload's own claims ("completed successfully", instructions to
    keep going) were never parsed as a real signal.
    """
    import kestrel.agent.loop as loop_module

    echoed = frame_untrusted(
        hostile_echo_case.payload,
        source="tool_stdout",
        origin="cat malicious-output.txt",
    )
    client = _SingleTurnClient(
        events=(
            TextDelta(text=echoed),
            UsageEvent(input_tokens=80, output_tokens=40, cached_tokens=0),
            StopEvent(reason="end_turn"),
        )
    )

    def _dispatch_should_never_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "dispatch must never be called: no real tool call was made"
        )

    monkeypatch.setattr(loop_module, "dispatch", _dispatch_should_never_be_called)

    deps = LoopDeps(
        client=client,
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
    )

    result = await run_task("summarize the last command's output", deps, task_id="rt-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 1
    assert client.call_count == 1
    assert echoed in result.history[-1]["content"]
