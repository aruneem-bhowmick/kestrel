"""System test: `run_task`'s `kb_context` parameter genuinely seeds a
fresh task's history once, at task start, and stays there unchanged on
every later outgoing model call -- proving it is seeded once and never
re-derived turn to turn.

Uses a scripted fake `ProviderClient` recording its own `messages`
argument on every call, matching `test_p045_effort_reaches_provider.py`'s
own scripted-client pattern -- what is under test here is the value
`run_task`'s own history seeding produces, not a real backend's request
body.
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

pytestmark = [pytest.mark.p061, pytest.mark.system]

_MODEL_ID = "glm-5.2"
_KB_CONTEXT = (
    "<<<UNTRUSTED:kb:nomic-embed-text>>>\n"
    "- prefer tabs in this repo\n"
    "<<<END_UNTRUSTED>>>"
)


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
class _MessagesRecordingClient:
    """A `ProviderClient` that replays one `_ScriptedTurn` per call, in
    order, recording the full `messages` argument it was actually called
    with on every single call."""

    turns: Sequence[_ScriptedTurn]
    recorded_messages: list[Sequence[Message]] = field(default_factory=list)
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
        """Record `messages`, then replay the next scripted turn's
        events."""
        self.recorded_messages.append(messages)
        turn = self.turns[self.call_count]
        self.call_count += 1
        for event in turn.events:
            yield event


def _stop_turn(text: str) -> _ScriptedTurn:
    """A turn that answers with plain text and requests no tool calls,
    ending the task."""
    return _ScriptedTurn(
        events=(
            TextDelta(text=text),
            UsageEvent(input_tokens=100, output_tokens=20, cached_tokens=0),
            StopEvent(reason="end_turn"),
        )
    )


def _build_deps(client: _MessagesRecordingClient, repo_root: Path) -> LoopDeps:
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
    )


async def test_kb_context_is_seeded_once_right_after_the_task_description(
    tmp_path: Path,
) -> None:
    """Given `run_task(..., kb_context=<a framed block>)` scripted to run
    two turns before completing, when the task runs, then `history[1]`
    is exactly that framed block as a user-role message, and every
    outgoing `.complete()` call -- including the second turn's -- carries
    that same message unchanged, proving it is seeded once and never
    re-derived."""
    client = _MessagesRecordingClient(turns=[_stop_turn("first"), _stop_turn("second")])
    deps = _build_deps(client, tmp_path)

    result = await run_task(
        "implement the feature",
        deps,
        task_id="sys-p061-kb-context",
        kb_context=_KB_CONTEXT,
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.history[1] == {"role": "user", "content": _KB_CONTEXT}

    kb_message: Message = {"role": "user", "content": _KB_CONTEXT}
    for call_messages in client.recorded_messages:
        assert kb_message in call_messages


async def test_kb_context_left_unset_preserves_prior_history_shape(
    tmp_path: Path,
) -> None:
    """Given `run_task` called without `kb_context` at all (the
    default), when the task runs, then its history holds only the
    task-description seed and the turn's own reply -- byte-identical to
    every `run_task` call written before this parameter existed, with no
    extra message inserted."""
    client = _MessagesRecordingClient(turns=[_stop_turn("done")])
    deps = _build_deps(client, tmp_path)

    result = await run_task("implement the feature", deps, task_id="sys-p061-no-kb")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.history == (
        {"role": "user", "content": "implement the feature"},
        {"role": "assistant", "content": "done"},
    )
