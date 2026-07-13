"""Drives one task through a tool-calling loop until it finishes or a
termination predicate trips.

Each iteration repeats the same shape: a pre-flight check against the
configured caps, a model call offering the full tool set, a
self-critique pass over what the model proposed, dispatching any
requested tool calls (approval-gated wherever a tool itself requires
it) through the shared registry, and finally folding the turn's token
usage into the running total and deciding whether to keep going.
`run_task` is the only entry point; everything it needs -- the
provider client, the model registry, the approval and undo managers,
the cost meter, and the per-task limits -- arrives through one
`LoopDeps` bundle, so nothing here reaches for global state or
constructs its own collaborators.

Whether a turn with no requested tool calls actually ends the task is
itself configurable: `LoopDeps.require_verification` (opt-in,
default-`False`) withholds `TASK_COMPLETE` from that turn until the
most recent `verify` tool call recorded in `deps.verification_reports`
passed, nudging the model to keep going instead of letting it declare
victory on its own say-so. Every caller that leaves the field at its
default sees exactly the prior, ungated behavior.

Deliberately out of scope for this module, each a real gap rather than
an oversight:

- No plan/fast mode switching -- every model call runs at a single,
  fixed effort level.
- No real self-critique model call -- `LoopDeps.self_critique_fn`
  defaults to always approving; the injection point exists so a real
  cheap-model check can be plugged in later without changing this
  module's control flow.
- No compaction -- a context-window overflow ends the task outright
  rather than being recovered from by trimming or summarizing history.
- No soft-cap degradation -- `LoopLimits` are hard stops; there is no
  reduced-cost fallback as a limit is approached.
- No artifact persistence -- `LoopResult` is a plain, in-memory value,
  never written to disk.
- No subagents -- `run_task` drives exactly one flat loop and never
  spawns a nested one.
- No continuity across separate `run_task` calls -- each call starts
  its own history seeded fresh from `task_description`; nothing here
  resumes a previous call's conversation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

from kestrel.cost.meter import CostMeter
from kestrel.kestrel_md import KestrelMd
from kestrel.managers.approval import ApprovalDenied, ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Message, ProviderClient
from kestrel.provider.cache import build_stable_prefix, mark_cache_breakpoints
from kestrel.provider.errors import ContextOverflowError
from kestrel.provider.events import (
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from kestrel.provider.retry import complete_with_retry
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.framing import frame_untrusted
from kestrel.tools.registry import ToolResult, all_schemas, dispatch
from kestrel.tools.verify import VerificationReport

_SYSTEM_PROMPT: Final[str] = (
    "You are Kestrel, an autonomous coding agent. Use the tools offered "
    "to you -- read_file, search, execute, edit_file -- to carry out the "
    "task you are given, then stop calling tools once it is complete."
)

_SELF_CRITIQUE_SKIP_CONTENT: Final[str] = (
    "The proposed action was not approved by self-critique and was not "
    "carried out. Reconsider the task and propose a different next step."
)

_VERIFICATION_REQUIRED_CONTENT: Final[str] = (
    "The task is not yet complete: call `verify` and make sure it "
    "passes before declaring the task done."
)


@dataclass(frozen=True, slots=True)
class LoopLimits:
    """Hard caps bounding a single `run_task` call.

    Attributes:
        max_turns: The most model calls one task may make before it is
            stopped with `TerminationReason.TURN_CAP`.
        max_total_tokens: The most cumulative input-plus-output tokens,
            summed across every turn, one task may spend before it is
            stopped with `TerminationReason.TOKEN_CAP`.
        max_wall_clock_s: The longest one task may run, measured from
            its first turn, before it is stopped with
            `TerminationReason.WALL_CLOCK_CAP`.
    """

    max_turns: int = 25
    max_total_tokens: int = 200_000
    max_wall_clock_s: float = 900.0


class TerminationReason(StrEnum):
    """Why a `run_task` call stopped.

    Every member is reachable and, for a given call, mutually exclusive
    with the rest -- exactly one names the reason a task ended.
    """

    TASK_COMPLETE = "task_complete"
    TURN_CAP = "turn_cap"
    TOKEN_CAP = "token_cap"
    WALL_CLOCK_CAP = "wall_clock_cap"
    CONTEXT_OVERFLOW = "context_overflow"
    USER_STOP = "user_stop"


@dataclass(frozen=True, slots=True)
class LoopResult:
    """The outcome of one `run_task` call.

    Attributes:
        reason: Which termination predicate ended the task.
        turns_used: How many model calls the task actually made.
        total_usd: The task's total priced cost, read from
            `LoopDeps.meter` at the moment the task ended.
        history: The full conversation, in order, as it stood when the
            task ended -- including every tool result folded in along
            the way.
    """

    reason: TerminationReason
    turns_used: int
    total_usd: Decimal
    history: tuple[Message, ...]


def _default_self_critique(proposal: str, history: list[Message]) -> bool:
    """Always approve -- the no-op default `LoopDeps.self_critique_fn`
    stands in until a real cheap-model check replaces it."""
    return True


@dataclass
class LoopDeps:
    """The injectable bundle of collaborators one `run_task` call needs.

    Attributes:
        client: Streams each turn's model call.
        registry: Resolves `model_id` to its priced registry entry.
        model_id: Which registry entry every turn in this task is sent to.
        repo_root: The repository a dispatched tool call acts on.
        approval: Gates a tool's own destructive actions.
        undo: Journals a tool's own file mutations.
        meter: Accumulates every turn's priced cost.
        limits: The hard caps this task must not exceed.
        self_critique_fn: Sanity-checks each turn's proposed action
            before it is acted on; returning `False` drops the
            proposal instead of carrying it out. Defaults to always
            approving.
        require_verification: When `True`, a turn that requests no
            tool calls only ends the task once the most recent report
            in `verification_reports` passed; otherwise the loop keeps
            going with a nudge instead of finishing. Defaults to
            `False`, in which case a no-tool-calls turn always ends
            the task exactly as it did before this field existed.
        verification_reports: Every `VerificationReport` a `verify`
            tool call has recorded for this task so far, oldest first.
            Threaded into `dispatch` as that tool's own `report_sink`,
            so it fills in as the task runs; `require_verification`
            reads only its last entry.
        kestrel_md: The target repo's project-memory file, loaded once
            by the caller before the task starts and never reloaded
            mid-task -- keeping it fixed for a whole task is what lets
            the leading prefix built from it stay byte-identical across
            every turn. `None` when the repo has no `KESTREL.md`.
    """

    client: ProviderClient
    registry: Registry
    model_id: str
    repo_root: Path
    approval: ApprovalManager
    undo: UndoManager
    meter: CostMeter
    limits: LoopLimits = field(default_factory=LoopLimits)
    self_critique_fn: Callable[[str, list[Message]], bool] = field(
        default=_default_self_critique
    )
    require_verification: bool = False
    verification_reports: list[VerificationReport] = field(default_factory=list)
    kestrel_md: KestrelMd | None = None


def _split_events(
    events: Sequence[StreamEvent],
) -> tuple[str, list[ToolCallEvent], UsageEvent]:
    """Fold one turn's raw event stream into its assistant text, every
    requested tool call (in the order the model made them), and its
    closing usage event.

    `events` is assumed to already satisfy the normalized stream
    grammar every provider adapter guarantees, so a `UsageEvent` is
    always present by the time this returns.
    """
    text_chunks: list[str] = []
    tool_calls: list[ToolCallEvent] = []
    usage_event: UsageEvent | None = None
    for event in events:
        match event:
            case TextDelta(text=text):
                text_chunks.append(text)
            case ToolCallEvent():
                tool_calls.append(event)
            case UsageEvent():
                usage_event = event
            case StopEvent():
                pass
            case _:
                assert_never(event)
    assert usage_event is not None
    return "".join(text_chunks), tool_calls, usage_event


def _proposal_summary(assistant_text: str, tool_calls: Sequence[ToolCallEvent]) -> str:
    """Render what one turn proposed to do, for `self_critique_fn` to
    judge: the assistant's own text when it wrote any, otherwise a
    one-line rendering of each tool call it requested instead."""
    if assistant_text:
        return assistant_text
    return "; ".join(f"{call.name}({call.arguments_json})" for call in tool_calls)


def _total_tokens(meter: CostMeter) -> int:
    """Sum of every recorded turn's input-plus-output tokens so far.

    Cached tokens are already counted within a turn's `input_tokens`,
    so adding them again separately would double-count them.
    """
    return sum(turn.input_tokens + turn.output_tokens for turn in meter.turns)


def _has_passing_verification(deps: LoopDeps) -> bool:
    """True iff at least one `VerificationReport` has been recorded for
    this task and the most recent one passed -- an earlier failing
    report does not linger once a later call supersedes it."""
    return bool(deps.verification_reports) and deps.verification_reports[-1].passed


def _dispatch_tool_call(
    event: ToolCallEvent, *, deps: LoopDeps, turn_id: int, task_id: str
) -> ToolResult:
    """Run one requested tool call through the shared dispatcher, turning
    a denied approval into a framed refusal result rather than letting
    it escape -- a model's own tool call is never fatal to the loop,
    whether it names an unregistered tool, sends malformed arguments,
    or is turned down at the approval gate.
    """
    try:
        return dispatch(
            event,
            repo_root=deps.repo_root,
            approval=deps.approval,
            undo=deps.undo,
            turn_id=turn_id,
            task_id=task_id,
            report_sink=deps.verification_reports,
        )
    except ApprovalDenied as exc:
        return ToolResult(
            tool_call_id=event.id,
            content=frame_untrusted(str(exc), source="tool_stderr", origin=event.name),
        )


async def _drain_think(
    deps: LoopDeps, history: Sequence[Message], entry: ModelEntry
) -> list[StreamEvent]:
    """Stream one full turn from `deps.client`, offering it the full
    tool set, and collect every event it yields.

    The leading messages sent ahead of `history` come from
    `build_stable_prefix`, folding `deps.kestrel_md` in when present, so
    every turn of one task sends the exact same prefix a cache-capable
    backend can reuse -- `mark_cache_breakpoints` then annotates it for
    `entry`, a no-op for every backend that needs no explicit marker.

    Routed through `complete_with_retry` so a transient rate-limit or
    server failure is retried with backoff before it ever reaches this
    loop as an exception.
    """
    prefix = mark_cache_breakpoints(
        build_stable_prefix(_SYSTEM_PROMPT, deps.kestrel_md), entry
    )
    messages: list[Message] = [*prefix, *history]
    return [
        event
        async for event in complete_with_retry(
            deps.client, messages, all_schemas(), deps.model_id, "high", stream=True
        )
    ]


async def run_task(
    task_description: str,
    deps: LoopDeps,
    task_id: str,
    *,
    clock_fn: Callable[[], float] = time.monotonic,
) -> LoopResult:
    """Drive `task_description` through the loop until it completes or a
    termination predicate trips.

    `deps.model_id` must already be a valid entry in `deps.registry` --
    resolving and validating a starting model is the caller's job, not
    this loop's, the same contract `kestrel.repl.run_repl` places on its
    own starting model id.

    Every iteration repeats: a pre-check against `deps.limits` (turns
    spent so far, wall-clock elapsed since the first turn) that ends the
    task with the matching `TerminationReason` before spending another
    model call, never after; a model call offering the full tool set;
    a `deps.self_critique_fn` pass over what was proposed, which, on
    `False`, drops the proposal, records a synthetic explanation of the
    skip in its place, and moves on to another turn instead of acting on
    it; dispatching every requested tool call in order through the
    shared tool registry, turning a denied approval into a framed
    refusal rather than raising; and finally folding the turn's usage
    into `deps.meter` and checking the token cap immediately, so a turn
    that crosses it never triggers one more, avoidable model call. A
    turn that requested no tools at all is the task's natural
    completion -- unless `deps.require_verification` is `True` and the
    most recent entry in `deps.verification_reports` has not passed
    (or none exists yet), in which case that turn instead folds a
    nudge to call `verify` into history and the loop keeps going,
    exactly as the self-critique-skip path already does.

    A `ContextOverflowError` raised while streaming a turn ends the task
    with `TerminationReason.CONTEXT_OVERFLOW` rather than propagating --
    there is no compaction here to recover the window and retry. A
    `KeyboardInterrupt` raised at any point during a turn -- streaming
    the model call or running a tool -- ends the task with
    `TerminationReason.USER_STOP`, keeping whatever turns and cost had
    already accumulated, rather than escaping this call.
    """
    entry = deps.registry.get(deps.model_id)
    start = clock_fn()
    history: list[Message] = [{"role": "user", "content": task_description}]
    turns_used = 0

    def finish(reason: TerminationReason) -> LoopResult:
        """Build the `LoopResult` for `reason`, snapshotting the
        turn count, priced total, and history as they stand right now."""
        return LoopResult(
            reason=reason,
            turns_used=turns_used,
            total_usd=deps.meter.session_usd,
            history=tuple(history),
        )

    try:
        while True:
            if turns_used >= deps.limits.max_turns:
                return finish(TerminationReason.TURN_CAP)
            if clock_fn() - start >= deps.limits.max_wall_clock_s:
                return finish(TerminationReason.WALL_CLOCK_CAP)
            if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                return finish(TerminationReason.TOKEN_CAP)

            turns_used += 1
            try:
                events = await _drain_think(deps, history, entry)
            except ContextOverflowError:
                return finish(TerminationReason.CONTEXT_OVERFLOW)

            assistant_text, tool_calls, usage_event = _split_events(events)
            proposal = _proposal_summary(assistant_text, tool_calls)

            if not deps.self_critique_fn(proposal, list(history)):
                if tool_calls:
                    history.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                            "tool_calls": tool_calls,
                        }
                    )
                    for call in tool_calls:
                        history.append(
                            {
                                "role": "tool",
                                "content": _SELF_CRITIQUE_SKIP_CONTENT,
                                "tool_call_id": call.id,
                            }
                        )
                else:
                    history.append({"role": "assistant", "content": assistant_text})
                    history.append(
                        {"role": "tool", "content": _SELF_CRITIQUE_SKIP_CONTENT}
                    )
                deps.meter.record(usage_event, entry)
                if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                    return finish(TerminationReason.TOKEN_CAP)
                continue

            if tool_calls:
                history.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                history.append({"role": "assistant", "content": assistant_text})

            if not tool_calls:
                if deps.require_verification and not _has_passing_verification(deps):
                    history.append(
                        {"role": "tool", "content": _VERIFICATION_REQUIRED_CONTENT}
                    )
                    deps.meter.record(usage_event, entry)
                    if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                        return finish(TerminationReason.TOKEN_CAP)
                    continue
                deps.meter.record(usage_event, entry)
                return finish(TerminationReason.TASK_COMPLETE)

            for call in tool_calls:
                result = await asyncio.to_thread(
                    _dispatch_tool_call,
                    call,
                    deps=deps,
                    turn_id=turns_used,
                    task_id=task_id,
                )
                history.append(
                    {"role": "tool", "content": result.content, "tool_call_id": call.id}
                )

            deps.meter.record(usage_event, entry)
            if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                return finish(TerminationReason.TOKEN_CAP)
    except KeyboardInterrupt:
        return finish(TerminationReason.USER_STOP)
