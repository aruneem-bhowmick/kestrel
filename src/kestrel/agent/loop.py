"""Drives one task through a tool-calling loop until it finishes or a
termination predicate trips.

Every iteration of the loop repeats the same six phases, in order:

1. Pre-check & compaction. Stop the task outright once it has used up
   its turn cap, its wall-clock budget, or its cumulative token cap.
   Otherwise, when the most recently recorded turn's own input tokens
   sit at or above 70% of the active model's context window, fold
   everything but the last few messages of `history` into one
   model-generated summary (`kestrel.agent.compaction.compact_history`)
   before spending another real model call. That summarization call is
   itself priced, journaled, and checked against the configured budget
   exactly like any other turn -- sharing its own journal record's
   `turn_id` with the real turn that follows it -- and can itself end
   the task (a hard budget halt or the token cap) before that turn's
   own model call is ever made.
2. Think. Send the stable leading prefix (the system prompt, plus the
   target repo's own project-memory file when one exists) followed by
   `history` to the active model -- possibly a cheaper entry than the
   task started on, once a budget threshold has been crossed -- through
   the shared retrying client wrapper.
3. Self-critique. Sanity-check what the model proposed via an
   injectable predicate before acting on it; a declined proposal is
   recorded as a skipped turn with a synthetic explanation in place of
   whatever it wanted to do, and the loop tries again instead.
4. Act / tool dispatch. Pass every requested tool call through the
   shared tool registry, in the order the model made them, threading in
   the task's approval and undo managers, its running verification
   reports, and enough identifying context (turn and task id) for a
   tool to journal its own effects.
5. Tool execution. Run each dispatched call and fold its result back
   into history as a tool-role message, keyed to the call it answers.
6. Post-process. Price the turn, journal it to the task's session when
   one is configured, classify the task's running spend against its
   configured budget (halting outright past the hard threshold,
   degrading to a cheaper model past the soft one), check the
   cumulative token cap, and -- only for a turn that requested no tool
   calls at all -- decide whether that is really the task's natural
   completion, or whether it should be nudged to keep going instead.

`run_task` is the entry point for a brand new task; everything it needs
-- the provider client, the model registry, the approval and undo
managers, the cost meter, and the per-task limits -- arrives through
one `LoopDeps` bundle, so nothing here reaches for global state or
constructs its own collaborators.

Whether a turn with no requested tool calls actually ends the task is
itself configurable: `LoopDeps.require_verification` (opt-in,
default-`False`) withholds `TASK_COMPLETE` from that turn until the
most recent `verify` tool call recorded in `deps.verification_reports`
passed, nudging the model to keep going instead of letting it declare
victory on its own say-so. Every caller that leaves the field at its
default sees exactly the prior, ungated behavior.

A task's own turns are durably journaled as they complete: when
`LoopDeps.session` is set, every real (non-declined) turn -- and every
compaction event folded into history along the way -- is appended to
it as a `TurnRecord` -- its own message deltas, priced cost, and
whichever `VerificationReport` is most current -- so a task interrupted
by a crash or a hard budget halt can be reconstructed via
`kestrel.managers.session.load_session` and continued with
`resume_task` rather than losing everything since the process's last
clean exit. `run_task` and `resume_task` both drive the same shared
`_drive` loop, seeded differently: `run_task` starts fresh history from
a `task_description`; `resume_task` seeds history, the cost meter, and
the last verification report from a prior session's own journal, and
continues the turn-cap counter from where that session left off rather
than from zero. Wall-clock budget is never inherited across a resume --
`clock_fn` is always sampled fresh at the start of whichever call is
driving the loop.

Every turn's own spend is also checked against `LoopDeps.budget`, when
set: a `BudgetManager.check` call folds this turn's own priced total on
top of the day/month baselines the caller computed once, before the
task started (`spent_day_usd`/`spent_month_usd`), and crossing the soft
threshold switches every following turn to whichever registry entry is
tagged `"cheap"`, logging a warning either way, while crossing the hard
threshold ends the task immediately with a `BUDGET_HALT`
`TerminationReason`. By the time that happens, whatever turn tripped it
has already been journaled to `deps.session` (when one is set), so
`resume_task` can pick the task back up once an operator raises the
cap. A task degrades at most once, to at most one cheaper entry, and
never un-degrades mid-task even if later spend looks fine again;
`LoopDeps.budget=None` (the default) skips every check above, leaving
every existing caller's behavior unchanged.

Context-window pressure is recovered from, not just detected: past the
70% threshold described in phase 1 above, `kestrel.agent.compaction`
folds the older portion of `history` into one carry-forward summary
that preserves the task's own goal, any next-step or TODO language the
conversation already contains, and the most recent verification
result, so a long-running task does not have to choose between running
out of room and forgetting what it was doing. This reduces the
*frequency* of a context overflow, not the possibility of one: a single
message that alone exceeds the window -- one enormous tool result, say
-- still ends the task `CONTEXT_OVERFLOW`, compaction or not, exactly
as it always did.

A task's own progress is observable as it happens, not only once it
ends: `LoopDeps.observer` (an injectable `kestrel.agent.observer.
LoopObserver`, defaulting to an all-no-op `NullLoopObserver`) is called
at seven fixed points as `_drive` runs -- a turn starting, each
streamed text chunk, a tool call starting and finishing, a fresh
`VerificationReport` landing, a turn's own priced cost settling, and
the task's own termination. Every call happens synchronously, inline,
on the same coroutine driving the task, so an observer that blocks or
raises stalls (or crashes) the task itself; no observer method's
return value is ever read, so nothing an observer does can change what
the loop decides next. Leaving `observer` unset is a no-op by
construction, so every caller written before this hook existed keeps
its exact prior behavior.

Which effort level a turn is sent at, and which tools it may call, are
themselves per-task fields rather than fixed constants: `LoopDeps.effort`
(default `"high"`) is threaded straight through to the retrying client
wrapper on every turn, and `LoopDeps.available_tools` (default `None`,
meaning every registered tool) both filters the schema list offered to
the model and gates a requested tool call against that same set --
calling anything outside it is refused with a framed result folded into
history as an ordinary tool-role message, never raised or otherwise
fatal to the loop. Every caller that leaves both fields at their
defaults sees identical behavior to before either field existed; this
module makes no decision about which effort or tool set a given task
should actually run with, only carries whichever values it is given.

`resume_task` can also fold one new instruction into a prior task's
history before continuing it: its `inject_message` parameter, when set,
is appended as a fresh user-role message right after the loaded history
and before the resumed drive begins, letting a caller continue a task
that already reached `TASK_COMPLETE` rather than only one a turn cap,
token cap, wall-clock cap, or crash halted mid-run. Leaving it unset
(the default) preserves the exact resume behavior every existing caller
already relies on.

Deliberately out of scope for this module, each a real gap rather than
an oversight:

- No plan/fast mode switching -- `LoopDeps.effort` and `available_tools`
  are plain per-task fields this module only carries; nothing here
  decides which value a given task should actually run with.
- No real self-critique model call -- `LoopDeps.self_critique_fn`
  defaults to always approving; the injection point exists so a real
  cheap-model check can be plugged in later without changing this
  module's control flow.
- No cascading, multi-tier budget degradation, and no un-degrading back
  to a costlier model once spend looks fine again -- a task degrades to
  at most one cheaper registry entry, at most once, for the rest of its
  own run.
- No configurable compaction threshold or keep-last-N window -- both
  are fixed constants today, not something a repo's own `kestrel.toml`
  can adjust yet.
- No artifact persistence beyond the session journal itself -- `LoopResult`
  is a plain, in-memory value, never written to disk on its own.
- No subagents -- `run_task` drives exactly one flat loop and never
  spawns a nested one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

from kestrel.agent.compaction import compact_history, should_compact
from kestrel.agent.observer import NULL_OBSERVER, LoopObserver
from kestrel.cost.meter import CostMeter, TurnCost
from kestrel.kestrel_md import KestrelMd
from kestrel.managers.approval import ApprovalDenied, ApprovalManager
from kestrel.managers.budget import BudgetManager, BudgetStatus
from kestrel.managers.session import SessionManager, TurnRecord, load_session
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ProviderClient
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
from kestrel.tools.registry import ToolResult, dispatch, schemas_for
from kestrel.tools.verify import VerificationReport

logger = logging.getLogger("kestrel.agent")

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
    BUDGET_HALT = "budget_halt"


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
        session: Journals every real turn of this task as a
            `TurnRecord`, when set -- `None` (the default) means no
            journal is kept and the task cannot later be resumed via
            `resume_task`. Every existing caller that leaves this unset
            sees identical behavior to before this field existed.
        budget: Classifies this task's spend against configured USD
            caps every turn, when set -- `None` (the default) skips the
            check entirely, leaving every existing caller's behavior
            unchanged.
        spent_day_usd: Spend already recorded, across every other task,
            for the current day -- a fixed baseline the caller computes
            once, up front (e.g. via
            `kestrel.managers.session.aggregate_historical_spend`), and
            never re-reads mid-task; this task's own growing
            `meter.session_usd` is added on top of it for every check.
        spent_month_usd: The same fixed baseline as `spent_day_usd`, but
            summed over the current month rather than the current day.
        observer: Called at seven fixed points as this task runs, purely
            for external visibility -- see `kestrel.agent.observer.
            LoopObserver` for the full contract. Defaults to
            `NullLoopObserver`, so every caller that leaves this unset
            sees identical behavior to before this field existed.
        effort: The reasoning-depth level every turn in this task is
            sent at. Defaults to `"high"`, identical to every caller
            before this field existed.
        available_tools: The tool names this task's turns may call, or
            `None` for every registered tool. Defaults to `None`,
            identical to every caller before this field existed.
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
    session: SessionManager | None = None
    budget: BudgetManager | None = None
    spent_day_usd: Decimal = Decimal(0)
    spent_month_usd: Decimal = Decimal(0)
    observer: LoopObserver = field(default_factory=lambda: NULL_OBSERVER)
    effort: Effort = "high"
    available_tools: frozenset[str] | None = None


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


def _find_cheap_entry(registry: Registry, *, exclude: str) -> str | None:
    """The first (sorted by id, for determinism) registry entry tagged
    `"cheap"` other than `exclude` (the currently active model --
    degrading to the same model would be a no-op); `None` when the
    registry has no such entry, in which case the caller keeps running
    on the current model rather than treating a missing target as an
    error."""
    for model_id in registry.ids():
        if model_id != exclude and "cheap" in registry.get(model_id).tags:
            return model_id
    return None


def _apply_budget_check(
    deps: LoopDeps,
    *,
    active_model_id: str,
    degraded: bool,
    spent_session: Decimal,
) -> tuple[str, bool, bool]:
    """Classify this turn's spend against `deps.budget` and decide
    whether the task should keep running as-is, degrade to a cheaper
    model, or halt outright.

    Returns `(active_model_id, degraded, should_halt)`: `active_model_id`
    is only changed the first time a SOFT threshold is crossed and a
    `"cheap"`-tagged entry other than the current one exists; `degraded`
    flips to `True` the first time a SOFT threshold is crossed at all,
    whether or not a cheap entry was actually available to switch to,
    since a task degrades at most once regardless; `should_halt` tells
    the caller to end the task with `TerminationReason.BUDGET_HALT`
    right away. A task with `deps.budget=None` never classifies
    anything and returns its own inputs back unchanged with
    `should_halt=False`.
    """
    if deps.budget is None:
        return active_model_id, degraded, False

    event = deps.budget.check(
        spent_session=spent_session,
        spent_day=deps.spent_day_usd + spent_session,
        spent_month=deps.spent_month_usd + spent_session,
    )
    if event.status is BudgetStatus.HARD:
        return active_model_id, degraded, True
    if event.status is BudgetStatus.SOFT and not degraded:
        cheap_id = _find_cheap_entry(deps.registry, exclude=active_model_id)
        if cheap_id is not None:
            logger.warning(
                "budget soft cap reached (%s); degrading to %r",
                event.tripped_cap,
                cheap_id,
            )
            return cheap_id, True, False
        logger.warning(
            "budget soft cap reached (%s); no 'cheap'-tagged registry "
            "entry to degrade to, continuing on %r",
            event.tripped_cap,
            active_model_id,
        )
        return active_model_id, True, False
    return active_model_id, degraded, False


def _record_session_turn(
    deps: LoopDeps,
    *,
    turn_id: int,
    task_id: str,
    history: Sequence[Message],
    messages_before: int,
    turn_cost: TurnCost,
    active_model_id: str,
    degraded: bool,
) -> None:
    """Append this turn's own `TurnRecord` to `deps.session`'s journal,
    when one is configured for this task -- a harmless no-op otherwise.

    `message_deltas` is exactly the slice of `history` appended since
    `messages_before` (`history`'s length immediately before this turn's
    own messages were folded in), and `verification` is the most recent
    report recorded so far, if any -- not necessarily one this turn
    itself produced.
    """
    if deps.session is None:
        return
    deps.session.record_turn(
        TurnRecord(
            turn_id=turn_id,
            task_id=task_id,
            timestamp=time.time(),
            message_deltas=tuple(history[messages_before:]),
            turn_cost=turn_cost,
            verification=(
                deps.verification_reports[-1] if deps.verification_reports else None
            ),
            active_model_id=active_model_id,
            degraded=degraded,
        )
    )


def _finish_turn(
    deps: LoopDeps,
    *,
    turn_id: int,
    task_id: str,
    history: Sequence[Message],
    messages_before: int,
    turn_cost: TurnCost,
    active_model_id: str,
    degraded: bool,
) -> None:
    """Journal this turn via `_record_session_turn`, then report it to
    `deps.observer.on_turn_finished` -- the pair every real turn and
    compaction fold in `_drive` performs together, in this order, as it
    wraps up."""
    _record_session_turn(
        deps,
        turn_id=turn_id,
        task_id=task_id,
        history=history,
        messages_before=messages_before,
        turn_cost=turn_cost,
        active_model_id=active_model_id,
        degraded=degraded,
    )
    deps.observer.on_turn_finished(
        turn_id=turn_id, turn_cost=turn_cost, active_model_id=active_model_id
    )


def _dispatch_tool_call(
    event: ToolCallEvent, *, deps: LoopDeps, turn_id: int, task_id: str
) -> ToolResult:
    """Run one requested tool call through the shared dispatcher, turning
    a denied approval into a framed refusal result rather than letting
    it escape -- a model's own tool call is never fatal to the loop,
    whether it names an unregistered tool, sends malformed arguments,
    or is turned down at the approval gate.

    A name outside `deps.available_tools` (when that allowlist is set)
    is refused the same way, before `dispatch` is ever called -- no
    tool executor runs, so a restricted task cannot trigger that tool's
    undo, approval, or verification side effects by having a call
    refused rather than skipped by the model itself.
    """
    if deps.available_tools is not None and event.name not in deps.available_tools:
        return ToolResult(
            tool_call_id=event.id,
            content=frame_untrusted(
                f"{event.name!r} is not available in this mode; only "
                f"{sorted(deps.available_tools)} may be called.",
                source="tool_stderr",
                origin=event.name,
            ),
        )
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
    deps: LoopDeps, history: Sequence[Message], entry: ModelEntry, model_id: str
) -> list[StreamEvent]:
    """Stream one full turn from `deps.client`, offering it
    `deps.available_tools`'s schemas (every registered tool's, when left
    at its default `None`) at `deps.effort`, and collect every event it
    yields.

    `model_id` is the turn's own actually-active model -- `deps.model_id`
    itself is never read here, since a budget-triggered degrade changes
    which model a later turn is sent to without ever mutating
    `deps.model_id`; `entry` is `model_id`'s own registry entry, already
    re-resolved by the caller, so the two always name the same model.

    The leading messages sent ahead of `history` come from
    `build_stable_prefix`, folding `deps.kestrel_md` in when present, so
    every turn of one task sends the exact same prefix a cache-capable
    backend can reuse -- `mark_cache_breakpoints` then annotates it for
    `entry`, a no-op for every backend that needs no explicit marker.

    Routed through `complete_with_retry` so a transient rate-limit or
    server failure is retried with backoff before it ever reaches this
    loop as an exception.

    Every `TextDelta` yielded along the way is also echoed to
    `deps.observer.on_text_delta`, in arrival order, before it is
    appended to the returned list -- the loop's one streaming
    observation point.
    """
    prefix = mark_cache_breakpoints(
        build_stable_prefix(_SYSTEM_PROMPT, deps.kestrel_md), entry
    )
    messages: list[Message] = [*prefix, *history]
    events: list[StreamEvent] = []
    async for event in complete_with_retry(
        deps.client,
        messages,
        schemas_for(deps.available_tools),
        model_id,
        deps.effort,
        stream=True,
    ):
        if isinstance(event, TextDelta):
            deps.observer.on_text_delta(event.text)
        events.append(event)
    return events


async def _drive(
    history: list[Message],
    deps: LoopDeps,
    task_id: str,
    clock_fn: Callable[[], float],
    *,
    turns_used_start: int,
    active_model_id_start: str | None = None,
    degraded_start: bool = False,
    unjournaled_seed_len: int = 0,
) -> LoopResult:
    """Drive `history` through the loop until it completes or a
    termination predicate trips -- the shared engine behind both
    `run_task` (fresh history, `turns_used_start=0`) and `resume_task`
    (history and turn count seeded from a prior session).

    Every iteration repeats: a pre-check against `deps.limits` (turns
    spent so far, wall-clock elapsed since this call's own start, and
    the cumulative token cap) that ends the task with the matching
    `TerminationReason` before spending another model call, never
    after; when the most recently recorded turn's own input tokens sit
    at or above 70% of the active model's context window, a
    summarize-and-fold compaction call (`kestrel.agent.compaction.
    compact_history`, sent at `deps.effort` exactly like every other
    turn) that replaces `history` with a shorter, equivalent one before
    this iteration's own model call is ever made -- itself priced,
    journaled, and budget-checked exactly like any other turn, and
    capable of ending the task on its own (a hard budget halt or the
    token cap) before that model call happens; a model call offering
    `deps.available_tools`'s schemas (every registered tool's, when
    left at its default `None`); a `deps.self_critique_fn` pass over
    what was proposed, which, on `False`, drops the proposal,
    records a synthetic explanation of the skip in its place, and moves
    on to another turn instead of acting on it; dispatching every
    requested tool call in order through the shared tool registry,
    turning a denied approval into a framed refusal rather than
    raising; and finally folding the turn's usage into `deps.meter` and
    checking the token cap immediately, so a turn that crosses it never
    triggers one more, avoidable model call. A turn that requested no
    tools at all is the task's natural completion -- unless
    `deps.require_verification` is `True` and the most recent entry in
    `deps.verification_reports` has not passed (or none exists yet), in
    which case that turn instead folds a nudge to call `verify` into
    history and the loop keeps going, exactly as the self-critique-skip
    path already does.

    Every real (non-declined) turn -- and every compaction fold along
    the way -- is also journaled to `deps.session`, when one is
    configured, as a `TurnRecord` covering exactly the messages that
    turn itself appended (a compaction's own record instead covers the
    whole, just-folded `history`, since folding replaces history rather
    than appending to it, and shares its `turn_id` with the real turn
    that immediately follows it). `turns_used_start == 0` (a fresh
    `run_task` call) means nothing in `history` has been journaled yet,
    so the first turn's own record also captures whatever seed messages
    `history` already held when this call began; `turns_used_start > 0`
    (a `resume_task` call) means every message in `history` except its
    last `unjournaled_seed_len` entries was already durably recorded by
    a prior call, so the first turn's own record captures only that
    trailing, not-yet-recorded slice on top of whatever it appends
    itself -- `unjournaled_seed_len` defaults to `0`, meaning the entire
    loaded history was already recorded, which is `resume_task`'s own
    behavior whenever its `inject_message` parameter is left unset. A
    compaction fold that happens before that first resumed turn ever
    runs already covers the whole post-fold history in its own record
    unconditionally, so `unjournaled_seed_len` is reset to `0` the
    moment one occurs -- otherwise the first resumed turn would count
    the tail of that fold's own already-recorded history as new all
    over again.

    A `ContextOverflowError` raised while streaming a turn -- or while
    streaming the compaction call itself -- ends the task with
    `TerminationReason.CONTEXT_OVERFLOW` rather than propagating:
    compaction lowers how often this happens, but a single message that
    alone exceeds the window still overflows regardless. A
    `KeyboardInterrupt` raised at any point during a turn -- streaming
    the model call or running a tool -- ends the task with
    `TerminationReason.USER_STOP`, keeping whatever turns and cost had
    already accumulated, rather than escaping this call.

    Every turn's own priced cost is also checked against `deps.budget`,
    once it has been recorded: crossing the hard threshold ends the task
    with `TerminationReason.BUDGET_HALT` right there, before another
    model call is ever made, and crossing the soft threshold switches
    `active_model_id` -- the model every subsequent `_drain_think` call
    actually targets, as opposed to the fixed `deps.model_id` the task
    started on -- to a `"cheap"`-tagged registry entry, at most once per
    task.
    """
    active_model_id = (
        active_model_id_start
        if turns_used_start > 0 and active_model_id_start is not None
        else deps.model_id
    )
    degraded = degraded_start if turns_used_start > 0 else False
    entry = deps.registry.get(active_model_id)
    start = clock_fn()
    turns_used = turns_used_start

    def finish(reason: TerminationReason) -> LoopResult:
        """Build the `LoopResult` for `reason`, snapshotting the turn
        count, priced total, and history as they stand right now, and
        report it to `deps.observer.on_termination` -- the loop's
        single choke point for ending a task, so every termination
        path fires that hook exactly once."""
        result = LoopResult(
            reason=reason,
            turns_used=turns_used,
            total_usd=deps.meter.session_usd,
            history=tuple(history),
        )
        deps.observer.on_termination(result)
        return result

    try:
        while True:
            if turns_used >= deps.limits.max_turns:
                return finish(TerminationReason.TURN_CAP)
            if clock_fn() - start >= deps.limits.max_wall_clock_s:
                return finish(TerminationReason.WALL_CLOCK_CAP)
            if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                return finish(TerminationReason.TOKEN_CAP)

            if deps.meter.turns and should_compact(
                deps.meter.turns[-1].input_tokens, entry.context_window
            ):
                try:
                    compacted_history, compaction_usage = await compact_history(
                        deps.client,
                        active_model_id,
                        history,
                        last_verification=(
                            deps.verification_reports[-1]
                            if deps.verification_reports
                            else None
                        ),
                        effort=deps.effort,
                    )
                except ContextOverflowError:
                    return finish(TerminationReason.CONTEXT_OVERFLOW)
                history = compacted_history
                # A fold's own record (below) covers the entire post-fold
                # history unconditionally, including any trailing slice
                # unjournaled_seed_len was tracking (e.g. resume_task's
                # own inject_message) -- so that slice is no longer
                # unjournaled once this fold is recorded, and the turn
                # that follows must not subtract it a second time.
                unjournaled_seed_len = 0
                compaction_cost = deps.meter.record(compaction_usage, entry)
                # The compaction record shares its turn_id with the real turn
                # that follows it (turns_used hasn't been incremented for
                # that turn yet) and captures the whole post-fold history as
                # its own delta, since a fold replaces history rather than
                # appending to it.
                _finish_turn(
                    deps,
                    turn_id=turns_used + 1,
                    task_id=task_id,
                    history=history,
                    messages_before=0,
                    turn_cost=compaction_cost,
                    active_model_id=active_model_id,
                    degraded=degraded,
                )
                active_model_id, degraded, should_halt = _apply_budget_check(
                    deps,
                    active_model_id=active_model_id,
                    degraded=degraded,
                    spent_session=deps.meter.session_usd,
                )
                entry = deps.registry.get(active_model_id)
                if should_halt:
                    return finish(TerminationReason.BUDGET_HALT)
                if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                    return finish(TerminationReason.TOKEN_CAP)

            turns_used += 1
            deps.observer.on_turn_started(
                turn_id=turns_used, active_model_id=active_model_id
            )
            # `turns_used == 1` is this call's very first turn with no
            # earlier iteration to have already journaled whatever seed
            # messages `history` started with -- true only for a fresh
            # `run_task` call (turns_used_start == 0), since a resumed
            # call's first turn has turns_used_start >= 1. That first
            # resumed turn instead subtracts unjournaled_seed_len, the
            # trailing slice of the loaded history a caller appended
            # after the prior session's own journal already covered it
            # (resume_task's own inject_message, when set), so that
            # slice is captured by this turn's own record rather than
            # treated as already journaled. Every later turn's own
            # boundary is simply `history`'s length as this iteration
            # begins, since the turn before it already journaled
            # everything up to that point.
            if turns_used == 1:
                turn_start_len = 0
            elif turns_used == turns_used_start + 1:
                turn_start_len = len(history) - unjournaled_seed_len
            else:
                turn_start_len = len(history)
            try:
                events = await _drain_think(deps, history, entry, active_model_id)
            except ContextOverflowError:
                return finish(TerminationReason.CONTEXT_OVERFLOW)

            assistant_text, tool_calls, usage_event = _split_events(events)
            proposal = _proposal_summary(assistant_text, tool_calls)

            if not await asyncio.to_thread(
                deps.self_critique_fn, proposal, list(history)
            ):
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
                turn_cost = deps.meter.record(usage_event, entry)
                _finish_turn(
                    deps,
                    turn_id=turns_used,
                    task_id=task_id,
                    history=history,
                    messages_before=turn_start_len,
                    turn_cost=turn_cost,
                    active_model_id=active_model_id,
                    degraded=degraded,
                )
                active_model_id, degraded, should_halt = _apply_budget_check(
                    deps,
                    active_model_id=active_model_id,
                    degraded=degraded,
                    spent_session=deps.meter.session_usd,
                )
                entry = deps.registry.get(active_model_id)
                if should_halt:
                    return finish(TerminationReason.BUDGET_HALT)
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
                    turn_cost = deps.meter.record(usage_event, entry)
                    _finish_turn(
                        deps,
                        turn_id=turns_used,
                        task_id=task_id,
                        history=history,
                        messages_before=turn_start_len,
                        turn_cost=turn_cost,
                        active_model_id=active_model_id,
                        degraded=degraded,
                    )
                    active_model_id, degraded, should_halt = _apply_budget_check(
                        deps,
                        active_model_id=active_model_id,
                        degraded=degraded,
                        spent_session=deps.meter.session_usd,
                    )
                    entry = deps.registry.get(active_model_id)
                    if should_halt:
                        return finish(TerminationReason.BUDGET_HALT)
                    if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                        return finish(TerminationReason.TOKEN_CAP)
                    continue
                turn_cost = deps.meter.record(usage_event, entry)
                _finish_turn(
                    deps,
                    turn_id=turns_used,
                    task_id=task_id,
                    history=history,
                    messages_before=turn_start_len,
                    turn_cost=turn_cost,
                    active_model_id=active_model_id,
                    degraded=degraded,
                )
                active_model_id, degraded, should_halt = _apply_budget_check(
                    deps,
                    active_model_id=active_model_id,
                    degraded=degraded,
                    spent_session=deps.meter.session_usd,
                )
                entry = deps.registry.get(active_model_id)
                if should_halt:
                    return finish(TerminationReason.BUDGET_HALT)
                return finish(TerminationReason.TASK_COMPLETE)

            for call in tool_calls:
                deps.observer.on_tool_call_started(call)
                verification_count_before = len(deps.verification_reports)
                result = await asyncio.to_thread(
                    _dispatch_tool_call,
                    call,
                    deps=deps,
                    turn_id=turns_used,
                    task_id=task_id,
                )
                deps.observer.on_tool_call_finished(call, result)
                if len(deps.verification_reports) > verification_count_before:
                    deps.observer.on_verification(deps.verification_reports[-1])
                history.append(
                    {"role": "tool", "content": result.content, "tool_call_id": call.id}
                )

            turn_cost = deps.meter.record(usage_event, entry)
            _finish_turn(
                deps,
                turn_id=turns_used,
                task_id=task_id,
                history=history,
                messages_before=turn_start_len,
                turn_cost=turn_cost,
                active_model_id=active_model_id,
                degraded=degraded,
            )
            active_model_id, degraded, should_halt = _apply_budget_check(
                deps,
                active_model_id=active_model_id,
                degraded=degraded,
                spent_session=deps.meter.session_usd,
            )
            entry = deps.registry.get(active_model_id)
            if should_halt:
                return finish(TerminationReason.BUDGET_HALT)
            if _total_tokens(deps.meter) >= deps.limits.max_total_tokens:
                return finish(TerminationReason.TOKEN_CAP)
    except KeyboardInterrupt:
        return finish(TerminationReason.USER_STOP)


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

    Starts a brand new conversation seeded from `task_description` and
    drives it via the shared `_drive` engine (see its own docstring for
    the loop's full turn-by-turn behavior); `resume_task` is the sibling
    entry point that continues a prior session instead of starting one.
    """
    history: list[Message] = [{"role": "user", "content": task_description}]
    return await _drive(history, deps, task_id, clock_fn, turns_used_start=0)


async def resume_task(
    task_id: str,
    deps: LoopDeps,
    *,
    inject_message: str | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
) -> LoopResult:
    """Reconstruct a prior task's state via
    `kestrel.managers.session.load_session(deps.repo_root, task_id)` and
    continue driving it through the same `_drive` engine `run_task`
    uses, seeded from that state rather than from a fresh
    `task_description`.

    Seeds: `history` from the loaded session's own `history`; `deps.meter`
    re-populated from its `turns` (via `CostMeter`'s `initial_turns`
    parameter); `deps.verification_reports` pre-populated with
    `[last_verification]` when the session recorded one, else left
    empty; the loop's own turn-cap counter starts at the session's
    `turns_used`, not zero.

    Wall-clock budget is NOT inherited: `clock_fn()` is sampled fresh at
    the start of `_drive`'s own call, and `deps.limits.max_wall_clock_s`
    applies to this resumed call's own elapsed time only -- a task
    halted by a hard budget cap yesterday and resumed today must not
    immediately trip `WALL_CLOCK_CAP` from time that passed while no
    process was even running.

    `inject_message`, when set, is appended as one new user-role message
    right after the loaded history and before this call resumes driving
    it -- `None` (the default) preserves every existing caller's exact
    behavior. Unlike every other field this function reconstructs from
    the session journal, `inject_message` is never itself journaled as
    part of the *prior* task's own history; it becomes journaled the
    ordinary way, as this call's own first new turn's input, once
    `_drive` records that turn. This also means a task that already
    reached `TASK_COMPLETE` can be resumed, not only one a cap or crash
    halted mid-run -- `_drive`'s own control flow places no precondition
    on the loaded state's prior termination reason.

    Raises:
        FileNotFoundError: propagated from `load_session` unchanged --
            no journal exists for `task_id` under `deps.repo_root`.
    """
    state = load_session(deps.repo_root, task_id)
    deps.meter = CostMeter(initial_turns=state.turns)
    deps.verification_reports = (
        [state.last_verification] if state.last_verification is not None else []
    )
    history: list[Message] = list(state.history)
    if inject_message is not None:
        history.append({"role": "user", "content": inject_message})
    return await _drive(
        history,
        deps,
        task_id,
        clock_fn,
        turns_used_start=state.turns_used,
        active_model_id_start=state.active_model_id,
        degraded_start=state.degraded,
        unjournaled_seed_len=1 if inject_message is not None else 0,
    )
