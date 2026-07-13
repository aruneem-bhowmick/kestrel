"""Recovers from context-window pressure by folding older conversation
history into one model-generated summary before it ever reaches the
provider as an oversized request.

A registry entry's own ``context_window`` is the hard ceiling on how
much a model call may send plus receive; once one turn's own billed
prompt size gets close to it, every following turn in the same task
risks being rejected outright by the backend before ever producing an
answer. :func:`should_compact` reads the most recently billed turn's
own input-token count -- the best available signal of current context
pressure, since Kestrel does not run its own tokenizer independently
of what the backend already reported -- against the active model's
context window, and :func:`compact_history` performs the fold itself:
everything but the most recent handful of messages is replaced by one
carry-forward summary, explicitly instructed to preserve the task's
own goal, any plan or TODO language the conversation already contains,
and the most recent verification outcome, rather than inventing new
steps of its own.

The summarization call this module makes is priced and accounted for
exactly like any other model call: it returns its own ``UsageEvent``
for the caller to fold into whichever cost meter, budget classifier,
or session journal it maintains, rather than tracking any of that
state itself. This keeps the module a pure, injectable strategy --
it knows nothing about the agent loop's own dependency bundle, and a
caller supplies a real provider client and gets back a replacement
history and a priced usage event, nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Final

from kestrel.provider.base import Message, ProviderClient
from kestrel.provider.events import TextDelta, UsageEvent
from kestrel.provider.retry import complete_with_retry
from kestrel.tools.verify import VerificationReport

_COMPACTION_THRESHOLD: Final[Decimal] = Decimal("0.70")
_DEFAULT_KEEP_LAST_N: Final[int] = 4

_COMPACTION_SYSTEM_PROMPT: Final[str] = (
    "Summarize the conversation so far into a concise carry-forward "
    "note for continuing this task. Preserve: the task's own goal, "
    "any explicit next-step or TODO statements already made, and "
    "anything about what has and has not been tried yet. Do not "
    "invent new steps."
)


def should_compact(
    last_input_tokens: int,
    context_window: int,
    *,
    threshold: Decimal = _COMPACTION_THRESHOLD,
) -> bool:
    """True when ``last_input_tokens / context_window >= threshold``.

    ``context_window <= 0`` never compacts -- defensive only, since
    registry validation already guarantees every real ``ModelEntry``
    carries a strictly positive ``context_window``, so this branch is
    unreachable through any real registry entry; it exists only so a
    caller that passes a raw denominator directly, rather than one
    already validated by the registry, cannot divide by zero or by a
    negative number.
    """
    if context_window <= 0:
        return False
    return Decimal(last_input_tokens) / Decimal(context_window) >= threshold


def _failing_command_names(report: VerificationReport) -> list[str]:
    """Every command name in ``report`` that did not cleanly pass, in
    the order ``report.commands`` lists them -- a nonzero exit code or
    a timeout both count as failing."""
    return [
        command.name
        for command in report.commands
        if command.timed_out or command.exit_code != 0
    ]


def _render_verification_note(last_verification: VerificationReport | None) -> str:
    """Render the fixed block appended after the model's own summary
    text, naming ``last_verification``'s pass/fail and, when it
    failed, the verbatim names of every command that failed. Returns
    an empty string when ``last_verification`` is ``None``, so no
    block -- and no literal ``"None"`` -- appears in that case.
    """
    if last_verification is None:
        return ""
    if last_verification.passed:
        return "\n\nLast verification: PASSED"
    failing = ", ".join(_failing_command_names(last_verification))
    return f"\n\nLast verification: FAILED (failing: {failing})"


async def _drain_summary_text(
    client: ProviderClient, messages: Sequence[Message], model_id: str
) -> tuple[str, UsageEvent]:
    """Stream one non-tool-calling completion from ``client`` and fold
    it into its rendered text and closing usage event.

    No tool schema is offered on this call, so a well-behaved backend
    never emits a ``ToolCallEvent`` here; one is simply not collected
    if it arrives, since nothing about a compaction summary should
    ever be treated as a tool request.
    """
    text_chunks: list[str] = []
    usage_event: UsageEvent | None = None
    async for event in complete_with_retry(
        client, messages, None, model_id, "high", stream=True
    ):
        if isinstance(event, TextDelta):
            text_chunks.append(event.text)
        elif isinstance(event, UsageEvent):
            usage_event = event
    assert usage_event is not None
    return "".join(text_chunks), usage_event


async def compact_history(
    client: ProviderClient,
    model_id: str,
    history: Sequence[Message],
    *,
    last_verification: VerificationReport | None,
    keep_last_n: int = _DEFAULT_KEEP_LAST_N,
) -> tuple[list[Message], UsageEvent]:
    """Fold everything but the most recent ``keep_last_n`` messages of
    ``history`` into one model-generated summary.

    When ``len(history) <= keep_last_n``, there is nothing old enough
    to fold: returns ``(list(history), UsageEvent(0, 0, 0))``
    unchanged, and -- critically -- makes no call to ``client`` at
    all, so a caller can never mistake "nothing to compact" for a
    free, zero-cost model call that actually happened.

    Otherwise, asks ``client`` (at ``effort="high"``, the only level
    this call ever uses) to summarize ``history[:-keep_last_n]`` --
    the older tail -- via one non-tool-calling completion routed
    through ``complete_with_retry``; the most recent ``keep_last_n``
    messages are kept verbatim rather than folded, since the model may
    still be actively reasoning about its own latest tool results.
    Returns ``([summary_message, *history[-keep_last_n:]],
    usage_event)``, where ``summary_message`` is a ``"system"``-role
    message holding the model's rendered summary followed by a fixed
    block naming ``last_verification``'s pass/fail -- and, when it
    failed, the exact names of every command that failed -- or nothing
    at all when ``last_verification`` is ``None``.

    This function never touches a cost meter, a session journal, or a
    budget classifier itself -- ``usage_event`` is returned for the
    caller to fold into whichever of those it maintains, exactly like
    any other turn's own usage, so a compaction call is accounted for
    by the same machinery every real model call already goes through.
    """
    if len(history) <= keep_last_n:
        return list(history), UsageEvent(
            input_tokens=0, output_tokens=0, cached_tokens=0
        )

    older_tail = history[:-keep_last_n]
    messages: list[Message] = [
        {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT},
        *older_tail,
    ]
    summary_text, usage_event = await _drain_summary_text(client, messages, model_id)
    summary_content = summary_text + _render_verification_note(last_verification)
    summary_message: Message = {"role": "system", "content": summary_content}
    return [summary_message, *history[-keep_last_n:]], usage_event
