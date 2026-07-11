"""Retry a provider call with bounded exponential backoff and full jitter.

A stream that fails before yielding anything is safe to replay: nothing
has reached the caller yet, so a retried call cannot duplicate output.
A stream that fails after yielding at least one event is a different
story -- replaying it would risk re-emitting text or a tool call the
caller already consumed. :func:`complete_with_retry` wraps any
:class:`~kestrel.provider.base.ProviderClient` with exactly that
distinction: only a pre-first-event failure is retried, and only when
its error type is one this codebase considers transient.

Of the four typed provider errors, only :class:`~kestrel.provider.errors.RateLimitError`
and :class:`~kestrel.provider.errors.ServerError` are retried here.
:class:`~kestrel.provider.errors.AuthError` means the credential itself
is bad, and :class:`~kestrel.provider.errors.ContextOverflowError` means
the request itself is too large -- neither improves on a later attempt
with the same inputs.

Backoff follows full jitter: the Nth retry (0-indexed) sleeps
``min(policy.max_delay_s, policy.base_delay_s * 2**N) * jitter_fn()``
seconds, chosen uniformly between zero and the exponential cap rather
than always sleeping the cap itself -- this is what keeps many clients
retrying the same failure from all waking up in lockstep. When the
raised :class:`RateLimitError` names an explicit ``retry_after_s``
(e.g. from a backend's ``Retry-After`` header), that value is honored
verbatim instead of the computed delay, since the backend has told the
caller exactly how long to wait.

Multi-model failover (retrying a different model or backend after a
failure) is out of scope for this module: it wraps one
``ProviderClient`` call with one model id per invocation, and has no
notion of a fallback chain.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

from kestrel.provider.base import Effort, Message, ProviderClient, ToolSchema
from kestrel.provider.errors import ProviderError, RateLimitError, ServerError
from kestrel.provider.events import StreamEvent

_RETRIABLE: tuple[type[ProviderError], ...] = (RateLimitError, ServerError)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounds and timing for :func:`complete_with_retry`'s backoff schedule.

    Attributes:
        max_attempts: Total calls allowed, including the first -- one
            initial attempt plus up to ``max_attempts - 1`` retries.
        base_delay_s: The base of the exponential backoff, in seconds,
            before jitter is applied.
        max_delay_s: The ceiling every computed delay is clamped to,
            before jitter is applied.
    """

    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


def _delay_for_attempt(
    exc: ProviderError,
    attempt: int,
    policy: RetryPolicy,
    jitter_fn: Callable[[], float],
) -> float:
    """Compute how long to sleep before the retry following ``exc``.

    ``attempt`` is the 0-indexed count of failures already observed (the
    first failure is attempt ``0``). A ``RateLimitError`` carrying a
    non-``None`` ``retry_after_s`` overrides the computed delay entirely;
    otherwise the delay is the full-jitter formula documented on this
    module.
    """
    if isinstance(exc, RateLimitError) and exc.retry_after_s is not None:
        return exc.retry_after_s
    capped = min(policy.max_delay_s, policy.base_delay_s * (2.0**attempt))
    return capped * jitter_fn()


async def complete_with_retry(
    client: ProviderClient,
    messages: Sequence[Message],
    tools: Sequence[ToolSchema] | None,
    model_id: str,
    effort: Effort,
    *,
    policy: RetryPolicy = RetryPolicy(),
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter_fn: Callable[[], float] = random.random,
    stream: bool = True,
    max_tokens: int | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream ``client.complete(...)``, retrying transient failures per ``policy``.

    A failure raised before this generator has yielded any event is
    retried when it is a :class:`~kestrel.provider.errors.RateLimitError`
    or :class:`~kestrel.provider.errors.ServerError` and attempts remain
    under ``policy.max_attempts``; any other error type, or the same
    error types once ``max_attempts`` is exhausted, re-raises immediately.
    A failure raised after at least one event has already been yielded to
    the caller is never retried, regardless of its type -- see the module
    docstring for why. ``sleep_fn`` and ``jitter_fn`` are injection points
    so a caller (in practice, a test) can replace real sleeping and
    randomness with deterministic stand-ins.
    """
    for attempt in range(policy.max_attempts):
        emitted = False
        try:
            async for event in client.complete(
                messages,
                tools,
                model_id,
                effort,
                stream=stream,
                max_tokens=max_tokens,
            ):
                emitted = True
                yield event
            return
        except _RETRIABLE as exc:
            if emitted or attempt == policy.max_attempts - 1:
                raise
            await sleep_fn(_delay_for_attempt(exc, attempt, policy, jitter_fn))
