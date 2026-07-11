"""Unit tests for provider-call retry: backoff timing, jitter, and the
mid-stream-never-retries rule.

Every case here drives ``complete_with_retry`` against a small scripted
fake ``ProviderClient`` -- a fixed list of per-call "attempts", each
either raising a given exception immediately or yielding a given event
sequence (optionally followed by a raised exception, for the mid-stream
case) -- rather than a real backend, so timing and error injection stay
fully deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import pytest

from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from kestrel.provider.events import (
    StopEvent,
    StreamEvent,
    TextDelta,
    UsageEvent,
    validate_stream_order,
)
from kestrel.provider.retry import RetryPolicy, _delay_for_attempt, complete_with_retry

pytestmark = [pytest.mark.p020, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_BACKEND = "openrouter"


@dataclass
class _Attempt:
    """One scripted call's outcome: events to yield, then an optional raise."""

    events: tuple[StreamEvent, ...] = ()
    raises: ProviderError | None = None


@dataclass
class _ScriptedClient:
    """A ``ProviderClient`` that replays one ``_Attempt`` per call, in order.

    Calling ``complete`` more times than there are scripted attempts is a
    test-authoring error and raises ``IndexError`` -- every test here
    scripts exactly as many attempts as it expects calls, so running out
    of script means the retry logic called the client more times than
    intended.
    """

    attempts: Sequence[_Attempt]
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
        """Replay the next scripted attempt, ignoring every argument but the count."""
        attempt = self.attempts[self.call_count]
        self.call_count += 1
        for event in attempt.events:
            yield event
        if attempt.raises is not None:
            raise attempt.raises


def _sleep_recorder() -> tuple[Callable[[float], Awaitable[None]], list[float]]:
    """Build a fake ``sleep_fn`` that never actually sleeps, recording each delay."""
    delays: list[float] = []

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    return _sleep, delays


def _rate_limit(retry_after_s: float | None = None) -> RateLimitError:
    """Build a RateLimitError naming the shared test model/backend."""
    return RateLimitError(
        "rate limited",
        model_id=_MODEL_ID,
        backend=_BACKEND,
        retry_after_s=retry_after_s,
    )


def _server_error() -> ServerError:
    """Build a ServerError naming the shared test model/backend."""
    return ServerError("server failed", model_id=_MODEL_ID, backend=_BACKEND)


_SUCCESS_EVENTS = (
    TextDelta(text="hello"),
    UsageEvent(input_tokens=10, output_tokens=5, cached_tokens=0),
    StopEvent(reason="end_turn"),
)


async def _collect(gen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    """Drain an event stream to a plain list."""
    return [event async for event in gen]


@pytest.mark.sanity
async def test_retries_once_on_rate_limit_then_succeeds() -> None:
    """Given a RateLimitError on the first call and success on the second,
    when the stream is drained, then exactly two calls are made, exactly
    one delay is slept, and the final successful events are yielded."""
    client = _ScriptedClient(
        attempts=[_Attempt(raises=_rate_limit()), _Attempt(events=_SUCCESS_EVENTS)]
    )
    sleep_fn, delays = _sleep_recorder()

    events = await _collect(
        complete_with_retry(
            client,
            messages=[],
            tools=None,
            model_id=_MODEL_ID,
            effort="high",
            sleep_fn=sleep_fn,
            jitter_fn=lambda: 0.5,
        )
    )

    assert client.call_count == 2
    assert len(delays) == 1
    assert events == list(_SUCCESS_EVENTS)


async def test_exhausting_max_attempts_reraises_last_exception() -> None:
    """Given every attempt raises ServerError, when the stream is drained,
    then the last exception re-raises and exactly max_attempts calls are
    made -- no attempt beyond the configured bound."""
    policy = RetryPolicy(max_attempts=4)
    client = _ScriptedClient(
        attempts=[_Attempt(raises=_server_error()) for _ in range(4)]
    )
    sleep_fn, _ = _sleep_recorder()

    with pytest.raises(ServerError):
        await _collect(
            complete_with_retry(
                client,
                messages=[],
                tools=None,
                model_id=_MODEL_ID,
                effort="high",
                policy=policy,
                sleep_fn=sleep_fn,
                jitter_fn=lambda: 0.5,
            )
        )

    assert client.call_count == 4


@pytest.mark.sanity
async def test_auth_error_never_retries() -> None:
    """Given the first call raises AuthError, when the stream is drained,
    then it raises immediately and exactly one call is made -- a bad
    credential cannot be fixed by retrying the same request."""
    client = _ScriptedClient(
        attempts=[
            _Attempt(raises=AuthError("bad key", model_id=_MODEL_ID, backend=_BACKEND))
        ]
    )
    sleep_fn, delays = _sleep_recorder()

    with pytest.raises(AuthError):
        await _collect(
            complete_with_retry(
                client,
                messages=[],
                tools=None,
                model_id=_MODEL_ID,
                effort="high",
                sleep_fn=sleep_fn,
                jitter_fn=lambda: 0.5,
            )
        )

    assert client.call_count == 1
    assert delays == []


@pytest.mark.sanity
async def test_context_overflow_error_never_retries() -> None:
    """Given the first call raises ContextOverflowError, when the stream is
    drained, then it raises immediately and exactly one call is made -- a
    too-large request cannot be fixed by retrying the same request."""
    client = _ScriptedClient(
        attempts=[
            _Attempt(
                raises=ContextOverflowError(
                    "too big", model_id=_MODEL_ID, backend=_BACKEND
                )
            )
        ]
    )
    sleep_fn, delays = _sleep_recorder()

    with pytest.raises(ContextOverflowError):
        await _collect(
            complete_with_retry(
                client,
                messages=[],
                tools=None,
                model_id=_MODEL_ID,
                effort="high",
                sleep_fn=sleep_fn,
                jitter_fn=lambda: 0.5,
            )
        )

    assert client.call_count == 1
    assert delays == []


async def test_mid_stream_failure_never_retries() -> None:
    """Given one TextDelta is yielded before ServerError is raised, when
    the stream is drained, then the exception propagates rather than
    triggering a retry, and exactly one call is made -- replaying a
    stream that already reached the caller would risk duplicating output
    it already consumed."""
    client = _ScriptedClient(
        attempts=[
            _Attempt(events=(TextDelta(text="partial"),), raises=_server_error()),
            _Attempt(events=_SUCCESS_EVENTS),
        ]
    )
    sleep_fn, delays = _sleep_recorder()

    received: list[StreamEvent] = []
    gen = complete_with_retry(
        client,
        messages=[],
        tools=None,
        model_id=_MODEL_ID,
        effort="high",
        sleep_fn=sleep_fn,
        jitter_fn=lambda: 0.5,
    )
    with pytest.raises(ServerError):
        async for event in gen:
            received.append(event)

    assert received == [TextDelta(text="partial")]
    assert client.call_count == 1
    assert delays == []


async def test_retry_after_s_overrides_computed_delay() -> None:
    """Given the raised RateLimitError names an explicit retry_after_s,
    when the retry sleeps, then it sleeps exactly that duration rather
    than the computed exponential-backoff delay, even with a jitter_fn
    that would otherwise produce a very different value."""
    client = _ScriptedClient(
        attempts=[
            _Attempt(raises=_rate_limit(retry_after_s=12.5)),
            _Attempt(events=_SUCCESS_EVENTS),
        ]
    )
    sleep_fn, delays = _sleep_recorder()

    await _collect(
        complete_with_retry(
            client,
            messages=[],
            tools=None,
            model_id=_MODEL_ID,
            effort="high",
            sleep_fn=sleep_fn,
            jitter_fn=lambda: 999.0,
        )
    )

    assert delays == [12.5]


async def test_backoff_without_retry_after_s_follows_full_jitter_formula() -> None:
    """Given three consecutive ServerErrors then success, and a fixed
    jitter_fn, when each retry sleeps, then the Nth delay (0-indexed)
    equals min(max_delay_s, base_delay_s * 2**N) * jitter_fn() -- growing
    with each attempt and staying strictly positive for a nonzero
    jitter_fn."""
    policy = RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=30.0)
    client = _ScriptedClient(
        attempts=[
            _Attempt(raises=_server_error()),
            _Attempt(raises=_server_error()),
            _Attempt(raises=_server_error()),
            _Attempt(events=_SUCCESS_EVENTS),
        ]
    )
    sleep_fn, delays = _sleep_recorder()

    await _collect(
        complete_with_retry(
            client,
            messages=[],
            tools=None,
            model_id=_MODEL_ID,
            effort="high",
            policy=policy,
            sleep_fn=sleep_fn,
            jitter_fn=lambda: 0.5,
        )
    )

    assert delays == [0.5, 1.0, 2.0]
    assert all(delay > 0 for delay in delays)


def test_delay_for_attempt_clamps_to_max_delay_s() -> None:
    """Given an attempt count high enough that the exponential term would
    exceed max_delay_s, when the delay is computed, then it is clamped to
    max_delay_s (before jitter) rather than growing unbounded."""
    policy = RetryPolicy(max_attempts=10, base_delay_s=1.0, max_delay_s=5.0)

    delay = _delay_for_attempt(
        _server_error(), attempt=10, policy=policy, jitter_fn=lambda: 1.0
    )

    assert delay == 5.0


def test_delay_for_attempt_ignores_retry_after_s_for_non_rate_limit_errors() -> None:
    """Given a ServerError (which has no retry_after_s concept at all),
    when the delay is computed, then it always follows the exponential
    formula -- the retry_after_s override applies only to RateLimitError."""
    policy = RetryPolicy(base_delay_s=2.0, max_delay_s=30.0)

    delay = _delay_for_attempt(
        _server_error(), attempt=1, policy=policy, jitter_fn=lambda: 1.0
    )

    assert delay == 4.0


async def test_successful_final_sequence_satisfies_stream_order() -> None:
    """Given a retried call that eventually succeeds, when the yielded
    events are validated, then they still satisfy the normalized stream
    grammar -- retrying never disturbs the underlying event ordering
    contract."""
    client = _ScriptedClient(
        attempts=[_Attempt(raises=_rate_limit()), _Attempt(events=_SUCCESS_EVENTS)]
    )
    sleep_fn, _ = _sleep_recorder()

    events = await _collect(
        complete_with_retry(
            client,
            messages=[],
            tools=None,
            model_id=_MODEL_ID,
            effort="high",
            sleep_fn=sleep_fn,
            jitter_fn=lambda: 0.5,
        )
    )

    assert validate_stream_order(events)


async def test_stream_and_max_tokens_are_forwarded_to_the_wrapped_client() -> None:
    """Given stream=False and an explicit max_tokens, when the call is
    made, then both are forwarded to the wrapped client's complete() call
    unchanged -- the retry wrapper is transparent to every other
    parameter it does not itself interpret."""
    seen: dict[str, object] = {}

    class _RecordingClient:
        async def complete(
            self,
            messages: Sequence[Message],
            tools: Sequence[ToolSchema] | None,
            model_id: str,
            effort: Effort,
            stream: bool = True,
            max_tokens: int | None = None,
        ) -> AsyncIterator[StreamEvent]:
            seen["stream"] = stream
            seen["max_tokens"] = max_tokens
            for event in _SUCCESS_EVENTS:
                yield event

    await _collect(
        complete_with_retry(
            _RecordingClient(),
            messages=[],
            tools=None,
            model_id=_MODEL_ID,
            effort="high",
            stream=False,
            max_tokens=256,
        )
    )

    assert seen == {"stream": False, "max_tokens": 256}
