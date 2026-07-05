"""Tests for the normalized provider event types and typed error taxonomy."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import assert_never

import pytest

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
    ToolCallEvent,
    UsageEvent,
)

pytestmark = [pytest.mark.p004, pytest.mark.unit, pytest.mark.sanity]


@pytest.mark.parametrize(
    "event",
    [
        TextDelta("hello"),
        ToolCallEvent(id="1", name="read_file", arguments_json="{}"),
        UsageEvent(input_tokens=1, output_tokens=1, cached_tokens=0),
        StopEvent(reason="end_turn"),
    ],
)
def test_event_types_are_frozen(event: StreamEvent) -> None:
    """Given any stream event type, when a field is reassigned, then it
    raises -- every event in the stream grammar is immutable once built."""
    field_name = dataclasses.fields(event)[0].name

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(event, field_name, "mutated")


def test_stream_event_match_is_exhaustive() -> None:
    """Given the StreamEvent union, when every member is matched with a
    fallback `assert_never` arm, then the match classifies each concrete
    instance correctly at runtime, and mypy's static exhaustiveness check
    would fail to compile if a fifth member were ever added here without
    a matching case."""

    def classify(event: StreamEvent) -> str:
        match event:
            case TextDelta():
                return "text_delta"
            case ToolCallEvent():
                return "tool_call"
            case UsageEvent():
                return "usage"
            case StopEvent():
                return "stop"
            case _:
                assert_never(event)

    assert classify(TextDelta("hi")) == "text_delta"
    assert classify(ToolCallEvent(id="1", name="x", arguments_json="{}")) == "tool_call"
    assert classify(UsageEvent(1, 2, 0)) == "usage"
    assert classify(StopEvent("max_tokens")) == "stop"


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: AuthError(
            "missing credentials", model_id="glm-5.2", backend="openrouter"
        ),
        lambda: ContextOverflowError(
            "too many tokens", model_id="glm-5.2", backend="zai"
        ),
        lambda: RateLimitError("slow down", model_id="glm-5.2", backend="openrouter"),
        lambda: ServerError("boom", model_id="glm-5.2", backend="zai"),
    ],
)
def test_error_constructors_carry_model_and_backend_into_str(
    make_error: Callable[[], ProviderError],
) -> None:
    """Given any ProviderError subclass, when constructed and stringified,
    then the rendered message names both the model id and the backend --
    a bare print(exc) at a call site is self-identifying."""
    error = make_error()

    assert error.model_id == "glm-5.2"
    assert error.model_id in str(error)
    assert error.backend in str(error)


def test_rate_limit_error_retry_after_defaults_to_none() -> None:
    """Given a RateLimitError built without a retry_after_s, when
    inspected, then retry_after_s is None -- the backend did not supply a
    Retry-After hint."""
    error = RateLimitError("slow down", model_id="glm-5.2", backend="openrouter")

    assert error.retry_after_s is None


def test_rate_limit_error_retry_after_accepts_a_value() -> None:
    """Given a RateLimitError built with an explicit retry_after_s, when
    inspected, then that value is preserved unchanged."""
    error = RateLimitError(
        "slow down", model_id="glm-5.2", backend="openrouter", retry_after_s=12.5
    )

    assert error.retry_after_s == 12.5


def test_server_error_status_defaults_to_none() -> None:
    """Given a ServerError built without a status, when inspected, then
    status is None -- e.g. a connection timeout with no HTTP response."""
    error = ServerError("timed out", model_id="glm-5.2", backend="zai")

    assert error.status is None


def test_server_error_status_accepts_a_value() -> None:
    """Given a ServerError built with an explicit status, when inspected,
    then that value is preserved unchanged."""
    error = ServerError("boom", model_id="glm-5.2", backend="zai", status=500)

    assert error.status == 500
