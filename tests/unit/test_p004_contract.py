"""Contract test: the normalized provider event stream's ordering invariant.

The grammar `(TextDelta | ToolCallEvent)* UsageEvent StopEvent` is the
contract every backend adapter's stream must satisfy -- this is the
property-based test-of-record for the checker (`validate_stream_order`)
that enforces it.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kestrel.provider.events import (
    StopEvent,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
    validate_stream_order,
)

pytestmark = [pytest.mark.p004, pytest.mark.api]

_STOP_REASONS: tuple[StopReason, ...] = ("end_turn", "tool_use", "max_tokens", "error")

_body_events = st.one_of(
    st.builds(TextDelta, text=st.text(max_size=20)),
    st.builds(
        ToolCallEvent,
        id=st.text(min_size=1, max_size=8),
        name=st.text(min_size=1, max_size=8),
        arguments_json=st.just("{}"),
    ),
)


@st.composite
def _valid_streams(draw: st.DrawFn) -> list[StreamEvent]:
    """Build a stream that matches the grammar exactly: zero or more body
    events, one UsageEvent, then one closing StopEvent."""
    body: list[StreamEvent] = draw(st.lists(_body_events, max_size=5))
    usage = UsageEvent(
        input_tokens=draw(st.integers(min_value=0, max_value=1000)),
        output_tokens=draw(st.integers(min_value=0, max_value=1000)),
        cached_tokens=draw(st.integers(min_value=0, max_value=1000)),
    )
    stop = StopEvent(reason=draw(st.sampled_from(_STOP_REASONS)))
    return [*body, usage, stop]


@given(_valid_streams())
def test_valid_streams_are_accepted(events: list[StreamEvent]) -> None:
    """Given any stream matching the grammar, when validated, then it is
    accepted."""
    assert validate_stream_order(events) is True


def test_empty_stream_is_rejected() -> None:
    """Given an empty stream, when validated, then it is rejected -- there
    is neither a usage event nor a closing stop event."""
    assert validate_stream_order([]) is False


@given(_valid_streams())
def test_usage_after_stop_is_rejected(events: list[StreamEvent]) -> None:
    """Given a valid stream with an extra UsageEvent appended after its
    closing StopEvent, when validated, then it is rejected."""
    corrupted = [*events, UsageEvent(1, 1, 0)]

    assert validate_stream_order(corrupted) is False


@given(_valid_streams())
def test_double_stop_is_rejected(events: list[StreamEvent]) -> None:
    """Given a valid stream with a second StopEvent appended, when
    validated, then it is rejected -- the stop event must be the final
    event, not merely present somewhere in the stream."""
    corrupted = [*events, StopEvent(reason="end_turn")]

    assert validate_stream_order(corrupted) is False


@given(_valid_streams())
def test_double_usage_is_rejected(events: list[StreamEvent]) -> None:
    """Given a valid stream with its UsageEvent duplicated immediately
    before the closing StopEvent, when validated, then it is rejected --
    exactly one usage event may appear in the stream."""
    *body, usage, stop = events
    corrupted = [*body, usage, usage, stop]

    assert validate_stream_order(corrupted) is False


@given(_valid_streams())
def test_delta_after_usage_is_rejected(events: list[StreamEvent]) -> None:
    """Given a valid stream with its closing StopEvent replaced by a
    TextDelta, when validated, then it is rejected -- nothing may follow
    the usage event except the closing stop event."""
    *body, usage, _stop = events
    corrupted = [*body, usage, TextDelta("late")]

    assert validate_stream_order(corrupted) is False


@given(_valid_streams())
def test_missing_stop_is_rejected(events: list[StreamEvent]) -> None:
    """Given a valid stream with its closing StopEvent removed, when
    validated, then it is rejected."""
    *rest, _stop = events

    assert validate_stream_order(rest) is False


def test_unrecognized_event_type_raises_via_assert_never() -> None:
    """Given a value that is not one of the four StreamEvent members, when
    validate_stream_order encounters it, then the match statement's
    exhaustiveness-enforcing fallback raises rather than silently
    accepting or misclassifying it -- the runtime half of the
    mypy-enforced exhaustiveness check."""

    class _NotAStreamEvent:
        pass

    with pytest.raises(AssertionError):
        validate_stream_order([_NotAStreamEvent()])  # type: ignore[list-item]
