"""Normalized provider event stream: ``(TextDelta | ToolCallEvent)* UsageEvent StopEvent``.

Every backend adapter (LiteLLM-based or otherwise) must translate its own
wire format into this vendor-neutral grammar before anything else in the
codebase sees it. Types here are frozen, slotted dataclasses rather than
pydantic models: a streaming response can emit thousands of ``TextDelta``
events per turn, and that hot path should not pay pydantic's validation
and construction overhead for values that are already known-good (they
come from an adapter's own normalization code, not from untrusted input).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, assert_never


@dataclass(frozen=True, slots=True)
class TextDelta:
    """An incremental chunk of assistant-visible text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A completed tool invocation requested by the model.

    Dormant in this phase of the project (no tools are offered yet), but
    part of the grammar now so every backend adapter is written against
    the final shape from the start.

    Attributes:
        id: The backend's identifier for this tool call, echoed back when
            the tool result is returned to the model.
        name: The tool's name, as declared in the request's tool schema.
        arguments_json: The raw JSON string of arguments. Parsing (and
            validating against the tool's schema) is the caller's job,
            not the event's -- a malformed payload should surface where
            it is used, not swallowed silently here.
    """

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Token accounting for one completed turn.

    Attributes:
        input_tokens: Prompt tokens billed for this turn.
        output_tokens: Completion tokens billed for this turn.
        cached_tokens: The subset of ``input_tokens`` served from a
            prompt cache, billed at the cache rate. ``0`` when the
            backend does not report cache hits.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass(frozen=True, slots=True)
class StopEvent:
    """The final event of every stream, naming why generation ended."""

    reason: StopReason


StreamEvent = TextDelta | ToolCallEvent | UsageEvent | StopEvent


def validate_stream_order(events: Sequence[StreamEvent]) -> bool:
    """Check that ``events`` matches the required stream grammar.

    The grammar is: zero or more ``TextDelta``/``ToolCallEvent`` events,
    followed by exactly one ``UsageEvent``, followed by exactly one
    ``StopEvent`` as the final event. Returns ``True`` when ``events``
    conforms and ``False`` otherwise (e.g. a usage event after the stop
    event, two stop events, a delta after usage, or a stream that never
    reaches a stop event at all).

    Every ``ProviderClient`` implementation's stream must satisfy this
    before any event reaches the cost meter or the REPL; it also doubles
    as the invariant checked by tests that build event sequences with
    hypothesis.
    """
    usage_seen = False
    last_index = len(events) - 1
    for index, event in enumerate(events):
        match event:
            case TextDelta() | ToolCallEvent():
                if usage_seen:
                    return False
            case UsageEvent():
                if usage_seen:
                    return False
                usage_seen = True
            case StopEvent():
                return usage_seen and index == last_index
            case _:
                assert_never(event)
    return False
