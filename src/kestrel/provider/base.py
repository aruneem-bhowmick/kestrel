"""The vendor-neutral client protocol every provider backend implements.

Nothing in this module names a vendor or imports a provider SDK -- it is
the contract that ``kestrel.provider.litellm_client`` (and any future
adapter) is written against, so call sites never need to know which
backend is actually serving a given model id.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from kestrel.provider.events import StreamEvent, ToolCallEvent


class Message(TypedDict):
    """One turn of conversation history passed to a model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: NotRequired[list[ToolCallEvent]]
    tool_call_id: NotRequired[str]


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool definition offered to the model, in JSON Schema form.

    Unused until tools are wired into the agent loop, but part of the
    provider contract now so every backend adapter accepts the same
    shape from the start.

    Attributes:
        name: The tool's callable name, as the model will reference it.
        description: Human/model-readable summary of what the tool does.
        parameters: JSON Schema describing the tool's argument object.
    """

    name: str
    description: str
    parameters: dict[str, Any]


Effort = Literal["high", "max"]


class ProviderClient(Protocol):
    """Single async client interface implemented by every backend adapter.

    ``complete`` streams a normalized event sequence for one turn: zero
    or more ``TextDelta``/``ToolCallEvent`` events, then exactly one
    ``UsageEvent``, then exactly one ``StopEvent`` as the final event
    (see :func:`kestrel.provider.events.validate_stream_order`). On
    failure, the iterator raises a
    :class:`~kestrel.provider.errors.ProviderError` subclass instead of
    emitting a ``StopEvent`` -- callers must not treat a mid-stream
    exception as if generation had merely stopped.
    """

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion for ``model_id`` given ``messages`` and optional ``tools``.

        When ``stream=True``, the adapter yields events incrementally (e.g., multiple
        ``TextDelta`` events as chunks arrive) as they are produced by the provider.

        When ``stream=False``, the adapter must buffer the response from the provider
        and emit a single consolidated sequence of events (e.g., a single ``TextDelta``
        containing the full accumulated generated text, or fully populated
        ``ToolCallEvent``s, followed by ``UsageEvent`` and ``StopEvent``) rather
        than yielding incremental updates. Even with ``stream=False``, the return
        value remains an ``AsyncIterator[StreamEvent]`` and must satisfy the normal
        stream ordering grammar.

        ``max_tokens``, when given, caps the completion tokens the backend is
        asked to generate -- callers that need a call's spend bounded by
        construction (e.g. a reachability probe) set it explicitly; ``None``
        leaves the backend's own default in effect, which is every existing
        call site's behavior.
        """
        ...
