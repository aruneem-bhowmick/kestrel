"""The LiteLLM-backed ``ProviderClient`` implementation.

``LiteLLMClient`` is the one place in the codebase that turns a registry
entry into an actual network call: it resolves a model id, reads its
credential from the environment, translates its ``backend`` into a
concrete LiteLLM ``model=``/``api_base=`` pair, and normalizes whatever
LiteLLM hands back -- streamed chunks or a single buffered response --
into the event grammar defined in :mod:`kestrel.provider.events`. Every
other module in the codebase reaches a model exclusively through
:class:`~kestrel.provider.base.ProviderClient`, so this file is the only
place a vendor name may legitimately appear.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, assert_never

import litellm
from litellm.exceptions import APIError as LiteLLMAPIError
from litellm.exceptions import AuthenticationError as LiteLLMAuthenticationError
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from litellm.exceptions import ServiceUnavailableError as LiteLLMServiceUnavailableError
from litellm.exceptions import Timeout as LiteLLMTimeout

from kestrel.provider.base import Effort, Message, ProviderClient, ToolSchema
from kestrel.provider.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from kestrel.provider.events import (
    StopEvent,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from kestrel.registry.model import ModelEntry, Registry

logger = logging.getLogger("kestrel.provider")

# LiteLLM prints ANSI-colored diagnostic hints (e.g. a "Provider List" link)
# straight to stdout of its own accord -- notably when a streamed chunk's
# own self-reported ``model`` field doesn't match a provider it recognizes,
# which is unrelated to whether the call itself is succeeding. A plain
# terminal REPL must not have a vendored HTTP library's internal debug
# output leak into the user's terminal, so this is disabled globally the
# moment this adapter is imported.
litellm.suppress_debug_info = True

_OPENROUTER_BASE_URL_ENV = "KESTREL_OPENROUTER_BASE_URL"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Applied to every litellm.acompletion call so a hung connection surfaces as
# a typed ServerError (via litellm's own Timeout exception) instead of
# blocking the caller indefinitely on whatever litellm's own default is.
_REQUEST_TIMEOUT_S = 120.0

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}

# Every litellm exception type this adapter knows how to map onto Kestrel's
# typed error taxonomy (see ``_map_error``). Anything else escapes uncaught,
# since it represents a programming error in this adapter rather than a
# provider-side failure.
_MAPPED_LITELLM_ERRORS: tuple[type[Exception], ...] = (
    LiteLLMAuthenticationError,
    LiteLLMRateLimitError,
    LiteLLMContextWindowExceededError,
    LiteLLMAPIError,
    LiteLLMServiceUnavailableError,
    LiteLLMTimeout,
)


def _litellm_params(entry: ModelEntry) -> dict[str, Any]:
    """Translate a registry entry's ``backend`` into LiteLLM call kwargs.

    This is the only function in the codebase that maps a backend name to
    a concrete routing decision; every other call site reaches a model
    through the vendor-neutral ``ProviderClient`` interface instead.

    The OpenRouter branch reads the optional ``KESTREL_OPENROUTER_BASE_URL``
    environment variable in place of the real OpenRouter endpoint. This is
    a deliberate, documented test seam: it defaults to the real OpenRouter
    base URL (so it is inert in production) and lets integration/system
    tests redirect every OpenRouter call to a local mock server without
    touching this function's interface or monkeypatching its internals.

    The zai branch has no equivalent env-var seam: ``entry.endpoint`` is
    used verbatim, since the registry already carries the base URL to call
    (and, for tests, may simply be pointed at a mock server directly). This
    keeps backend selection entirely registry-driven, with no vendor ever
    named outside this function.
    """
    match entry.backend:
        case "openrouter":
            return {
                "model": f"openrouter/{entry.provider_model}",
                "api_base": os.environ.get(
                    _OPENROUTER_BASE_URL_ENV, _DEFAULT_OPENROUTER_BASE_URL
                ),
            }
        case "zai":
            return {
                "model": f"openai/{entry.provider_model}",
                "api_base": entry.endpoint,
            }
        case "anthropic" | "ollama":
            raise ServerError(
                f"backend '{entry.backend}' is not implemented yet",
                model_id=entry.id,
                backend=entry.backend,
            )
        case _:
            assert_never(entry.backend)


def _require_api_key(entry: ModelEntry) -> str:
    """Read the entry's credential from its named environment variable.

    Checked before any network call so a missing credential fails fast
    with a typed error naming the environment variable -- never a value --
    instead of surfacing as a confusing HTTP-level error from the backend.
    """
    env_var = entry.api_key_env
    if not env_var:
        raise AuthError(
            f"model '{entry.id}' has no api_key_env configured",
            model_id=entry.id,
            backend=entry.backend,
        )
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise AuthError(
            f"environment variable '{env_var}' is not set",
            model_id=entry.id,
            backend=entry.backend,
        )
    return api_key


def _tool_schemas_to_litellm(
    tools: Sequence[ToolSchema] | None,
) -> list[dict[str, Any]] | None:
    """Render Kestrel's vendor-neutral tool schemas as OpenAI-style tool dicts.

    Returns ``None`` when no tools were offered, so callers can pass the
    result straight through to ``litellm.acompletion`` without a separate
    branch. No caller passes tools yet, but the conversion is implemented
    now so the adapter never needs revisiting once one does.
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _map_finish_reason(finish_reason: str | None) -> StopReason:
    """Map a backend's native finish reason onto the normalized ``StopReason``.

    A recognized reason maps directly; an unrecognized one maps to
    ``"error"`` rather than being silently misclassified as a normal stop.
    A stream that never reported any finish reason at all -- which a
    well-behaved backend should not produce -- defaults to ``"end_turn"``
    rather than manufacturing a spurious error for what otherwise looks
    like an ordinary completion.
    """
    if finish_reason is None:
        return "end_turn"
    return _FINISH_REASON_MAP.get(finish_reason, "error")


def _usage_event_from_litellm_usage(usage: Any) -> UsageEvent:
    """Map a litellm usage object onto the normalized ``UsageEvent``.

    Every field defaults to ``0`` when the backend omitted it, including
    ``cached_tokens`` when the backend reports no ``prompt_tokens_details``
    at all -- the common case for backends without a prompt cache.
    """
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = (
        getattr(prompt_tokens_details, "cached_tokens", None)
        if prompt_tokens_details
        else None
    )
    return UsageEvent(
        input_tokens=getattr(usage, "prompt_tokens", None) or 0,
        output_tokens=getattr(usage, "completion_tokens", None) or 0,
        cached_tokens=cached_tokens or 0,
    )


def _extract_retry_after(exc: Exception) -> float | None:
    """Read a ``Retry-After`` hint off a litellm rate-limit exception, if any.

    LiteLLM attaches the failed HTTP response's headers to every mapped
    exception via the ``litellm_response_headers`` attribute (its own
    documented mechanism for "accurate retry logic") rather than via the
    exception's ``response`` attribute, which is reconstructed internally
    and does not reliably carry the original headers. ``None`` covers both
    a backend that never sent the header and one that reported an
    unparseable value.
    """
    headers = getattr(exc, "litellm_response_headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _map_error(exc: Exception, *, model_id: str, backend: str) -> ProviderError:
    """Translate a caught litellm exception into Kestrel's typed error taxonomy."""
    if isinstance(exc, LiteLLMAuthenticationError):
        return AuthError(str(exc), model_id=model_id, backend=backend)
    if isinstance(exc, LiteLLMContextWindowExceededError):
        return ContextOverflowError(str(exc), model_id=model_id, backend=backend)
    if isinstance(exc, LiteLLMRateLimitError):
        return RateLimitError(
            str(exc),
            model_id=model_id,
            backend=backend,
            retry_after_s=_extract_retry_after(exc),
        )
    # litellm.APIError, ServiceUnavailableError, Timeout, and any other
    # mapped failure land here -- a generic, unretriable backend failure.
    return ServerError(
        str(exc),
        model_id=model_id,
        backend=backend,
        status=getattr(exc, "status_code", None),
    )


@dataclass
class _PendingToolCall:
    """Accumulator for one tool call's deltas as they arrive across chunks."""

    id: str = ""
    name: str = ""
    arguments_json: str = ""


@dataclass
class _StreamNormalizer:
    """Folds a sequence of litellm streaming chunks into the event grammar.

    Text deltas are surfaced immediately from :meth:`feed` -- assistant-
    visible text should reach the caller as it arrives. Tool-call deltas
    are accumulated internally instead (a single tool call's arguments
    routinely arrive split across many chunks) and only surface, fully
    assembled, from :meth:`finish`, alongside the closing usage and stop
    events -- matching the grammar's requirement that every
    ``ToolCallEvent`` precede the terminal ``UsageEvent``/``StopEvent`` pair.
    """

    _tool_calls: dict[int, _PendingToolCall] = field(default_factory=dict)
    _usage: Any = None
    _finish_reason: str | None = None

    def feed(self, chunk: Any) -> Iterator[StreamEvent]:
        """Process one streamed chunk, yielding any events it directly produces."""
        choices = getattr(chunk, "choices", None) or []
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    yield TextDelta(text=content)
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    self._accumulate_tool_calls(tool_calls)
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                self._finish_reason = finish_reason
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = usage

    def finish(self) -> Iterator[StreamEvent]:
        """Flush any accumulated tool calls, then the closing usage and stop events."""
        for index in sorted(self._tool_calls):
            pending = self._tool_calls[index]
            yield ToolCallEvent(
                id=pending.id, name=pending.name, arguments_json=pending.arguments_json
            )
        if self._usage is not None:
            yield _usage_event_from_litellm_usage(self._usage)
        else:
            logger.warning(
                "backend reported no usage for this turn; recording zeroed usage"
            )
            yield UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0)
        yield StopEvent(reason=_map_finish_reason(self._finish_reason))

    def _accumulate_tool_calls(self, tool_calls: Iterable[Any]) -> None:
        """Fold one chunk's tool-call deltas into their per-index accumulators."""
        for tool_call in tool_calls:
            index = getattr(tool_call, "index", 0)
            pending = self._tool_calls.setdefault(index, _PendingToolCall())
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id:
                pending.id = tool_call_id
            function = getattr(tool_call, "function", None)
            if function is not None:
                name = getattr(function, "name", None)
                if name:
                    pending.name = name
                arguments = getattr(function, "arguments", None)
                if arguments:
                    pending.arguments_json += arguments


def _events_from_response(
    response: Any, *, model_id: str, backend: str
) -> list[StreamEvent]:
    """Normalize a single, buffered (``stream=False``) litellm response.

    Produces the same event grammar as the streaming path -- one
    ``TextDelta`` holding the full response text instead of incremental
    chunks, any complete tool calls, the turn's usage, and the closing
    stop event.

    Raises:
        ServerError: ``response.choices`` is empty. A well-formed chat
            completion always has at least one choice; a backend that
            returns none is a malformed response, not something this
            adapter should crash on with a raw ``IndexError``.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ServerError(
            "backend returned a response with no choices",
            model_id=model_id,
            backend=backend,
        )
    choice = choices[0]
    message = choice.message
    events: list[StreamEvent] = []

    content = getattr(message, "content", None)
    if content:
        events.append(TextDelta(text=content))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.function
            events.append(
                ToolCallEvent(
                    id=getattr(tool_call, "id", None) or "",
                    name=getattr(function, "name", None) or "",
                    arguments_json=getattr(function, "arguments", None) or "",
                )
            )

    usage = getattr(response, "usage", None)
    if usage is not None:
        events.append(_usage_event_from_litellm_usage(usage))
    else:
        logger.warning(
            "backend reported no usage for this turn; recording zeroed usage"
        )
        events.append(UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0))

    events.append(
        StopEvent(reason=_map_finish_reason(getattr(choice, "finish_reason", None)))
    )
    return events


class LiteLLMClient(ProviderClient):
    """``ProviderClient`` implementation over ``litellm.acompletion``.

    One instance is bound to a single :class:`~kestrel.registry.model.Registry`
    and can serve any model entry in it; the entry's ``backend`` decides how
    (see :func:`_litellm_params`), not the caller.
    """

    def __init__(self, registry: Registry) -> None:
        """Bind this client to the registry used to resolve model ids."""
        self._registry = registry

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream (or buffer) a completion for ``model_id``, satisfying ``ProviderClient``.

        See :class:`kestrel.provider.base.ProviderClient` for the grammar and
        error-handling contract this must uphold. ``effort`` is accepted and
        logged, not yet acted on -- mapping it onto backend-specific
        reasoning parameters is out of scope until model tiering lands.
        """
        entry = self._registry.get(model_id)
        api_key = _require_api_key(entry)
        litellm_params = _litellm_params(entry)
        litellm_tools = _tool_schemas_to_litellm(tools)
        litellm_messages: list[Any] = list(messages)
        logger.debug(
            "effort=%r accepted for model_id=%r (not yet acted on)", effort, model_id
        )

        try:
            if stream:
                normalizer = _StreamNormalizer()
                response: Any = await litellm.acompletion(
                    **litellm_params,
                    messages=litellm_messages,
                    tools=litellm_tools,
                    api_key=api_key,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=_REQUEST_TIMEOUT_S,
                )
                async for chunk in response:
                    for event in normalizer.feed(chunk):
                        yield event
                for event in normalizer.finish():
                    yield event
            else:
                response = await litellm.acompletion(
                    **litellm_params,
                    messages=litellm_messages,
                    tools=litellm_tools,
                    api_key=api_key,
                    stream=False,
                    timeout=_REQUEST_TIMEOUT_S,
                )
                for event in _events_from_response(
                    response, model_id=entry.id, backend=entry.backend
                ):
                    yield event
        except _MAPPED_LITELLM_ERRORS as exc:
            raise _map_error(exc, model_id=entry.id, backend=entry.backend) from exc
