"""Unit tests for the LiteLLM chunk normalizer, dispatch, and error mapping.

Every case here is deterministic and network-free: chunks are hand-built
fakes shaped like litellm's own streaming objects (plain ``SimpleNamespace``
trees exposing only the attributes the normalizer reads), not real litellm
responses -- the mock-server-backed integration suite is what exercises
this code against genuinely produced litellm objects.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from kestrel.provider.base import ToolSchema
from kestrel.provider.errors import (
    AuthError,
    ContextOverflowError,
    RateLimitError,
    ServerError,
)
from kestrel.provider.events import StopEvent, TextDelta, ToolCallEvent, UsageEvent
from kestrel.provider.litellm_client import (
    LiteLLMClient,
    _events_from_response,
    _extract_retry_after,
    _litellm_params,
    _map_error,
    _map_finish_reason,
    _require_api_key,
    _StreamNormalizer,
    _tool_schemas_to_litellm,
    _usage_event_from_litellm_usage,
)
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p005, pytest.mark.unit]


def _entry(**overrides: Any) -> ModelEntry:
    """Build a valid ModelEntry, overriding only the fields a test cares about."""
    fields: dict[str, Any] = {
        "id": "glm-5.2",
        "backend": "openrouter",
        "provider_model": "z-ai/glm-5.2",
        "api_key_env": "OPENROUTER_API_KEY",
        "context_window": 200_000,
        "max_output": 16_384,
        "usd_per_mtok_input": Decimal("0.60"),
        "usd_per_mtok_output": Decimal("2.20"),
        "usd_per_mtok_cached": Decimal("0.11"),
        "supports_tools": True,
        "supports_cache": True,
    }
    fields.update(overrides)
    return ModelEntry(**fields)


def _registry(*entries: ModelEntry) -> Registry:
    """Build a Registry keyed by each entry's own id."""
    return Registry(models={entry.id: entry for entry in entries}, source=None)


def _delta_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    """Build a fake streaming chunk exposing one ``choices[0].delta``."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _usage_chunk(
    *, prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None
) -> Any:
    """Build a fake streaming chunk carrying only a terminal ``usage`` payload."""
    details = (
        None if cached_tokens is None else SimpleNamespace(cached_tokens=cached_tokens)
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=details,
    )
    return SimpleNamespace(choices=[], usage=usage)


def _tool_call_delta(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    """Build one fake tool-call delta entry, as would appear in ``delta.tool_calls``."""
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


def _feed(normalizer: _StreamNormalizer, chunk: Any) -> None:
    """Drive a chunk through the normalizer, discarding any yielded events.

    ``feed`` is a generator function -- it does nothing until iterated --
    so tests that only care about its effect on the normalizer's internal
    state (not the events it yields immediately) must still consume it.
    """
    for _ in normalizer.feed(chunk):
        pass


# --- _StreamNormalizer: text deltas -----------------------------------------


def test_text_deltas_are_yielded_immediately_and_in_order() -> None:
    """Given chunks with successive content fragments, when fed to the
    normalizer, then each yields its TextDelta immediately, in arrival order."""
    normalizer = _StreamNormalizer()

    first = list(normalizer.feed(_delta_chunk(content="Hello")))
    second = list(normalizer.feed(_delta_chunk(content=" world")))

    assert first == [TextDelta(text="Hello")]
    assert second == [TextDelta(text=" world")]


def test_empty_or_absent_content_yields_no_text_delta() -> None:
    """Given a chunk whose delta has no content, when fed, then it yields
    nothing -- a content-free delta (e.g. a role-only opener) is not text."""
    normalizer = _StreamNormalizer()

    events = list(normalizer.feed(_delta_chunk(content=None)))

    assert events == []


# --- _StreamNormalizer: usage mapping ----------------------------------------


@pytest.mark.sanity
def test_usage_mapping_defaults_cached_tokens_to_zero() -> None:
    """Given a usage-bearing chunk with no prompt_tokens_details, when the
    stream finishes, then cached_tokens defaults to 0."""
    normalizer = _StreamNormalizer()
    _feed(normalizer, _delta_chunk(content="hi", finish_reason="stop"))
    _feed(normalizer, _usage_chunk(prompt_tokens=42, completion_tokens=7))

    events = list(normalizer.finish())

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage == UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0)


def test_usage_mapping_carries_explicit_cached_tokens() -> None:
    """Given a usage-bearing chunk that reports cached tokens, when the
    stream finishes, then cached_tokens is carried through unchanged."""
    normalizer = _StreamNormalizer()
    _feed(normalizer, _delta_chunk(content="hi", finish_reason="stop"))
    _feed(
        normalizer,
        _usage_chunk(prompt_tokens=100, completion_tokens=10, cached_tokens=25),
    )

    usage = next(e for e in normalizer.finish() if isinstance(e, UsageEvent))

    assert usage.cached_tokens == 25


# --- _StreamNormalizer: finish-reason mapping --------------------------------


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("native_reason", "expected"),
    [("stop", "end_turn"), ("tool_calls", "tool_use"), ("length", "max_tokens")],
)
def test_finish_reason_mapping(native_reason: str, expected: str) -> None:
    """Given each recognized native finish reason, when the stream finishes,
    then the closing StopEvent carries the corresponding normalized reason."""
    normalizer = _StreamNormalizer()
    _feed(normalizer, _delta_chunk(content="hi", finish_reason=native_reason))
    _feed(normalizer, _usage_chunk(prompt_tokens=1, completion_tokens=1))

    stop = next(e for e in normalizer.finish() if isinstance(e, StopEvent))

    assert stop.reason == expected


def test_unrecognized_finish_reason_maps_to_error() -> None:
    """Given a finish reason outside the recognized set, when mapped, then
    it becomes "error" rather than being silently misclassified."""
    assert _map_finish_reason("some_new_provider_reason") == "error"


def test_missing_finish_reason_defaults_to_end_turn() -> None:
    """Given no finish reason was ever observed, when mapped, then it
    defaults to "end_turn" rather than manufacturing a spurious error."""
    assert _map_finish_reason(None) == "end_turn"


# --- _StreamNormalizer: missing usage ----------------------------------------


def test_missing_usage_synthesizes_zeros_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given a stream that never reports usage, when it finishes, then a
    zeroed UsageEvent is synthesized and a warning is logged -- a cost of
    $0.0000 is a visible signal that something is wrong, not a silent gap."""
    normalizer = _StreamNormalizer()
    _feed(normalizer, _delta_chunk(content="hi", finish_reason="stop"))

    with caplog.at_level("WARNING", logger="kestrel.provider"):
        events = list(normalizer.finish())

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage == UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0)
    assert "no usage" in caplog.text


# --- _StreamNormalizer: tool-call accumulation -------------------------------


def test_tool_call_deltas_accumulate_into_one_event() -> None:
    """Given a tool call's id/name/arguments arriving split across several
    chunks, when the stream finishes, then exactly one ToolCallEvent
    surfaces with the concatenated arguments_json."""
    normalizer = _StreamNormalizer()
    _feed(
        normalizer,
        _delta_chunk(tool_calls=[_tool_call_delta(0, id="call_1", name="read_file")]),
    )
    _feed(
        normalizer,
        _delta_chunk(tool_calls=[_tool_call_delta(0, arguments='{"path": ')]),
    )
    _feed(
        normalizer,
        _delta_chunk(
            tool_calls=[_tool_call_delta(0, arguments='"a.py"}')],
            finish_reason="tool_calls",
        ),
    )
    _feed(normalizer, _usage_chunk(prompt_tokens=5, completion_tokens=5))

    events = list(normalizer.finish())

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert tool_calls == [
        ToolCallEvent(id="call_1", name="read_file", arguments_json='{"path": "a.py"}')
    ]
    # Grammar: every ToolCallEvent precedes the closing usage/stop pair.
    assert isinstance(events[-2], UsageEvent)
    assert isinstance(events[-1], StopEvent)


def test_multiple_tool_calls_accumulate_independently_by_index() -> None:
    """Given two concurrent tool calls interleaved by index, when the
    stream finishes, then each surfaces as its own ToolCallEvent, ordered
    by index."""
    normalizer = _StreamNormalizer()
    _feed(
        normalizer,
        _delta_chunk(
            tool_calls=[
                _tool_call_delta(0, id="call_0", name="a", arguments="{}"),
                _tool_call_delta(1, id="call_1", name="b", arguments="{}"),
            ],
            finish_reason="tool_calls",
        ),
    )
    _feed(normalizer, _usage_chunk(prompt_tokens=1, completion_tokens=1))

    tool_calls = [e for e in normalizer.finish() if isinstance(e, ToolCallEvent)]

    assert tool_calls == [
        ToolCallEvent(id="call_0", name="a", arguments_json="{}"),
        ToolCallEvent(id="call_1", name="b", arguments_json="{}"),
    ]


# --- _usage_event_from_litellm_usage -----------------------------------------


def test_usage_event_defaults_missing_fields_to_zero() -> None:
    """Given a usage object missing every field, when mapped, then every
    resulting token count defaults to 0 rather than raising."""
    usage = SimpleNamespace(
        prompt_tokens=None, completion_tokens=None, prompt_tokens_details=None
    )

    event = _usage_event_from_litellm_usage(usage)

    assert event == UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0)


# --- _events_from_response (the stream=False path) ---------------------------


def _buffered_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = "stop",
    usage: Any = None,
) -> Any:
    """Build a fake buffered (``stream=False``) litellm response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_events_from_response_yields_one_text_delta_and_closing_events() -> None:
    """Given a buffered response with only text content, when normalized,
    then it yields one TextDelta holding the full text, the turn's usage,
    and a closing StopEvent -- the same grammar the streaming path yields
    incrementally, collapsed into a single pass."""
    usage = SimpleNamespace(
        prompt_tokens=42, completion_tokens=7, prompt_tokens_details=None
    )
    response = _buffered_response(content="Hello from GLM-5.2", usage=usage)

    events = _events_from_response(response)

    assert events == [
        TextDelta(text="Hello from GLM-5.2"),
        UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0),
        StopEvent(reason="end_turn"),
    ]


def test_events_from_response_includes_complete_tool_calls() -> None:
    """Given a buffered response whose message carries complete tool
    calls, when normalized, then each surfaces as its own ToolCallEvent
    ahead of the closing usage/stop pair."""
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="read_file", arguments='{"path": "a.py"}'),
    )
    usage = SimpleNamespace(
        prompt_tokens=5, completion_tokens=5, prompt_tokens_details=None
    )
    response = _buffered_response(
        tool_calls=[tool_call], finish_reason="tool_calls", usage=usage
    )

    events = _events_from_response(response)

    assert (
        ToolCallEvent(id="call_1", name="read_file", arguments_json='{"path": "a.py"}')
        in events
    )
    assert events[-1] == StopEvent(reason="tool_use")


def test_events_from_response_missing_usage_synthesizes_zeros_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given a buffered response with no usage at all, when normalized,
    then a zeroed UsageEvent is synthesized and a warning is logged."""
    response = _buffered_response(content="hi", usage=None)

    with caplog.at_level("WARNING", logger="kestrel.provider"):
        events = _events_from_response(response)

    assert UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0) in events
    assert "no usage" in caplog.text


# --- _litellm_params ----------------------------------------------------------


@pytest.mark.sanity
def test_litellm_params_for_openrouter_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an OpenRouter registry entry, when translated, then the model
    is prefixed for litellm's router and api_base defaults to the real
    OpenRouter endpoint when no test-seam override is set."""
    monkeypatch.delenv("KESTREL_OPENROUTER_BASE_URL", raising=False)
    entry = _entry(backend="openrouter", provider_model="z-ai/glm-5.2")

    params = _litellm_params(entry)

    assert params == {
        "model": "openrouter/z-ai/glm-5.2",
        "api_base": "https://openrouter.ai/api/v1",
    }


def test_litellm_params_openrouter_honors_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the KESTREL_OPENROUTER_BASE_URL test seam is set, when
    translated, then api_base uses it instead of the real endpoint."""
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", "http://127.0.0.1:9999/v1")
    entry = _entry(backend="openrouter")

    params = _litellm_params(entry)

    assert params["api_base"] == "http://127.0.0.1:9999/v1"


@pytest.mark.parametrize("backend", ["zai", "anthropic", "ollama"])
def test_litellm_params_raises_not_implemented_for_other_backends(backend: str) -> None:
    """Given a registry entry for a backend this adapter does not yet
    implement, when translated, then it raises NotImplementedError instead
    of silently falling back to some default routing."""
    endpoint = "https://example.invalid" if backend in ("zai", "ollama") else None
    entry = _entry(backend=backend, endpoint=endpoint)

    with pytest.raises(NotImplementedError, match=backend):
        _litellm_params(entry)


# --- _require_api_key ---------------------------------------------------------


@pytest.mark.sanity
def test_require_api_key_returns_value_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the entry's credential env var is set, when required, then
    its value is returned."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    entry = _entry(api_key_env="OPENROUTER_API_KEY")

    assert _require_api_key(entry) == "sk-test-value"


def test_require_api_key_raises_auth_error_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the entry's credential env var is not set, when required,
    then AuthError names the env var -- never a value, since none exists."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    entry = _entry(api_key_env="OPENROUTER_API_KEY")

    with pytest.raises(AuthError, match="OPENROUTER_API_KEY"):
        _require_api_key(entry)


def test_require_api_key_raises_auth_error_when_env_var_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the entry's credential env var is set to an empty string,
    when required, then it is treated the same as unset."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    entry = _entry(api_key_env="OPENROUTER_API_KEY")

    with pytest.raises(AuthError, match="OPENROUTER_API_KEY"):
        _require_api_key(entry)


def test_require_api_key_raises_auth_error_when_entry_has_no_env_configured() -> None:
    """Given a registry entry with no api_key_env at all, when required,
    then AuthError names the model rather than crashing on a None lookup."""
    entry = _entry(api_key_env=None)

    with pytest.raises(AuthError, match="glm-5.2"):
        _require_api_key(entry)


# --- absent-credential path never reaches the network ------------------------


async def test_missing_api_key_short_circuits_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no credential is available for the active model, when
    complete() is iterated, then AuthError is raised and litellm.acompletion
    is never invoked -- a missing key must never reach the network."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    async def _must_not_be_called(*args: Any, **kwargs: Any) -> Any:
        """Fail the test immediately if litellm.acompletion is ever reached."""
        raise AssertionError("litellm.acompletion must not be called")

    monkeypatch.setattr(litellm, "acompletion", _must_not_be_called)
    client = LiteLLMClient(_registry(_entry()))

    with pytest.raises(AuthError, match="OPENROUTER_API_KEY"):
        async for _ in client.complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model_id="glm-5.2",
            effort="high",
            stream=True,
        ):
            pass


# --- complete(): the stream=False path ---------------------------------------


async def test_complete_stream_false_uses_the_buffered_response_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given stream=False, when complete() is iterated, then it calls
    litellm.acompletion with stream=False and normalizes the single
    buffered response it gets back, rather than treating it as a chunk
    iterator."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    usage = SimpleNamespace(
        prompt_tokens=3, completion_tokens=2, prompt_tokens_details=None
    )
    response = _buffered_response(content="hi there", usage=usage)

    async def _fake_acompletion(**kwargs: Any) -> Any:
        """Stand in for litellm.acompletion, asserting stream=False was requested."""
        assert kwargs["stream"] is False
        return response

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    client = LiteLLMClient(_registry(_entry()))

    events = [
        event
        async for event in client.complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            model_id="glm-5.2",
            effort="high",
            stream=False,
        )
    ]

    assert events == [
        TextDelta(text="hi there"),
        UsageEvent(input_tokens=3, output_tokens=2, cached_tokens=0),
        StopEvent(reason="end_turn"),
    ]


# --- _tool_schemas_to_litellm --------------------------------------------------


def test_tool_schemas_to_litellm_returns_none_for_no_tools() -> None:
    """Given no tools were offered, when converted, then the result is
    None -- callers pass it straight through without a separate branch."""
    assert _tool_schemas_to_litellm(None) is None
    assert _tool_schemas_to_litellm([]) is None


def test_tool_schemas_to_litellm_renders_openai_function_shape() -> None:
    """Given a Kestrel tool schema, when converted, then it renders as the
    OpenAI-style function-tool dict litellm expects."""
    schema = ToolSchema(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {}},
    )

    rendered = _tool_schemas_to_litellm([schema])

    assert rendered == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# --- _extract_retry_after ------------------------------------------------------


def test_extract_retry_after_reads_litellm_response_headers() -> None:
    """Given a litellm exception carrying a Retry-After header via its
    litellm_response_headers attribute, when read, then the numeric value
    is returned."""
    exc = Exception("rate limited")
    exc.litellm_response_headers = {"retry-after": "12"}  # type: ignore[attr-defined]

    assert _extract_retry_after(exc) == 12.0


def test_extract_retry_after_returns_none_when_header_absent() -> None:
    """Given an exception with headers but no Retry-After entry, when
    read, then None is returned rather than raising."""
    exc = Exception("rate limited")
    exc.litellm_response_headers = {}  # type: ignore[attr-defined]

    assert _extract_retry_after(exc) is None


def test_extract_retry_after_returns_none_when_attribute_missing() -> None:
    """Given an exception with no litellm_response_headers attribute at
    all, when read, then None is returned."""
    assert _extract_retry_after(Exception("boom")) is None


def test_extract_retry_after_returns_none_for_unparseable_value() -> None:
    """Given a Retry-After value that is not a number, when read, then
    None is returned rather than propagating a ValueError."""
    exc = Exception("rate limited")
    exc.litellm_response_headers = {"retry-after": "not-a-number"}  # type: ignore[attr-defined]

    assert _extract_retry_after(exc) is None


# --- _map_error -----------------------------------------------------------------


def test_map_error_authentication_error() -> None:
    """Given a litellm AuthenticationError, when mapped, then it becomes
    Kestrel's AuthError."""
    exc = litellm.AuthenticationError(
        "bad key", llm_provider="openrouter", model="glm-5.2"
    )

    mapped = _map_error(exc, model_id="glm-5.2", backend="openrouter")

    assert isinstance(mapped, AuthError)


def test_map_error_context_window_exceeded() -> None:
    """Given a litellm ContextWindowExceededError, when mapped, then it
    becomes Kestrel's ContextOverflowError."""
    exc = litellm.ContextWindowExceededError(
        "too long", llm_provider="openrouter", model="glm-5.2"
    )

    mapped = _map_error(exc, model_id="glm-5.2", backend="openrouter")

    assert isinstance(mapped, ContextOverflowError)


def test_map_error_rate_limit_carries_retry_after() -> None:
    """Given a litellm RateLimitError whose response carried a Retry-After
    header, when mapped, then Kestrel's RateLimitError carries the same
    retry_after_s value."""
    exc = litellm.RateLimitError(
        "slow down", llm_provider="openrouter", model="glm-5.2"
    )
    exc.litellm_response_headers = {"retry-after": "5"}

    mapped = _map_error(exc, model_id="glm-5.2", backend="openrouter")

    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after_s == 5.0


@pytest.mark.parametrize(
    "exc",
    [
        litellm.APIError(
            status_code=500,
            message="boom",
            llm_provider="openrouter",
            model="glm-5.2",
            request=None,
        ),
        litellm.ServiceUnavailableError(
            "down", llm_provider="openrouter", model="glm-5.2"
        ),
        litellm.Timeout("slow", llm_provider="openrouter", model="glm-5.2"),
    ],
)
def test_map_error_generic_failures_become_server_error(exc: Exception) -> None:
    """Given any of litellm's generic-failure exception types, when
    mapped, then each becomes Kestrel's ServerError."""
    mapped = _map_error(exc, model_id="glm-5.2", backend="openrouter")

    assert isinstance(mapped, ServerError)


def test_map_error_carries_model_and_backend_in_message() -> None:
    """Given any mapped error, when stringified, then the message names
    both the model id and the backend that were active."""
    exc = litellm.AuthenticationError(
        "bad key", llm_provider="openrouter", model="glm-5.2"
    )

    mapped = _map_error(exc, model_id="glm-5.2", backend="openrouter")

    assert "glm-5.2" in str(mapped)
    assert "openrouter" in str(mapped)
