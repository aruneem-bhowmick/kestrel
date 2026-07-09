"""Unit tests for turn execution, slash-command dispatch, and the REPL loop.

A scripted ``FakeProviderClient`` stands in for a real backend throughout:
these tests are about the REPL's own bookkeeping (history, cost, model
hot-swap, error handling), not about any provider adapter, so nothing
here touches the network or a real event loop beyond what ``run_turn``
itself drives.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any

import pytest

from kestrel.config import KestrelConfig
from kestrel.cost.meter import CostMeter, format_cost_line
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.errors import AuthError
from kestrel.provider.events import StopEvent, StreamEvent, TextDelta, UsageEvent
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import (
    ReplSession,
    _dispatch_command,
    _run_repl_loop,
    run_repl,
    run_turn,
)

pytestmark = [pytest.mark.p008, pytest.mark.unit]


def _entry(*, id: str, backend: str, **overrides: Any) -> ModelEntry:
    """Build a valid ModelEntry for ``id``/``backend``, overriding only
    the fields a given test cares about."""
    fields: dict[str, Any] = {
        "id": id,
        "backend": backend,
        "provider_model": "z-ai/glm-5.2" if backend == "openrouter" else "glm-5.2",
        "api_key_env": (
            "OPENROUTER_API_KEY" if backend == "openrouter" else "ZAI_API_KEY"
        ),
        "context_window": 200_000,
        "max_output": 16_384,
        "usd_per_mtok_input": Decimal("0.60"),
        "usd_per_mtok_output": Decimal("2.20"),
        "usd_per_mtok_cached": Decimal("0.11"),
        "supports_tools": True,
        "supports_cache": True,
    }
    if backend == "zai":
        fields["endpoint"] = "https://example.invalid/v1"
    fields.update(overrides)
    return ModelEntry(**fields)


def _registry() -> Registry:
    """A two-entry registry (one openrouter, one zai) for hot-swap tests."""
    return Registry(
        models={
            "glm-5.2": _entry(id="glm-5.2", backend="openrouter"),
            "glm-5.2-zai": _entry(id="glm-5.2-zai", backend="zai"),
        },
        source=None,
    )


class FakeProviderClient:
    """A scripted ``ProviderClient`` stand-in.

    Each call to :meth:`complete` pops the next scripted item list off an
    internal queue and yields its events in order; any ``BaseException``
    instance found in that list is raised in place instead of yielded,
    simulating a provider failure (or a keyboard interrupt) partway
    through a stream.
    """

    def __init__(self, scripts: list[list[StreamEvent | BaseException]]) -> None:
        """Queue up one scripted item list per expected ``complete`` call."""
        self._scripts = list(scripts)
        self.call_count = 0
        self.model_ids_called: list[str] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted item list, raising any exception inline."""
        self.call_count += 1
        self.model_ids_called.append(model_id)
        script = self._scripts.pop(0)
        for item in script:
            if isinstance(item, BaseException):
                raise item
            yield item


def _session(client: FakeProviderClient, *, model_id: str = "glm-5.2") -> ReplSession:
    """Build a fresh session wired to ``client``, starting at ``model_id``."""
    return ReplSession(
        config=KestrelConfig(),
        registry=_registry(),
        client=client,
        meter=CostMeter(),
        active_model_id=model_id,
        history=[],
    )


@pytest.mark.sanity
async def test_successful_turn_appends_user_and_assistant_history_in_order() -> None:
    """Given a turn that streams text and stops normally, when it
    completes, then the user message and the full assistant reply are
    appended to history in that order."""
    client = FakeProviderClient(
        [
            [
                TextDelta("Hello"),
                TextDelta(" there"),
                UsageEvent(input_tokens=10, output_tokens=5, cached_tokens=0),
                StopEvent(reason="end_turn"),
            ]
        ]
    )
    session = _session(client)

    await run_turn(session, "hi", io.StringIO())

    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there"},
    ]


@pytest.mark.sanity
async def test_cost_line_printed_after_stop_matches_format_cost_line() -> None:
    """Given a turn that reports usage and stops, when it completes, then
    the exact string produced by ``format_cost_line`` for that turn
    appears in the rendered output."""
    client = FakeProviderClient(
        [
            [
                TextDelta("hi"),
                UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0),
                StopEvent(reason="end_turn"),
            ]
        ]
    )
    session = _session(client)
    out = io.StringIO()

    await run_turn(session, "hi", out)

    turn = session.meter.turns[0]
    expected_line = format_cost_line(turn, session.meter.session_usd)
    assert expected_line in out.getvalue()


async def test_provider_error_mid_stream_drops_the_failed_exchange() -> None:
    """Given a stream that yields partial text and then raises a
    ``ProviderError``, when the turn runs, then one error line is printed,
    history is left untouched, and the next turn still succeeds."""
    client = FakeProviderClient(
        [
            [
                TextDelta("partial "),
                AuthError("boom", model_id="glm-5.2", backend="openrouter"),
            ],
            [
                TextDelta("ok"),
                UsageEvent(input_tokens=1, output_tokens=1, cached_tokens=0),
                StopEvent(reason="end_turn"),
            ],
        ]
    )
    session = _session(client)
    first_out = io.StringIO()

    await run_turn(session, "first", first_out)

    assert session.history == []
    assert "auth error" in first_out.getvalue()

    second_out = io.StringIO()
    await run_turn(session, "second", second_out)

    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "second"}


def test_model_command_switches_active_model_and_preserves_history() -> None:
    """Given a session with existing history, when ``/model`` targets a
    known id, then the active model changes and history is preserved
    verbatim."""
    client = FakeProviderClient([])
    session = _session(client)
    session.history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    history_before = list(session.history)
    out = io.StringIO()

    _dispatch_command(session, "/model glm-5.2-zai", out)

    assert session.active_model_id == "glm-5.2-zai"
    assert session.history == history_before


def test_model_command_unknown_id_leaves_model_and_history_unchanged() -> None:
    """Given a session, when ``/model`` targets an unknown id, then the
    error lists available ids and neither the active model nor history
    changes."""
    client = FakeProviderClient([])
    session = _session(client)
    session.history = [{"role": "user", "content": "hi"}]
    history_before = list(session.history)
    out = io.StringIO()

    _dispatch_command(session, "/model nope", out)

    assert "unknown model id 'nope'" in out.getvalue()
    assert "glm-5.2" in out.getvalue()
    assert session.active_model_id == "glm-5.2"
    assert session.history == history_before


def test_model_command_with_no_argument_prints_usage() -> None:
    """Given ``/model`` with no id argument, when dispatched, then a
    usage message is printed and the active model is unchanged."""
    client = FakeProviderClient([])
    session = _session(client)
    out = io.StringIO()

    _dispatch_command(session, "/model", out)

    assert "usage: /model" in out.getvalue()
    assert session.active_model_id == "glm-5.2"


def test_models_command_lists_ids_marking_the_active_one() -> None:
    """Given a two-entry registry, when ``/models`` runs, then both ids
    are listed and only the active one is marked."""
    client = FakeProviderClient([])
    session = _session(client, model_id="glm-5.2-zai")
    out = io.StringIO()

    _dispatch_command(session, "/models", out)

    lines = out.getvalue().splitlines()
    assert any(line.startswith("* glm-5.2-zai") for line in lines)
    assert any(line.startswith("  glm-5.2 ") for line in lines)


def test_cost_command_prints_session_total_and_per_turn_breakdown() -> None:
    """Given a session with one recorded turn, when ``/cost`` runs, then
    it prints the session total and a per-turn line."""
    client = FakeProviderClient([])
    session = _session(client)
    session.meter.record(
        UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0),
        session.registry.get("glm-5.2"),
    )
    out = io.StringIO()

    _dispatch_command(session, "/cost", out)

    assert "session total:" in out.getvalue()
    assert "1." in out.getvalue()


def test_help_command_lists_every_command() -> None:
    """Given any session, when ``/help`` runs, then every documented
    command name appears in the output."""
    client = FakeProviderClient([])
    session = _session(client)
    out = io.StringIO()

    _dispatch_command(session, "/help", out)

    output = out.getvalue()
    for command in ("/model", "/models", "/cost", "/help", "/quit"):
        assert command in output


@pytest.mark.sanity
def test_unknown_command_is_not_sent_to_the_client() -> None:
    """Given an unrecognized slash command, when dispatched, then an
    error is printed and the provider client is never called."""
    client = FakeProviderClient([])
    session = _session(client)
    out = io.StringIO()

    _dispatch_command(session, "/frobnicate", out)

    assert client.call_count == 0
    assert "unknown command" in out.getvalue()


@pytest.mark.sanity
def test_eof_exits_the_loop_cleanly() -> None:
    """Given ``input_fn`` raises ``EOFError`` immediately, when the loop
    runs, then it exits with code 0 without calling the client."""
    client = FakeProviderClient([])
    session = _session(client)

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    exit_code = _run_repl_loop(session, input_fn=_raise_eof, out=io.StringIO())

    assert exit_code == 0
    assert client.call_count == 0


@pytest.mark.sanity
def test_empty_line_does_not_call_the_client() -> None:
    """Given a blank input line followed by ``/quit``, when the loop
    runs, then the client is never called and the loop exits 0."""
    client = FakeProviderClient([])
    session = _session(client)
    lines = iter(["", "   ", "/quit"])

    exit_code = _run_repl_loop(
        session, input_fn=lambda _prompt: next(lines), out=io.StringIO()
    )

    assert exit_code == 0
    assert client.call_count == 0


def test_quit_command_exits_the_loop_with_code_zero() -> None:
    """Given ``/quit`` as the first input, when the loop runs, then it
    returns exit code 0 immediately."""
    client = FakeProviderClient([])
    session = _session(client)

    exit_code = _run_repl_loop(
        session, input_fn=lambda _prompt: "/quit", out=io.StringIO()
    )

    assert exit_code == 0


def test_keyboard_interrupt_mid_turn_cancels_only_that_turn() -> None:
    """Given a turn that raises ``KeyboardInterrupt`` mid-stream, when the
    loop runs, then the turn is cancelled but the loop keeps going and
    still processes the next line."""
    client = FakeProviderClient([[TextDelta("partial"), KeyboardInterrupt()]])
    session = _session(client)
    lines = iter(["hello", "/quit"])
    out = io.StringIO()

    exit_code = _run_repl_loop(session, input_fn=lambda _prompt: next(lines), out=out)

    assert exit_code == 0
    assert "[turn cancelled]" in out.getvalue()
    assert session.history == []


def test_non_slash_input_streams_a_turn_and_preserves_model_swap_across_it() -> None:
    """Given a scripted turn, a model swap, and a second scripted turn,
    when driven through the public ``run_repl`` entry point, then both
    turns' streamed text reaches the output and the loop exits cleanly."""
    client = FakeProviderClient(
        [
            [
                TextDelta("Hello from GLM-5.2"),
                UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0),
                StopEvent(reason="end_turn"),
            ],
            [
                TextDelta("Hello from Z.ai GLM"),
                UsageEvent(input_tokens=40, output_tokens=6, cached_tokens=0),
                StopEvent(reason="end_turn"),
            ],
        ]
    )
    lines = iter(["hello", "/model glm-5.2-zai", "hello again", "/quit"])
    out = io.StringIO()

    exit_code = run_repl(
        KestrelConfig(),
        _registry(),
        client,
        "glm-5.2",
        input_fn=lambda _prompt: next(lines),
        out=out,
    )

    output = out.getvalue()
    assert exit_code == 0
    assert client.call_count == 2
    assert client.model_ids_called == ["glm-5.2", "glm-5.2-zai"]
    assert "Hello from GLM-5.2" in output
    assert output.index("Hello from GLM-5.2") < output.index("Hello from Z.ai GLM")
