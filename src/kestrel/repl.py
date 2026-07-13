"""A plain-terminal REPL: streams completions, prices them, hot-swaps models.

No tool calls, no agent loop, no TUI live here -- this module owns exactly
the user-facing loop that reads a line, streams a completion for it,
prices the turn, and renders both safely to a real terminal. ``/model``
re-points subsequent turns at a different registry entry without losing
the conversation built up so far: the same ``history`` list is simply sent
to whichever model is active next.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, TextIO, assert_never

from kestrel import __version__
from kestrel.config import KestrelConfig
from kestrel.cost.meter import CostMeter, TurnCost, format_cost_line
from kestrel.kestrel_md import KestrelMd
from kestrel.provider.base import Message, ProviderClient
from kestrel.provider.cache import build_stable_prefix, mark_cache_breakpoints
from kestrel.provider.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from kestrel.provider.events import StopEvent, TextDelta, ToolCallEvent, UsageEvent
from kestrel.registry.model import Registry, UnknownModelError

SYSTEM_PROMPT: Final[str] = "You are Kestrel, a terminal coding assistant. Be concise."

_PROMPT: Final[str] = "kestrel> "
_EXIT_COMMANDS: Final[frozenset[str]] = frozenset({"/quit", "/exit"})

# Every ProviderError subclass gets a short, human label; anything else
# (a future subclass this module doesn't know about yet) falls back to the
# generic label rather than raising on an unrecognized type.
_ERROR_LABELS: Final[dict[type[ProviderError], str]] = {
    AuthError: "auth error",
    RateLimitError: "rate limit error",
    ContextOverflowError: "context overflow error",
    ServerError: "server error",
}

# A full ANSI/CSI/OSC escape sequence, in either its 7-bit (ESC-prefixed) or
# 8-bit (single C1 introducer byte) form. Matched and stripped as whole
# sequences first so a lone final byte (e.g. the "J" in "\x1b[2J") is never
# left behind looking like ordinary text.
_ANSI_ESCAPE_RE = re.compile(
    r"""
    (?: \x1b\[ | \x9b )              # CSI introducer, 7-bit or 8-bit
    [0-9:;<=>?]* [ -/]* [@-~]        # parameter/intermediate/final bytes
    |
    (?: \x1b\] | \x9d )               # OSC introducer, 7-bit or 8-bit
    [^\x07\x1b]*                      # body, up to the terminator
    (?: \x07 | \x1b\\ | \x9c )        # BEL, or ST (7-bit or 8-bit)
    |
    \x1b [@-Z\\-_]                    # any other two-byte Fe escape
    """,
    re.VERBOSE,
)
# Any C0 control byte other than newline/tab, plus any C1 control byte, that
# survived the escape-sequence pass above (e.g. a bare carriage return, or a
# lone escape/CSI-introducer byte not part of a well-formed sequence).
_STRAY_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x80-\x9f]")


def sanitize_terminal(text: str) -> str:
    """Strip terminal control sequences from untrusted model output.

    All file contents, tool outputs, and model completions are untrusted
    data; printing one verbatim to a real terminal would let a hostile
    completion clear the screen, retitle the window, or otherwise inject
    control sequences of its own. Newlines and tabs pass through
    unchanged; every other C0 control byte, every C1 control byte, and
    every recognized ANSI/CSI/OSC escape sequence (7-bit or 8-bit) is
    removed. Ordinary multibyte Unicode text is never touched.
    """
    without_escapes = _ANSI_ESCAPE_RE.sub("", text)
    return _STRAY_CONTROL_RE.sub("", without_escapes)


@dataclass
class ReplSession:
    """Mutable state carried across turns and commands for one REPL run.

    Attributes:
        config: The loaded configuration this session started with.
        registry: The model registry every ``/model`` lookup resolves against.
        client: The provider client every turn streams a completion from.
        meter: Accumulates priced turns across the whole session.
        active_model_id: The registry id the next turn will be sent to.
        history: Conversation turns so far, excluding the system prompt
            (which is prepended fresh on every call). Preserved verbatim
            across a ``/model`` hot-swap.
        kestrel_md: The working directory's project-memory file, loaded
            once when the session starts and never reloaded mid-session
            -- keeping it fixed is what lets the leading prefix built
            from it stay byte-identical across every turn, including
            across a ``/model`` hot-swap. ``None`` when the working
            directory has no ``KESTREL.md``.
    """

    config: KestrelConfig
    registry: Registry
    client: ProviderClient
    meter: CostMeter
    active_model_id: str
    history: list[Message] = field(default_factory=list)
    kestrel_md: KestrelMd | None = None


def _format_provider_error(exc: ProviderError) -> str:
    """Render a caught provider failure as a one-line REPL message."""
    label = _ERROR_LABELS.get(type(exc), "provider error")
    return f"{label}: {exc}"


async def run_turn(session: ReplSession, user_text: str, out: TextIO) -> None:
    """Stream one completion for ``user_text`` against the active model.

    Text deltas are sanitized and printed as they arrive, flushed after
    each one; the closing usage event is priced through ``session.meter``,
    and the per-turn cost line is printed once the stream's stop event
    arrives. The user message and the raw (unsanitized) assistant reply
    are appended to ``session.history`` only once the turn completes
    successfully. On a ``ProviderError`` mid-stream, one error line is
    printed and the failed exchange is dropped entirely -- history is left
    exactly as it was, so the next turn resumes cleanly.

    The leading messages sent ahead of history come from
    ``build_stable_prefix``, folding ``session.kestrel_md`` in when
    present, so every turn of the session -- even after a ``/model``
    hot-swap -- sends the same prefix a cache-capable backend can reuse.
    """
    entry = session.registry.get(session.active_model_id)
    user_message: Message = {"role": "user", "content": user_text}
    prefix = mark_cache_breakpoints(
        build_stable_prefix(SYSTEM_PROMPT, session.kestrel_md), entry
    )
    messages: list[Message] = [*prefix, *session.history, user_message]

    raw_chunks: list[str] = []
    turn_cost: TurnCost | None = None
    try:
        async for event in session.client.complete(
            messages, None, session.active_model_id, "high", stream=True
        ):
            match event:
                case TextDelta(text=text):
                    raw_chunks.append(text)
                    out.write(sanitize_terminal(text))
                    out.flush()
                case UsageEvent():
                    turn_cost = session.meter.record(event, entry)
                case StopEvent():
                    assert turn_cost is not None
                    out.write("\n")
                    out.write(format_cost_line(turn_cost, session.meter.session_usd))
                    out.write("\n")
                case ToolCallEvent():
                    pass  # dormant: this call never offers tools
                case _:
                    assert_never(event)
    except ProviderError as exc:
        out.write(f"{_format_provider_error(exc)}\n")
        return

    session.history.append(user_message)
    session.history.append({"role": "assistant", "content": "".join(raw_chunks)})


def _cmd_model(session: ReplSession, argument: str, out: TextIO) -> None:
    """Handle ``/model <id>``: hot-swap the active model, keeping history.

    An unknown id leaves ``active_model_id`` (and history) untouched and
    prints the error, which already names every available id.
    """
    target = argument.strip()
    if not target:
        out.write("usage: /model <id>\n")
        return
    try:
        session.registry.get(target)
    except UnknownModelError as exc:
        out.write(f"{exc}\n")
        return
    session.active_model_id = target
    out.write(f"active model: {target}\n")


def _cmd_models(session: ReplSession, out: TextIO) -> None:
    """Handle ``/models``: list every registered id, marking the active one."""
    for model_id in session.registry.ids():
        entry = session.registry.get(model_id)
        marker = "*" if model_id == session.active_model_id else " "
        tags = ", ".join(sorted(entry.tags)) or "-"
        out.write(f"{marker} {model_id} ({entry.backend}) [{tags}]\n")


def _cmd_cost(session: ReplSession, out: TextIO) -> None:
    """Handle ``/cost``: print the running session total and a per-turn table."""
    out.write(f"session total: ${session.meter.session_usd:.4f}\n")
    for index, turn in enumerate(session.meter.turns, start=1):
        out.write(f"  {index}. {format_cost_line(turn, session.meter.session_usd)}\n")


def _cmd_help(out: TextIO) -> None:
    """Handle ``/help``: list every REPL command."""
    out.write(
        "commands:\n"
        "  /model <id>   switch the active model (history is preserved)\n"
        "  /models       list every registered model id\n"
        "  /cost         show session and per-turn cost\n"
        "  /help         show this message\n"
        "  /quit         exit\n"
    )


def _dispatch_command(session: ReplSession, line: str, out: TextIO) -> None:
    """Parse and execute one ``/``-prefixed command line.

    An unrecognized command prints an error and never reaches the model --
    only ``run_turn`` calls ``session.client.complete``.
    """
    command, _, argument = line.partition(" ")
    if command == "/model":
        _cmd_model(session, argument, out)
    elif command == "/models":
        _cmd_models(session, out)
    elif command == "/cost":
        _cmd_cost(session, out)
    elif command == "/help":
        _cmd_help(out)
    else:
        out.write(f"unknown command: {command} (try /help)\n")


def _print_banner(session: ReplSession, out: TextIO) -> None:
    """Print the startup banner naming the version and active model."""
    out.write(f"kestrel {__version__} -- active model: {session.active_model_id}\n")
    out.write("plain REPL: no tools, no agent loop. Type /help for commands.\n")


def _run_repl_loop(
    session: ReplSession, *, input_fn: Callable[[str], str], out: TextIO
) -> int:
    """Drive the read-eval-print loop against an already-built session.

    Empty lines re-prompt without calling the model. Slash commands are
    dispatched locally and never reach the model. ``EOFError`` from
    ``input_fn`` (piped stdin exhausted, or Ctrl-D) exits cleanly with
    ``0``. A ``KeyboardInterrupt`` raised while a turn is streaming cancels
    only that turn -- the session, and the loop, continue.
    """
    _print_banner(session, out)
    while True:
        try:
            line = input_fn(_PROMPT)
        except EOFError:
            return 0

        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _EXIT_COMMANDS:
            return 0
        if stripped.startswith("/"):
            _dispatch_command(session, stripped, out)
            continue

        try:
            asyncio.run(run_turn(session, stripped, out))
        except KeyboardInterrupt:
            out.write("\n[turn cancelled]\n")


def run_repl(
    config: KestrelConfig,
    registry: Registry,
    client: ProviderClient,
    model_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
) -> int:
    """Run the interactive REPL until ``/quit``, EOF, or a fatal signal.

    ``model_id`` must already be a valid registry id -- callers resolve
    and validate the starting model (honoring an explicit override versus
    the configured default) before entering the loop, so an unknown
    starting model is a caller-side error, not something this loop handles.
    """
    session = ReplSession(
        config=config,
        registry=registry,
        client=client,
        meter=CostMeter(),
        active_model_id=model_id,
        history=[],
    )
    return _run_repl_loop(session, input_fn=input_fn, out=out)
