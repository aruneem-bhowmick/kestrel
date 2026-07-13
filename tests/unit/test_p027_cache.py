"""Unit tests for the byte-stable cache prefix: `build_stable_prefix`'s
message shape with and without `KESTREL.md` folded in, its repeat-call
stability, the optional `file_context` message, and
`mark_cache_breakpoints`'s marking rule across backends that do and do
not need an explicit cache boundary.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.kestrel_md import KestrelMd, VerifyCommands
from kestrel.provider.base import Message
from kestrel.provider.cache import (
    _MEMORY_BANNER,
    build_stable_prefix,
    mark_cache_breakpoints,
)
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p027, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p027_stable_prefix.golden"
)

_SYSTEM_PROMPT = "You are Kestrel, an autonomous coding agent."


def _entry(
    *,
    backend: str,
    supports_cache: bool,
    requires_explicit_cache_breakpoint: bool = False,
) -> ModelEntry:
    """Build a minimal, otherwise-valid `ModelEntry` for one backend and
    cache-support combination -- every field but the two under test is
    an arbitrary but valid placeholder. Direct backends (`zai`) require
    an `endpoint`, so one is always supplied even for backends that
    ignore it."""
    return ModelEntry(
        id="test-model",
        backend=backend,  # type: ignore[arg-type]
        provider_model="test/model",
        endpoint="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        context_window=100_000,
        max_output=8_192,
        usd_per_mtok_input=Decimal("1.00"),
        usd_per_mtok_output=Decimal("2.00"),
        usd_per_mtok_cached=Decimal("0.20"),
        supports_tools=True,
        supports_cache=supports_cache,
        requires_explicit_cache_breakpoint=requires_explicit_cache_breakpoint,
    )


@pytest.mark.sanity
def test_no_kestrel_md_yields_one_message_holding_the_prompt_exactly() -> None:
    """Given no KESTREL.md, when the prefix is built, then it is exactly
    one system message whose content equals the system prompt, with
    nothing folded in."""
    prefix = build_stable_prefix(_SYSTEM_PROMPT, None)

    assert len(prefix) == 1
    assert prefix[0] == {"role": "system", "content": _SYSTEM_PROMPT}


@pytest.mark.sanity
def test_kestrel_md_present_folds_prompt_banner_and_raw_text_in_order() -> None:
    """Given a loaded KESTREL.md, when the prefix is built, then its one
    message's content contains the system prompt, the fixed memory
    banner, and the file's raw text verbatim, each appearing after the
    one before it."""
    kestrel_md = KestrelMd(
        raw_text="# Conventions\n\nKeep it small.\n",
        verify_commands=VerifyCommands(),
    )

    prefix = build_stable_prefix(_SYSTEM_PROMPT, kestrel_md)

    assert len(prefix) == 1
    content = prefix[0]["content"]
    prompt_index = content.index(_SYSTEM_PROMPT)
    banner_index = content.index(_MEMORY_BANNER)
    raw_text_index = content.index(kestrel_md.raw_text)
    assert prompt_index < banner_index < raw_text_index


@pytest.mark.sanity
def test_repeated_calls_with_equal_arguments_are_byte_identical() -> None:
    """Given the same system prompt and an equal (but separately
    constructed) KESTREL.md, when the prefix is built twice, then the
    rendered content strings are equal by value, not merely by
    structural equality of the surrounding dataclasses."""
    kestrel_md_a = KestrelMd(
        raw_text="shared memory\n", verify_commands=VerifyCommands()
    )
    kestrel_md_b = KestrelMd(
        raw_text="shared memory\n", verify_commands=VerifyCommands()
    )

    first = build_stable_prefix(_SYSTEM_PROMPT, kestrel_md_a)
    second = build_stable_prefix(_SYSTEM_PROMPT, kestrel_md_b)

    assert first[0]["content"] == second[0]["content"]


def test_file_context_given_appends_a_second_message_holding_it_verbatim() -> None:
    """Given a `file_context` string, when the prefix is built, then it
    is exactly two messages, the second holding `file_context` verbatim
    and untouched."""
    file_context = "src/greet.py:\n  print('hi')\n"

    prefix = build_stable_prefix(_SYSTEM_PROMPT, None, file_context=file_context)

    assert len(prefix) == 2
    assert prefix[1] == {"role": "system", "content": file_context}


@pytest.mark.sanity
@pytest.mark.parametrize("backend", ["openrouter", "zai"])
def test_cache_capable_non_anthropic_backend_leaves_messages_unmarked(
    backend: str,
) -> None:
    """Given an entry for a backend that needs no explicit cache
    boundary marker -- every backend actually wired up today -- when
    breakpoints are marked, then no message carries the
    `cache_breakpoint` key at all, even though the entry itself
    supports caching."""
    entry = _entry(backend=backend, supports_cache=True)
    messages: list[Message] = [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "task"},
    ]

    marked = mark_cache_breakpoints(messages, entry)

    assert marked == messages
    for message in marked:
        assert "cache_breakpoint" not in message


def test_cache_capable_entry_with_explicit_marker_marks_only_the_last_message() -> None:
    """Given a synthetic entry for a backend that does need an explicit
    marker, with caching supported, when breakpoints are marked, then
    exactly the last message carries `cache_breakpoint=True` and every
    earlier message is left untouched."""
    entry = _entry(backend="anthropic", supports_cache=True, requires_explicit_cache_breakpoint=True)
    messages: list[Message] = [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "task"},
    ]

    marked = mark_cache_breakpoints(messages, entry)

    assert "cache_breakpoint" not in marked[0]
    assert marked[1]["cache_breakpoint"] is True
    assert marked[0] == messages[0]
    assert marked[1]["content"] == messages[1]["content"]


def test_entry_with_explicit_marker_without_cache_support_leaves_messages_unmarked() -> None:
    """Given a synthetic entry for the explicit-marker backend whose
    caching is disabled, when breakpoints are marked, then no message
    is annotated -- caching support gates the marker independently of
    the backend name."""
    entry = _entry(backend="anthropic", supports_cache=False, requires_explicit_cache_breakpoint=True)
    messages: list[Message] = [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "task"},
    ]

    marked = mark_cache_breakpoints(messages, entry)

    assert marked == messages
    for message in marked:
        assert "cache_breakpoint" not in message


@pytest.mark.regression
def test_stable_prefix_rendering_matches_golden_snapshot() -> None:
    """One canonical `build_stable_prefix` rendering, with a sample
    KESTREL.md folded in, normalized to sorted JSON, matches a pinned
    snapshot byte-for-byte -- an accidental change to the prefix's
    layout shows up as a diff here instead of silently regressing a
    cache-capable backend's hit rate."""
    kestrel_md = KestrelMd(
        raw_text=(
            "# Project conventions\n\nKeep changes small and covered by tests.\n"
        ),
        verify_commands=VerifyCommands(test="pytest -q"),
    )

    prefix = build_stable_prefix(
        "You are Kestrel, an autonomous coding agent.", kestrel_md
    )

    normalized = json.dumps(prefix, indent=2, sort_keys=True)
    assert normalized + "\n" == _GOLDEN_FILE.read_text()
