"""Builds the fixed-order leading messages sent at the start of every
model call for one task or session.

A cache-capable backend can only reuse its cached compute across turns
when the leading portion of a request is byte-for-byte identical to a
prior request -- any difference in that shared prefix, down to a single
byte, forces the backend to reprocess it from scratch. This module is
the one place that leading prefix gets assembled, so every caller (the
tool-calling agent loop and the plain REPL) builds it the same way and
never accidentally reformats it turn to turn.

``mark_cache_breakpoints`` exists alongside the builder as a dormant
extension point: no backend wired up today needs an explicit marker
telling it where the cacheable prefix ends, so it is currently a no-op
for every real registry entry. A backend that does need one reads
``Message.cache_breakpoint`` off the last message this function
annotates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from kestrel.kestrel_md import KestrelMd
from kestrel.provider.base import Message
from kestrel.registry.model import ModelEntry

_MEMORY_BANNER: Final[str] = "Project memory (KESTREL.md):"

# Backends whose wire protocol accepts an explicit cache-boundary marker.
# Empty in effect for every backend actually wired up today (see this
# module's docstring); adding a name here without a matching adapter
# change is inert, since nothing yet reads the marker back out.
_EXPLICIT_BREAKPOINT_BACKENDS: Final[frozenset[str]] = frozenset({"anthropic"})


def build_stable_prefix(
    system_prompt: str,
    kestrel_md: KestrelMd | None,
    *,
    file_context: str | None = None,
) -> list[Message]:
    """Build the fixed-order leading messages a model call sends first.

    Returns one system-role message holding `system_prompt` verbatim
    when `kestrel_md` is `None`. When `kestrel_md` is given, that same
    message instead holds `system_prompt`, a blank line, a fixed
    project-memory banner, a blank line, and `kestrel_md.raw_text`
    verbatim, joined with newlines -- the same string, byte for byte,
    for a given `(system_prompt, kestrel_md)` pair on every call, which
    is what lets a cache-capable backend actually reuse its cached
    compute across turns.

    `file_context`, when given, becomes a second system-role message
    appended after the first, holding it verbatim. No caller populates
    this yet -- it exists now so wiring in a stable file-context
    producer later only needs a new argument, not a signature change.
    """
    if kestrel_md is None:
        prefix_content = system_prompt
    else:
        prefix_content = "\n".join(
            (system_prompt, "", _MEMORY_BANNER, "", kestrel_md.raw_text)
        )

    messages: list[Message] = [{"role": "system", "content": prefix_content}]
    if file_context is not None:
        messages.append({"role": "system", "content": file_context})
    return messages


def mark_cache_breakpoints(
    messages: Sequence[Message], entry: ModelEntry
) -> list[Message]:
    """Annotate the last of `messages` with an explicit cache breakpoint
    when `entry`'s backend needs one; otherwise return `messages`
    unchanged.

    Sets `cache_breakpoint=True` on exactly the last message when
    `entry.backend` is in the small set of backends whose wire protocol
    requires an explicit marker for where the cacheable prefix ends,
    and `entry.supports_cache` is also `True`. In every other case --
    including every backend actually wired up today -- this returns a
    new list with the same messages, none of them touched, since an
    implicit, position-based cache boundary is all any of them need.
    """
    if entry.backend not in _EXPLICIT_BREAKPOINT_BACKENDS or not entry.supports_cache:
        return list(messages)
    if not messages:
        return list(messages)

    marked: list[Message] = list(messages[:-1])
    marked.append({**messages[-1], "cache_breakpoint": True})
    return marked
