"""Routes the agent loop's self-critique phase to a real, cheap-tagged
model call.

`LoopDeps.self_critique_fn` starts out as an always-approve no-op (see
`kestrel.agent.loop._default_self_critique`) so the loop's own control
flow never depends on a live model call existing. `make_self_critique_fn`
builds the real replacement this module provides: one short, non-streamed
completion asking whether a turn's proposed action looks reasonable,
parsed down to a plain `bool`. It never opens a multi-turn conversation
and never accepts a configurable prompt.

A critique call goes through the exact same `ProviderClient` a task's own
turns do, so it is billed by the backend like any other request -- never
a free, simulated side channel. It is deliberately *not* folded into
`deps.meter`'s running total or `deps.session`'s own turn-by-turn journal,
though: neither collaborator reaches this module, since
`LoopDeps.self_critique_fn`'s own call site (`agent/loop.py`) passes it
only a turn's proposal and history, nothing else. A resumed task's
budget/cache-hit/cost-meter figures therefore reflect its real turns
only, not the (small, capped) critique spend alongside them.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Final

from kestrel.provider.base import Message, ProviderClient
from kestrel.provider.events import StreamEvent, TextDelta
from kestrel.provider.retry import complete_with_retry

_CRITIQUE_SYSTEM_PROMPT: Final[str] = (
    "You are a fast sanity-checker for an autonomous coding agent's "
    "next proposed action. Given the conversation so far and one "
    "proposed action, reply with exactly one word: APPROVE if the "
    "action is reasonable, in scope, and not needlessly destructive; "
    "REJECT otherwise. No other text."
)
_CRITIQUE_MAX_TOKENS: Final[int] = 16


def _parses_as_approve(text: str) -> bool:
    """`True` unless `text`'s own stripped, uppercased form starts with
    `"REJECT"` -- fails open (approves) on an empty, garbled, or
    otherwise unparseable reply, matching `_default_self_critique`'s own
    always-approve stance for anything this function cannot confidently
    read as a rejection."""
    return not text.strip().upper().startswith("REJECT")


async def _critique_async(
    proposal: str, history: list[Message], *, client: ProviderClient, model_id: str
) -> bool:
    """Send one short, non-streamed completion asking whether `proposal`
    looks reasonable given `history`, and parse the reply via
    `_parses_as_approve`. Offers no tools (`tools=None`) and
    `stream=False` -- a normalized single `TextDelta` is guaranteed by
    `ProviderClient`'s own contract regardless of backend."""
    messages: list[Message] = [
        {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
        *history[-4:],
        {"role": "user", "content": f"Proposed action:\n{proposal}"},
    ]
    events: list[StreamEvent] = []
    async for event in complete_with_retry(
        client,
        messages,
        None,
        model_id,
        "high",
        stream=False,
        max_tokens=_CRITIQUE_MAX_TOKENS,
    ):
        events.append(event)
    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    return _parses_as_approve(text)


def model_self_critique(
    proposal: str, history: list[Message], *, client: ProviderClient, model_id: str
) -> bool:
    """Runs synchronously (see `agent/loop.py`'s `asyncio.to_thread`
    wrapping of `LoopDeps.self_critique_fn`) -- safe to open its own
    event loop via `asyncio.run` here specifically because this always
    executes on a dedicated `asyncio.to_thread` worker thread, never the
    loop driving the task itself."""
    return asyncio.run(
        _critique_async(proposal, history, client=client, model_id=model_id)
    )


def make_self_critique_fn(
    *, client: ProviderClient, model_id: str
) -> Callable[[str, list[Message]], bool]:
    """Bind `client`/`model_id` into `model_self_critique`, ready to
    assign as `LoopDeps.self_critique_fn`."""
    return functools.partial(model_self_critique, client=client, model_id=model_id)
