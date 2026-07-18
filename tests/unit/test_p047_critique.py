"""Unit tests for `kestrel.agent.critique`: the routed self-critique
implementation that replaces `agent.loop`'s own always-approve default.

Covers `_parses_as_approve`'s APPROVE/REJECT parsing (including every
adversarial payload in the injection corpus, standing in for a hostile
critique-model reply) and `model_self_critique` exercised the same way
`agent/loop.py`'s own `_drive` calls it: via `asyncio.to_thread`, against
a scripted fake `ProviderClient`, never a real network call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from kestrel.agent.critique import _parses_as_approve, model_self_critique
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StopEvent, StreamEvent, TextDelta, UsageEvent
from kestrel.security.corpus import load_corpus

pytestmark = [pytest.mark.p047, pytest.mark.unit]

_MODEL_ID = "cheap-model"


@dataclass
class _ScriptedCritiqueClient:
    """A `ProviderClient` that always answers its one expected
    non-streamed call with a fixed reply text, ignoring every other
    argument -- enough to drive `model_self_critique` without a real
    network call."""

    text: str

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield one `TextDelta` carrying `self.text`, then the closing
        usage and stop events -- the normalized grammar every backend
        must satisfy regardless of `stream`."""
        yield TextDelta(text=self.text)
        yield UsageEvent(input_tokens=30, output_tokens=2, cached_tokens=0)
        yield StopEvent(reason="end_turn")


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROVE", True),
        ("approve, looks fine", True),
        ("REJECT", False),
        ("Reject: too destructive", False),
        ("", True),
        ("maybe?", True),
    ],
)
def test_parses_as_approve(text: str, expected: bool) -> None:
    """Given each of six representative reply shapes, when
    `_parses_as_approve` parses them, then only a reply whose stripped,
    uppercased form starts with `"REJECT"` reads as a decline -- every
    other shape, including an empty or unparseable one, fails open."""
    assert _parses_as_approve(text) is expected


async def test_model_self_critique_approves_via_a_real_background_thread() -> None:
    """Given a scripted client replying `"APPROVE"`, when
    `model_self_critique` runs the way the loop actually calls it -- via
    `asyncio.to_thread` from an async test function, exercising the real
    threading path rather than a direct call -- then it returns `True`.
    """
    client = _ScriptedCritiqueClient(text="APPROVE")

    result = await asyncio.to_thread(
        model_self_critique,
        "run `rm -rf build/`",
        [],
        client=client,
        model_id=_MODEL_ID,
    )

    assert result is True


async def test_model_self_critique_rejects_via_a_real_background_thread() -> None:
    """Given a scripted client replying `"REJECT"`, when
    `model_self_critique` runs via `asyncio.to_thread`, then it returns
    `False`."""
    client = _ScriptedCritiqueClient(text="REJECT")

    result = await asyncio.to_thread(
        model_self_critique,
        "run `rm -rf build/`",
        [],
        client=client,
        model_id=_MODEL_ID,
    )

    assert result is False


@pytest.mark.redteam
def test_hostile_critique_reply_never_crashes_parsing() -> None:
    """Given every adversarial payload in the checked-in injection
    corpus, as if each were the critique model's own reply text, when
    `_parses_as_approve` parses it, then it never raises and always
    returns a plain `bool` -- the token path a hostile completion could
    reach is exactly this one function, and it must fail open, not
    fail loud."""
    for case in load_corpus():
        result = _parses_as_approve(case.payload)
        assert isinstance(result, bool)
