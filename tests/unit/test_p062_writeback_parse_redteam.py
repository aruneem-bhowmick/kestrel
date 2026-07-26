"""Red-team proof that `_parse_proposal_line` cannot be made to crash --
or, transitively, that `propose_learnings` cannot be made to raise
anything but its own documented `WritebackError` -- by a hostile model
reply.

A writeback proposal is untrusted, model-generated text parsed as data:
every payload in the checked-in injection corpus stands in for one raw
reply line here, matching `test_p047_critique.py`'s own
`test_hostile_critique_reply_never_crashes_parsing` precedent for a
different narrow-parse call site. Unlike self-critique's own fail-*open*
stance, this suite's correct outcome is fail-*closed*: a hostile payload
degrading to "nothing parsed" is safe here specifically because nothing
`_parse_proposal_line` ever produces is committed to the knowledge base
without a human separately approving it downstream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from decimal import Decimal

import pytest

from kestrel.agent.loop import TerminationReason
from kestrel.agent.walkthrough import Walkthrough
from kestrel.kb.writeback import WritebackError, _parse_proposal_line, propose_learnings
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StopEvent, StreamEvent, TextDelta, UsageEvent
from kestrel.security.corpus import load_corpus

pytestmark = [pytest.mark.p062, pytest.mark.unit, pytest.mark.redteam]

_MODEL_ID = "trivial-model"


def _walkthrough() -> Walkthrough:
    """A minimal, otherwise-unused `Walkthrough` -- only `propose_
    learnings`'s own rendering of it matters here, not its content."""
    return Walkthrough(
        task_id="task-1",
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0"),
        touched_paths=(),
        verification=None,
    )


@dataclass
class _HostileReplyClient:
    """A `ProviderClient` that always answers its one expected call with
    a fixed, possibly-hostile reply text."""

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
        usage and stop events."""
        yield TextDelta(text=self.text)
        yield UsageEvent(input_tokens=10, output_tokens=5, cached_tokens=0)
        yield StopEvent(reason="end_turn")


def test_every_corpus_payload_parses_to_none_or_a_harmless_learning() -> None:
    """Given every payload in the injection corpus, each fed as a single
    proposal reply line, when `_parse_proposal_line` parses it, then it
    never raises -- and, since no corpus payload is crafted to match the
    documented `LEARNING: ... | TAGS: ...` shape, every one of them
    parses to `None`."""
    for case in load_corpus():
        assert _parse_proposal_line(case.payload) is None


async def test_every_corpus_payload_as_a_full_reply_never_crashes_propose_learnings() -> (
    None
):
    """Given every payload in the injection corpus, each standing in for
    a whole hostile model reply (not just one line of it), when
    `propose_learnings` runs against a client scripted to return it, then
    the call never raises anything other than the documented
    `WritebackError` -- and since this scripted client never itself
    fails, it does not raise even that, always returning cleanly instead
    (possibly with an empty tuple, when the payload never matches the
    documented line format)."""
    for case in load_corpus():
        client = _HostileReplyClient(text=case.payload)
        try:
            result = await propose_learnings(
                _walkthrough(), client=client, model_id=_MODEL_ID
            )
        except WritebackError:
            continue
        assert isinstance(result, tuple)
        assert len(result) <= 3
