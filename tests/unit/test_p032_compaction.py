"""Unit tests for `kestrel.agent.compaction`'s own two building blocks:
`should_compact`'s ratio-vs-threshold arithmetic, and `compact_history`'s
folding behavior -- when it makes no model call at all, what it sends
when it does, and exactly what it hands back to its caller.

Reuses the scripted-`ProviderClient` pattern already established by the
agent-loop test suites (a fake object satisfying the `ProviderClient`
protocol structurally, recording what it was called with) rather than a
live mock server, since nothing here needs a real network seam --
`tests/system/test_p032_compaction_scripted.py` owns that coverage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

import pytest

from kestrel.agent.compaction import (
    _COMPACTION_SYSTEM_PROMPT,
    _DEFAULT_KEEP_LAST_N,
    compact_history,
    should_compact,
)
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StopEvent, StreamEvent, TextDelta, UsageEvent
from kestrel.tools.verify import VerificationCommandResult, VerificationReport

pytestmark = [pytest.mark.p032, pytest.mark.unit]

_MODEL_ID = "glm-5.2"


@dataclass
class _ScriptedSummaryClient:
    """A `ProviderClient` that always replies with one fixed text-only
    turn, recording every call's own `messages` argument -- enough for
    this suite to assert both how many times `compact_history` called
    the client and exactly what it sent when it did."""

    reply_text: str = "carry-forward summary"
    input_tokens: int = 42
    output_tokens: int = 7
    call_count: int = field(default=0, init=False)
    received_messages: list[list[Message]] = field(default_factory=list, init=False)
    received_efforts: list[Effort] = field(default_factory=list, init=False)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Record this call's own messages and effort, then yield one
        plain text-only reply -- no tool calls, matching the contract
        `compact_history` relies on."""
        self.call_count += 1
        self.received_messages.append(list(messages))
        self.received_efforts.append(effort)
        yield TextDelta(text=self.reply_text)
        yield UsageEvent(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=0,
        )
        yield StopEvent(reason="end_turn")


def _message(
    role: Literal["system", "user", "assistant", "tool"], content: str
) -> Message:
    """Build one plain `Message` with no optional fields set."""
    return {"role": role, "content": content}


def _history(n: int) -> list[Message]:
    """`n` plain user/assistant messages alternating, numbered in their
    own content so a test can tell which ones a fold kept or dropped."""
    return [
        _message("user" if i % 2 == 0 else "assistant", f"message-{i}")
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("last_input_tokens", "context_window", "expected"),
    [
        pytest.param(70, 100, True, id="exactly-at-threshold"),
        pytest.param(69, 100, False, id="just-below-threshold"),
        pytest.param(71, 100, True, id="just-above-threshold"),
    ],
)
def test_should_compact_at_the_threshold_boundary(
    last_input_tokens: int, context_window: int, expected: bool
) -> None:
    """Given a context window of 100, when the last turn's own input
    tokens sit at exactly 70, just below it, or just above it, then
    `should_compact` is true starting exactly at the boundary (`>=`,
    not `>`)."""
    assert should_compact(last_input_tokens, context_window) is expected


@pytest.mark.sanity
@pytest.mark.parametrize("context_window", [0, -1])
def test_should_compact_never_compacts_a_non_positive_context_window(
    context_window: int,
) -> None:
    """Given a zero or negative `context_window`, when checked against
    any input token count, then `should_compact` is always false --
    defensive only, since a real `ModelEntry` can never carry one."""
    assert should_compact(1_000_000, context_window) is False


async def test_compact_history_with_short_history_returns_it_unchanged() -> None:
    """Given a history no longer than `keep_last_n`, when
    `compact_history` runs, then it returns the same messages unchanged
    and a zeroed `UsageEvent`, and never calls the client at all."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N)

    result_history, usage = await compact_history(
        client, _MODEL_ID, history, last_verification=None
    )

    assert result_history == history
    assert usage == UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0)
    assert client.call_count == 0


async def test_compact_history_calls_the_client_once_with_the_older_tail() -> None:
    """Given a history longer than `keep_last_n`, when `compact_history`
    runs, then the client is called exactly once, and the messages it
    received are the compaction system prompt followed by exactly
    `history[:-keep_last_n]`, verbatim."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N + 3)

    await compact_history(client, _MODEL_ID, history, last_verification=None)

    assert client.call_count == 1
    (sent,) = client.received_messages
    assert sent[0] == {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT}
    assert sent[1:] == history[:-_DEFAULT_KEEP_LAST_N]


@pytest.mark.parametrize("effort", ["high", "max"])
async def test_compact_history_sends_the_given_effort(effort: Effort) -> None:
    """Given an explicit `effort`, when `compact_history` calls the
    client, then that exact value is what the client receives -- not
    the `"high"` literal every call used before this parameter existed."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N + 3)

    await compact_history(
        client, _MODEL_ID, history, last_verification=None, effort=effort
    )

    assert client.received_efforts == [effort]


async def test_compact_history_defaults_to_high_effort() -> None:
    """Given no explicit `effort`, when `compact_history` calls the
    client, then it defaults to `"high"` -- identical to every caller
    written before this parameter existed."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N + 3)

    await compact_history(client, _MODEL_ID, history, last_verification=None)

    assert client.received_efforts == ["high"]


async def test_summary_message_names_a_failing_verifications_failing_commands() -> None:
    """Given a failing `last_verification`, when `compact_history` folds
    history, then the returned summary message's content names the
    FAILED status and the exact, verbatim name of every command that
    failed."""
    client = _ScriptedSummaryClient(reply_text="work remains on the greeter")
    history = _history(_DEFAULT_KEEP_LAST_N + 1)
    report = VerificationReport(
        task_id="t-1",
        turn_id=3,
        commands=(
            VerificationCommandResult(
                name="lint",
                command="ruff check .",
                exit_code=1,
                timed_out=False,
                stdout="",
                stderr="E501",
            ),
            VerificationCommandResult(
                name="test",
                command="pytest",
                exit_code=0,
                timed_out=False,
                stdout="",
                stderr="",
            ),
        ),
        passed=False,
    )

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=report
    )

    summary_content = result_history[0]["content"]
    assert "work remains on the greeter" in summary_content
    assert "FAILED" in summary_content
    assert "lint" in summary_content
    assert "test" not in summary_content.split("failing:")[1]


async def test_summary_message_names_a_passing_verification() -> None:
    """Given a passing `last_verification`, when `compact_history` folds
    history, then the returned summary message's content names the
    PASSED status."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N + 1)
    report = VerificationReport(task_id="t-2", turn_id=1, commands=(), passed=True)

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=report
    )

    assert "PASSED" in result_history[0]["content"]


async def test_summary_message_omits_the_verification_block_when_none_is_given() -> (
    None
):
    """Given `last_verification=None`, when `compact_history` folds
    history, then the returned summary message's content is exactly the
    model's own rendered text, with no verification block -- and no
    literal `"None"` string -- appended."""
    client = _ScriptedSummaryClient(reply_text="only the model's own words")
    history = _history(_DEFAULT_KEEP_LAST_N + 1)

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=None
    )

    assert result_history[0]["content"] == "only the model's own words"
    assert "None" not in result_history[0]["content"]


async def test_returned_history_is_exactly_the_summary_plus_the_kept_tail() -> None:
    """Given a history longer than `keep_last_n`, when `compact_history`
    folds it, then the returned history is exactly `[summary_message,
    *history[-keep_last_n:]]`, verbatim and in that order -- no
    reordering, and no message from the kept tail dropped or
    duplicated."""
    client = _ScriptedSummaryClient()
    history = _history(_DEFAULT_KEEP_LAST_N + 5)

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=None
    )

    assert len(result_history) == _DEFAULT_KEEP_LAST_N + 1
    assert result_history[0]["role"] == "system"
    assert result_history[1:] == history[-_DEFAULT_KEEP_LAST_N:]


async def test_compact_history_honors_an_explicit_keep_last_n() -> None:
    """Given an explicit `keep_last_n` narrower than the default, when
    `compact_history` folds a history sized right at that boundary,
    then exactly that many trailing messages survive verbatim and
    everything else is folded into the summary."""
    client = _ScriptedSummaryClient()
    history = _history(5)

    result_history, _usage = await compact_history(
        client, _MODEL_ID, history, last_verification=None, keep_last_n=2
    )

    assert len(result_history) == 3
    assert result_history[1:] == history[-2:]
    (sent,) = client.received_messages
    assert sent[1:] == history[:3]


def test_registered_threshold_and_keep_last_n_match_the_documented_defaults() -> None:
    """Given the module's own private constants, when read directly,
    then the threshold is 70% and the default kept tail is four
    messages -- pinning the documented defaults against an accidental
    change to either."""
    from kestrel.agent.compaction import _COMPACTION_THRESHOLD

    assert _COMPACTION_THRESHOLD == Decimal("0.70")
    assert _DEFAULT_KEEP_LAST_N == 4
