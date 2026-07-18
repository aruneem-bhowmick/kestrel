"""Tests for `kestrel.agent.plan`: parsing a finished task's own final
reply into a numbered `ImplementationPlan`, rendering that plan and a
reviewer's line comments back to text, and persisting a plan under
`.kestrel/artifacts/`.

`render_plan_comments`'s deliberate omission of `frame_untrusted` is
this module's own most security-relevant contract, so it gets its own
explicit assertion here (no delimiter marker anywhere in the output)
rather than relying on the absence being noticed incidentally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, LoopResult, TerminationReason
from kestrel.agent.plan import (
    ImplementationPlan,
    PlanComment,
    PlanError,
    PlanLine,
    extract_plan_from_result,
    parse_plan_lines,
    persist_plan,
    render_plan_comments,
    render_plan_markdown,
    revise_plan,
)
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StreamEvent, ToolCallEvent
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.corpus import InjectionCase, load_corpus

pytestmark = [pytest.mark.p048, pytest.mark.unit]


class _UnreachableClient:
    """A `ProviderClient` whose `complete` fails the test if ever
    invoked -- used to prove `revise_plan` rejects an empty `comments`
    sequence before it ever reaches `resume_task`."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Fail the test immediately -- this stand-in must never be
        called."""
        pytest.fail("revise_plan must not reach the client for empty comments")
        yield  # pragma: no cover -- unreachable; satisfies the generator protocol

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
_MARKDOWN_GOLDEN = _GOLDEN_DIR / "p048_plan_markdown.golden"
_COMMENTS_GOLDEN = _GOLDEN_DIR / "p048_plan_comments.golden"

_CORPUS_CASES = load_corpus()


def _plan_result(*, task_id: str = "task-1") -> LoopResult:
    """A minimal `LoopResult` ending on a plain assistant reply, for
    tests that only care about `history[-1]`."""
    return LoopResult(
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0"),
        history=({"role": "assistant", "content": "1. do X"},),
    )


# -- parse_plan_lines ---------------------------------------------------


@pytest.mark.sanity
def test_three_non_blank_lines_become_three_numbered_plan_lines() -> None:
    """Given three non-blank lines, when parsed, then three `PlanLine`s
    are returned, numbered 1, 2, and 3 in order of appearance."""
    result = parse_plan_lines("first step\nsecond step\nthird step")

    assert result == (
        PlanLine(index=1, text="first step"),
        PlanLine(index=2, text="second step"),
        PlanLine(index=3, text="third step"),
    )


@pytest.mark.sanity
def test_blank_lines_are_dropped_and_remaining_lines_stay_contiguous() -> None:
    """Given blank lines interleaved between real content, when parsed,
    then the blank lines are dropped entirely -- not counted -- and the
    remaining lines are still numbered contiguously from 1."""
    result = parse_plan_lines("first step\n\n\nsecond step\n   \nthird step")

    assert [line.index for line in result] == [1, 2, 3]
    assert [line.text for line in result] == ["first step", "second step", "third step"]


@pytest.mark.sanity
def test_surrounding_whitespace_on_a_line_is_stripped() -> None:
    """Given a line padded with leading and trailing whitespace, when
    parsed, then the resulting `PlanLine.text` has that whitespace
    stripped."""
    result = parse_plan_lines("   indented step   \n\tsecond, tab-indented\t")

    assert result[0].text == "indented step"
    assert result[1].text == "second, tab-indented"


@pytest.mark.sanity
def test_empty_string_parses_to_no_lines() -> None:
    """Given an empty string, when parsed, then no `PlanLine`s are
    returned."""
    assert parse_plan_lines("") == ()


# -- extract_plan_from_result --------------------------------------------


def test_extract_plan_from_result_parses_the_last_assistant_message() -> None:
    """Given a `LoopResult` whose last message is a plain assistant reply,
    when extracted, then the returned `ImplementationPlan` carries the
    given task id, the reply's raw text verbatim, and its parsed
    lines."""
    result = _plan_result(task_id="task-42")

    plan = extract_plan_from_result(result, task_id="task-42")

    assert plan == ImplementationPlan(
        task_id="task-42",
        raw_text="1. do X",
        lines=(PlanLine(index=1, text="1. do X"),),
    )


@pytest.mark.sanity
def test_extract_plan_from_result_raises_when_the_last_message_carries_tool_calls() -> (
    None
):
    """Given a `LoopResult` whose last message carries `tool_calls` (the
    task ended mid tool-call, e.g. at `TURN_CAP`), when extracted, then
    `PlanError` is raised -- there is no plan text to parse."""
    result = LoopResult(
        reason=TerminationReason.TURN_CAP,
        turns_used=1,
        total_usd=Decimal("0"),
        history=(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")
                ],
            },
        ),
    )

    with pytest.raises(PlanError, match="task-mid-call"):
        extract_plan_from_result(result, task_id="task-mid-call")


@pytest.mark.sanity
def test_extract_plan_from_result_raises_on_empty_history() -> None:
    """Given a `LoopResult` with no history at all, when extracted, then
    `PlanError` is raised."""
    result = LoopResult(
        reason=TerminationReason.USER_STOP,
        turns_used=0,
        total_usd=Decimal("0"),
        history=(),
    )

    with pytest.raises(PlanError, match="task-empty"):
        extract_plan_from_result(result, task_id="task-empty")


def test_extract_plan_from_result_raises_when_the_last_message_is_not_assistant() -> (
    None
):
    """Given a `LoopResult` whose last message is a tool-role result (not
    a plain assistant reply), when extracted, then `PlanError` is
    raised."""
    result = LoopResult(
        reason=TerminationReason.TURN_CAP,
        turns_used=1,
        total_usd=Decimal("0"),
        history=({"role": "tool", "content": "ran read_file", "tool_call_id": "call-1"},),
    )

    with pytest.raises(PlanError):
        extract_plan_from_result(result, task_id="task-tool-last")


# -- render_plan_markdown -------------------------------------------------


def test_render_plan_markdown_matches_golden_snapshot() -> None:
    """The exact byte output of one canonical three-line plan's own
    `render_plan_markdown` call must match a pinned snapshot."""
    plan = ImplementationPlan(
        task_id="demo-task",
        raw_text="raw",
        lines=(
            PlanLine(index=1, text="Read the existing config parser."),
            PlanLine(index=2, text="Add a new field for the retry timeout."),
            PlanLine(index=3, text="Write a regression test for the new field."),
        ),
    )

    rendered = render_plan_markdown(plan)

    assert rendered == _MARKDOWN_GOLDEN.read_text(encoding="utf-8")


# -- render_plan_comments --------------------------------------------------


def test_render_plan_comments_contains_the_raw_comment_with_no_untrusted_marker() -> (
    None
):
    """Given one comment, when rendered, then the returned text contains
    the raw comment string verbatim and carries no `<<<UNTRUSTED`
    marker anywhere -- comments are deliberately never framed."""
    comments = (
        PlanComment(
            line_index=2,
            line_text="Add a new field.",
            comment="Also validate the field's range.",
        ),
    )

    rendered = render_plan_comments(comments)

    assert "Also validate the field's range." in rendered
    assert "<<<UNTRUSTED" not in rendered
    assert "<<<END_UNTRUSTED>>>" not in rendered


def test_render_plan_comments_matches_golden_snapshot() -> None:
    """The exact byte output of two canonical comments' own
    `render_plan_comments` call must match a pinned snapshot, and
    preserves the given comments in their input order."""
    comments = (
        PlanComment(
            line_index=1,
            line_text="Read the existing config parser.",
            comment="Also check for a legacy config path.",
        ),
        PlanComment(
            line_index=3,
            line_text="Write a regression test for the new field.",
            comment="Cover the legacy-path case too.",
        ),
    )

    rendered = render_plan_comments(comments)

    assert rendered + "\n" == _COMMENTS_GOLDEN.read_text(encoding="utf-8")
    first_index = rendered.index("Also check for a legacy config path.")
    second_index = rendered.index("Cover the legacy-path case too.")
    assert first_index < second_index


# -- persist_plan -----------------------------------------------------------


def _demo_plan(task_id: str = "task-persist") -> ImplementationPlan:
    """A minimal one-line `ImplementationPlan` for persistence tests."""
    return ImplementationPlan(
        task_id=task_id,
        raw_text="1. do the thing",
        lines=(PlanLine(index=1, text="1. do the thing"),),
    )


def test_persist_plan_writes_the_expected_path_and_content(tmp_path: Path) -> None:
    """Given a plan, when persisted, then the exact rendered markdown is
    written to `.kestrel/artifacts/plan-<task_id>.md`."""
    plan = _demo_plan()

    written = persist_plan(plan, repo_root=tmp_path)

    expected_path = tmp_path / ".kestrel" / "artifacts" / "plan-task-persist.md"
    assert written == expected_path
    assert written.read_text(encoding="utf-8") == render_plan_markdown(plan)


def test_persist_plan_a_second_call_gets_the_numeric_suffix_path(tmp_path: Path) -> None:
    """Given two plans persisted for the same task id, when the second is
    written, then it lands at the `-1`-suffixed path rather than
    overwriting the first -- matching `persist_verification_report`'s
    own collision behavior."""
    plan = _demo_plan()

    first_path = persist_plan(plan, repo_root=tmp_path)
    second_path = persist_plan(plan, repo_root=tmp_path)

    assert first_path != second_path
    assert second_path.name == "plan-task-persist-1.md"
    assert first_path.is_file()
    assert second_path.is_file()


def test_persist_plan_refuses_a_symlinked_artifacts_dir_naming_the_remedy(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Given `.kestrel` replaced with a symlink pointing outside the repo
    root, when a plan is persisted, then `PlanError` names the escape
    and nothing is written through the symlink."""
    outside = tmp_path_factory.mktemp("plan-outside-target")
    (tmp_path / ".kestrel").symlink_to(outside, target_is_directory=True)
    plan = _demo_plan()

    with pytest.raises(PlanError, match="escapes the repository root"):
        persist_plan(plan, repo_root=tmp_path)

    assert list(outside.iterdir()) == []


# -- revise_plan --------------------------------------------------------


def _registry() -> Registry:
    """A single-entry `Registry` matching this suite's `_UnreachableClient`."""
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={"glm-5.2": entry}, source=None)


async def test_revise_plan_rejects_empty_comments_before_touching_the_client(
    tmp_path: Path,
) -> None:
    """Given an empty `comments` sequence, when `revise_plan` is called,
    then `ValueError` is raised and the client is never reached -- there
    is nothing to inject, so no new turn should be driven."""
    registry = _registry()
    deps = LoopDeps(
        client=_UnreachableClient(),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await revise_plan("task-empty-comments", deps, [])


# -- Red-team: hostile plan text must never raise --------------------------


@pytest.mark.redteam
@pytest.mark.parametrize("case", _CORPUS_CASES, ids=lambda case: case.id)
def test_hostile_plan_text_parses_without_raising(case: InjectionCase) -> None:
    """Given every case in the injection corpus, used as a task's own
    raw plan text, when parsed via `parse_plan_lines` and
    `extract_plan_from_result`, then neither raises -- rendering that
    text safely for on-screen display is a separate concern from this
    module's own pure parsing."""
    parse_plan_lines(case.payload)

    result = LoopResult(
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0"),
        history=({"role": "assistant", "content": case.payload},),
    )
    extract_plan_from_result(result, task_id=f"redteam-{case.id}")
