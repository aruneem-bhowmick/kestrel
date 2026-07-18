"""Turns a finished PLAN-mode task into a persisted, line-addressable
`ImplementationPlan` artifact, renders it back to markdown, and turns a
set of user-authored line comments into one new injectable message.

This is a pure data layer: nothing here talks to a live model, a CLI
flag, or a Textual widget. `extract_plan_from_result` reads a completed
`LoopResult`'s own final assistant message and parses it into numbered
`PlanLine`s; `render_plan_markdown`/`persist_plan` turn that back into
markdown and write it under `.kestrel/artifacts/`; `render_plan_comments`/
`revise_plan` turn a reviewer's line comments into a new turn on the same
underlying task, via `kestrel.agent.loop.resume_task`'s own
`inject_message` parameter.

Unlike every other content path in this codebase, `render_plan_comments`
never passes its output through `kestrel.security.framing.frame_untrusted`.
A plan comment is typed by the user reviewing the plan, not read from a
file, a tool's stdout, or any other external source -- it carries the
same trust level as a task description typed directly into the prompt,
per `kestrel.security.framing`'s own "only non-user-typed content is
untrusted" rule. This is a deliberate omission, not an oversight.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from kestrel.agent._artifact_paths import persist_markdown_artifact
from kestrel.agent.loop import LoopDeps, LoopResult, resume_task


@dataclass(frozen=True, slots=True)
class PlanLine:
    """One numbered line of an implementation plan.

    Attributes:
        index: 1-based position within the plan, stable for as long as
            the plan is not itself re-parsed from a revised reply.
        text: The line's own text, stripped of surrounding whitespace.
    """

    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ImplementationPlan:
    """One PLAN-mode task's own parsed, persisted plan.

    Attributes:
        task_id: The PLAN-mode task this plan was parsed from.
        raw_text: The model's own final message, verbatim.
        lines: `raw_text` split into `PlanLine`s via `parse_plan_lines`.
    """

    task_id: str
    raw_text: str
    lines: tuple[PlanLine, ...]


class PlanError(Exception):
    """`extract_plan_from_result` found no plan text to parse, or
    `persist_plan` could not write one. `str(self)` names the remedy."""


def parse_plan_lines(raw_text: str) -> tuple[PlanLine, ...]:
    """Every non-blank line of `raw_text`, stripped, numbered 1-based in
    order of appearance -- deliberately not markdown-aware (no special
    handling of `-`/`1.` list markers): a PLAN-mode reply is expected to
    already be one step per line, and a naive split is enough for
    "name a line by number" need. A blank line is dropped entirely, not
    counted -- line numbers index non-blank content only."""
    return tuple(
        PlanLine(index=i, text=stripped)
        for i, stripped in enumerate(
            (line.strip() for line in raw_text.splitlines() if line.strip()),
            start=1,
        )
    )


def extract_plan_from_result(result: LoopResult, *, task_id: str) -> ImplementationPlan:
    """Parse `result.history`'s own last message into an
    `ImplementationPlan`.

    Raises:
        PlanError: `result.history` is empty, or its last message is
            not a plain assistant message with no `tool_calls` (e.g.
            the task ended `TURN_CAP` mid tool-call) -- there is no
            plan text to parse in that case.
    """
    if not result.history:
        raise PlanError(f"task {task_id!r} produced no history to parse a plan from")
    last = result.history[-1]
    if last["role"] != "assistant" or "tool_calls" in last:
        raise PlanError(
            f"task {task_id!r} did not end on a plain assistant message "
            "-- no plan text to parse"
        )
    raw_text = last["content"]
    return ImplementationPlan(
        task_id=task_id, raw_text=raw_text, lines=parse_plan_lines(raw_text)
    )


def render_plan_markdown(plan: ImplementationPlan) -> str:
    """`# Implementation Plan: {task_id}`, then one `N. {text}` line per
    `plan.lines` entry, in order."""
    lines = [f"# Implementation Plan: {plan.task_id}", ""]
    lines.extend(f"{line.index}. {line.text}" for line in plan.lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def persist_plan(plan: ImplementationPlan, *, repo_root: Path) -> Path:
    """Persist `render_plan_markdown(plan)` via
    `persist_markdown_artifact`, stem `f"plan-{plan.task_id}"`.

    Raises:
        PlanError: see `persist_markdown_artifact`'s own `ValueError`.
    """
    try:
        return persist_markdown_artifact(
            render_plan_markdown(plan), repo_root=repo_root, stem=f"plan-{plan.task_id}"
        )
    except ValueError as exc:
        raise PlanError(
            f".kestrel/artifacts: refusing to persist plan-{plan.task_id} ({exc})"
        ) from exc


@dataclass(frozen=True, slots=True)
class PlanComment:
    """One user comment attached to one plan line.

    Attributes:
        line_index: The commented `PlanLine.index`.
        line_text: That line's own text, captured at comment time so a
            later re-render still shows what the comment was actually
            about even if the plan itself has since changed.
        comment: The user's own comment text, verbatim.
    """

    line_index: int
    line_text: str
    comment: str


def render_plan_comments(comments: Sequence[PlanComment]) -> str:
    """Render `comments` as one new user-role message body asking the
    model to revise the plan -- plain, trusted, user-authored text,
    never passed through `frame_untrusted` (comments are direct user
    input, the same trust level as a typed task description, per
    `kestrel.security.framing`'s own "only non-user-typed content is
    untrusted" rule)."""
    lines = [
        "The user has reviewed your implementation plan and left the "
        "following comments. Revise the plan to address them, then "
        "reply with the complete revised plan, one step per line.",
        "",
    ]
    lines.extend(f'- Line {c.line_index} ("{c.line_text}"): {c.comment}' for c in comments)
    return "\n".join(lines)


async def revise_plan(
    task_id: str,
    deps: LoopDeps,
    comments: Sequence[PlanComment],
    *,
    clock_fn: Callable[[], float] = time.monotonic,
) -> LoopResult:
    """Continue `task_id`'s own PLAN-mode conversation with `comments`
    injected as a new user turn, via a thin, named wrapper around
    `resume_task(..., inject_message=render_plan_comments(comments))`
    for exactly this call shape, so a caller does not need to know
    `render_plan_comments` exists to use it.

    Raises:
        ValueError: `comments` is empty -- nothing to inject.
    """
    if not comments:
        raise ValueError("revise_plan: comments must not be empty")
    return await resume_task(
        task_id, deps, inject_message=render_plan_comments(comments), clock_fn=clock_fn
    )
