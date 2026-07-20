"""Turns a finished task's own `LoopResult` into a persisted `Walkthrough`
artifact naming what changed, which files, its most recent verification
(if any), and its total cost.

This is a pure data layer: nothing here talks to a live model, a CLI
flag, or a Textual widget -- `kestrel.agent.plan` already establishes
that same split for `ImplementationPlan`, and this module follows it for
`Walkthrough`. `build_walkthrough` folds an already-completed
`LoopResult`, an `UndoManager`'s own journal (filtered to one task), and
that task's own verification history into one `Walkthrough`;
`render_walkthrough_markdown`/`persist_walkthrough` turn that back into
markdown and write it under `.kestrel/artifacts/` through
`kestrel.agent._artifact_paths.persist_markdown_artifact` -- P-048's own
shared helper, reused verbatim here as its second caller.

`cli.py`'s own `_print_task_summary` already computes "every distinct
path the task's own `UndoManager` journal recorded a mutation for"
inline; `build_walkthrough` performs the identical computation
independently rather than refactoring that already-tested function to
share it. This is a deliberate, small duplication rather than an
oversight: giving a new artifact type its own module is in scope here,
reshaping an existing, already-tested CLI helper is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from kestrel.agent._artifact_paths import persist_markdown_artifact
from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.managers.undo import UndoManager
from kestrel.tools.verify import VerificationReport, render_verification_markdown


@dataclass(frozen=True, slots=True)
class Walkthrough:
    """One completed task's own auto-generated summary artifact.

    Attributes:
        task_id: The task this walkthrough describes.
        reason: How the task ended.
        turns_used: How many model calls it made.
        total_usd: Its total priced cost.
        touched_paths: Every distinct repo-relative path its own undo
            journal recorded a mutation for, sorted.
        verification: The most recent `VerificationReport` it produced,
            or `None` if it never called `verify`.
    """

    task_id: str
    reason: TerminationReason
    turns_used: int
    total_usd: Decimal
    touched_paths: tuple[str, ...]
    verification: VerificationReport | None


class WalkthroughError(Exception):
    """`persist_walkthrough` could not write its artifact. `str(self)`
    names the remedy."""


def build_walkthrough(
    result: LoopResult,
    *,
    task_id: str,
    undo: UndoManager,
    verification_reports: Sequence[VerificationReport],
) -> Walkthrough:
    """Fold `result`, `undo`'s own journal (filtered to `task_id`), and
    the task's own `verification_reports` (its last entry, if any) into
    one `Walkthrough`."""
    touched = sorted({entry.path for entry in undo.entries if entry.task_id == task_id})
    verification = verification_reports[-1] if verification_reports else None
    return Walkthrough(
        task_id=task_id,
        reason=result.reason,
        turns_used=result.turns_used,
        total_usd=result.total_usd,
        touched_paths=tuple(touched),
        verification=verification,
    )


def render_walkthrough_markdown(walkthrough: Walkthrough) -> str:
    """`# Walkthrough: {task_id}`, then `## Result` (reason/turns/cost),
    `## Files changed` (a bullet list, or `_none_` when empty), and
    `## Verification` -- `render_verification_markdown`'s own output,
    embedded verbatim, when `walkthrough.verification is not None`, else
    `_no verification ran_`."""
    lines = [f"# Walkthrough: {walkthrough.task_id}", ""]

    lines.append("## Result")
    lines.append(f"- reason: {walkthrough.reason.name}")
    lines.append(f"- turns: {walkthrough.turns_used}")
    lines.append(f"- total_usd: ${walkthrough.total_usd:.4f}")
    lines.append("")

    lines.append("## Files changed")
    if walkthrough.touched_paths:
        lines.extend(f"- {path}" for path in walkthrough.touched_paths)
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## Verification")
    if walkthrough.verification is not None:
        lines.append(render_verification_markdown(walkthrough.verification))
    else:
        lines.append("_no verification ran_")

    return "\n".join(lines).rstrip("\n") + "\n"


def persist_walkthrough(walkthrough: Walkthrough, *, repo_root: Path) -> Path:
    """Persist `render_walkthrough_markdown(walkthrough)` via
    `persist_markdown_artifact`, stem `f"walkthrough-{walkthrough.task_id}"`.

    Raises:
        WalkthroughError: see `persist_markdown_artifact`'s own
            `ValueError`.
    """
    try:
        return persist_markdown_artifact(
            render_walkthrough_markdown(walkthrough),
            repo_root=repo_root,
            stem=f"walkthrough-{walkthrough.task_id}",
        )
    except ValueError as exc:
        raise WalkthroughError(
            f".kestrel/artifacts: refusing to persist "
            f"walkthrough-{walkthrough.task_id} ({exc})"
        ) from exc
