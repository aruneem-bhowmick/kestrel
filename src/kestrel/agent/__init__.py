"""The tool-calling agent loop.

`run_task` drives one task through repeated model calls, self-critique,
approval-gated tool dispatch, and cost accounting until it completes or
a termination predicate trips. Everything a call needs arrives through
one `LoopDeps` bundle, so nothing in this package reaches for global
state or constructs its own collaborators.
"""

from kestrel.agent.loop import (
    LoopDeps,
    LoopLimits,
    LoopResult,
    TerminationReason,
    run_task,
)
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
from kestrel.agent.walkthrough import (
    Walkthrough,
    WalkthroughError,
    build_walkthrough,
    persist_walkthrough,
    render_walkthrough_markdown,
)

__all__ = [
    "ImplementationPlan",
    "LoopDeps",
    "LoopLimits",
    "LoopResult",
    "PlanComment",
    "PlanError",
    "PlanLine",
    "TerminationReason",
    "Walkthrough",
    "WalkthroughError",
    "build_walkthrough",
    "extract_plan_from_result",
    "parse_plan_lines",
    "persist_plan",
    "persist_walkthrough",
    "render_plan_comments",
    "render_plan_markdown",
    "render_walkthrough_markdown",
    "revise_plan",
    "run_task",
]
