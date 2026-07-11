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

__all__ = [
    "LoopDeps",
    "LoopLimits",
    "LoopResult",
    "TerminationReason",
    "run_task",
]
