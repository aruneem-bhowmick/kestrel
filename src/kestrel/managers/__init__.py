"""Managers own a slice of Kestrel's own runtime state, on disk under
the target repo -- distinct from tools, which only ever act on the
model's behalf and never persist anything themselves. `ApprovalManager`,
`ModeManager`, and `BudgetManager` are exceptions to the on-disk half of
that rule: `ApprovalManager`'s state is purely in-memory and scoped to a
single session, since "approved for the rest of this session" is
meaningless once the process exits; `ModeManager` likewise holds nothing
but a session's own live PLAN/FAST state; and `BudgetManager` holds no
per-session state at all -- it is a pure classifier over spend figures
its caller already computed.
"""

from kestrel.managers.approval import (
    ApprovalDecision,
    ApprovalDenied,
    ApprovalManager,
    ApprovalRequest,
    DestructiveKind,
)
from kestrel.managers.budget import (
    BudgetEvent,
    BudgetLimits,
    BudgetManager,
    BudgetStatus,
)
from kestrel.managers.mode import Mode, ModeManager
from kestrel.managers.undo import UndoConflictError, UndoEntry, UndoManager

__all__ = [
    "ApprovalDecision",
    "ApprovalDenied",
    "ApprovalManager",
    "ApprovalRequest",
    "BudgetEvent",
    "BudgetLimits",
    "BudgetManager",
    "BudgetStatus",
    "DestructiveKind",
    "Mode",
    "ModeManager",
    "UndoConflictError",
    "UndoEntry",
    "UndoManager",
]
