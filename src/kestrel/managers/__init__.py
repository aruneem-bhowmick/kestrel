"""Managers own a slice of Kestrel's own runtime state, on disk under
the target repo -- distinct from tools, which only ever act on the
model's behalf and never persist anything themselves. `ApprovalManager`
is the one exception to the on-disk half of that rule: its state is
purely in-memory and scoped to a single session, since "approved for
the rest of this session" is meaningless once the process exits.
"""

from kestrel.managers.approval import (
    ApprovalDecision,
    ApprovalDenied,
    ApprovalManager,
    ApprovalRequest,
    DestructiveKind,
)
from kestrel.managers.undo import UndoConflictError, UndoEntry, UndoManager

__all__ = [
    "ApprovalDecision",
    "ApprovalDenied",
    "ApprovalManager",
    "ApprovalRequest",
    "DestructiveKind",
    "UndoConflictError",
    "UndoEntry",
    "UndoManager",
]
