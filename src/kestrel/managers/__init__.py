"""Managers own a slice of Kestrel's own runtime state, on disk under
the target repo -- distinct from tools, which only ever act on the
model's behalf and never persist anything themselves.
"""

from kestrel.managers.undo import UndoConflictError, UndoEntry, UndoManager

__all__ = [
    "UndoConflictError",
    "UndoEntry",
    "UndoManager",
]
