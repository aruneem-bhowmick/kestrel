"""Shared repo-relative path containment guard.

`resolve_repo_path` is the one function every filesystem-touching tool
in this package calls on a caller-supplied path before doing anything
else with it: it resolves `path` against a repo root, following any
symlink, and refuses a location that ends up outside that root.
`read_file` and `edit_file` both call through this single helper
rather than each keeping its own copy of the check, so a future fix to
how containment is verified only has to happen in one place, and the
two tools can never drift apart on what counts as an escape.
"""

from __future__ import annotations

from pathlib import Path


def resolve_repo_path(path: str, *, repo_root: Path) -> Path:
    """Resolve `path` against `repo_root`, following any symlink, and
    return the resolved, absolute `Path` -- whether or not anything
    actually exists there yet.

    Raises:
        ValueError: the resolved location falls outside `repo_root`,
            whether by a `..` climb or by following a symlink that
            points outside it. The message names `path` and is safe to
            surface to a model verbatim.
    """
    resolved_root = repo_root.resolve()
    candidate = (repo_root / path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"{path}: escapes the repository root")
    return candidate
