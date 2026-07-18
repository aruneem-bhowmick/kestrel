"""Shared markdown-artifact persistence, private to `kestrel.agent.*`.

Every typed artifact this package produces -- an `ImplementationPlan`
today, a future `Walkthrough` later -- ends its life the same way: it is
rendered to a markdown string by its own module, then written under
`repo_root / ".kestrel" / "artifacts"` through this one shared helper.
Centralizing that write here means the containment guard, the directory
bootstrap, and the collision-avoiding filename rule only have to be
gotten right once, and every artifact type that lands under this
directory is guaranteed to follow it identically -- the same rationale
`kestrel.tools._paths.resolve_repo_path` already established for every
filesystem-touching tool.

`kestrel.tools.verify.persist_verification_report` predates this module
and keeps its own private copy of the same shape rather than depending
on it, since a dispatched tool's own artifact and a task-level artifact
like `ImplementationPlan` are produced through genuinely different call
paths; this module exists so nothing built after it needs to repeat
that copy a third time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from kestrel.tools._paths import resolve_repo_path

_ARTIFACTS_DIRNAME: Final[str] = ".kestrel/artifacts"


def _reject_unsafe_stem(stem: str) -> None:
    """Refuse a `stem` that could turn `artifacts_dir / f"{stem}.md"`
    into a path outside `artifacts_dir`.

    Raises:
        ValueError: `stem` is empty, contains a path separator (`/` or
            `\\`, so a caller-supplied `stem` can never introduce extra
            path components), or is itself a `.`/`..` traversal segment.
    """
    if not stem or "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ValueError(f"{stem!r}: not a valid artifact filename stem")


def allocate_artifact_path(artifacts_dir: Path, *, stem: str) -> Path:
    """Atomically reserve an unused markdown path under `artifacts_dir`
    named `{stem}.md` when free, else `{stem}-{n}.md` for the first free
    numeric suffix -- the exact collision rule `kestrel.tools.verify
    ._allocate_report_path` already established, generalized to a
    caller-supplied stem. Each candidate is claimed via exclusive file
    creation rather than an `exists()` check beforehand, so two
    concurrent callers can never both select the same path, and a
    dangling symlink squatting on a candidate name is refused --
    exclusive creation fails on an existing symlink entry regardless of
    where it points -- rather than followed and written through.

    Raises:
        ValueError: `stem` is empty, contains a path separator, or is a
            `.`/`..` segment -- see `_reject_unsafe_stem`.
    """
    _reject_unsafe_stem(stem)
    suffix = 0
    while True:
        name = f"{stem}.md" if suffix == 0 else f"{stem}-{suffix}.md"
        candidate = artifacts_dir / name
        try:
            with candidate.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def persist_markdown_artifact(text: str, *, repo_root: Path, stem: str) -> Path:
    """Resolve `repo_root/.kestrel/artifacts` through the same
    `resolve_repo_path` containment guard `persist_verification_report`
    uses, create it if needed, atomically reserve a free path via
    `allocate_artifact_path`, write `text` into it, and return the path.

    Raises:
        ValueError: the artifacts directory resolves outside
            `repo_root`, or `stem` fails `allocate_artifact_path`'s own
            validation -- callers translate either into their own typed
            error (`PlanError`/`WalkthroughError`), matching
            `persist_verification_report`'s own `VerifyError`
            translation of the identical containment failure.
    """
    artifacts_dir = resolve_repo_path(_ARTIFACTS_DIRNAME, repo_root=repo_root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = allocate_artifact_path(artifacts_dir, stem=stem)
    path.write_text(text, encoding="utf-8")
    return path
