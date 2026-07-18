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


def allocate_artifact_path(artifacts_dir: Path, *, stem: str) -> Path:
    """An unused markdown path under `artifacts_dir` named `{stem}.md`
    when free, else `{stem}-{n}.md` for the first free numeric suffix
    -- the exact collision rule `kestrel.tools.verify
    ._allocate_report_path` already established, generalized to a
    caller-supplied stem."""
    candidate = artifacts_dir / f"{stem}.md"
    suffix = 1
    while candidate.exists():
        candidate = artifacts_dir / f"{stem}-{suffix}.md"
        suffix += 1
    return candidate


def persist_markdown_artifact(text: str, *, repo_root: Path, stem: str) -> Path:
    """Resolve `repo_root/.kestrel/artifacts` through the same
    `resolve_repo_path` containment guard `persist_verification_report`
    uses, create it if needed, allocate a free path via
    `allocate_artifact_path`, write `text`, and return the path.

    Raises:
        ValueError: the artifacts directory resolves outside
            `repo_root` -- callers translate this into their own typed
            error (`PlanError`/`WalkthroughError`), matching
            `persist_verification_report`'s own `VerifyError`
            translation of the identical failure.
    """
    artifacts_dir = resolve_repo_path(_ARTIFACTS_DIRNAME, repo_root=repo_root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = allocate_artifact_path(artifacts_dir, stem=stem)
    path.write_text(text, encoding="utf-8")
    return path
