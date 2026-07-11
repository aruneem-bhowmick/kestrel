"""System test: a real `execute` call, through the real `bwrap` sandbox
and the real `_prompt_stdin` decision function, driven by a piped
answer instead of a real terminal -- a denied delete leaves its target
file untouched, and an approved one really deletes it.

Skipped when `bwrap` is not on `PATH`, the same real local seam
`tests/integration/test_p016_sandbox.py` skips on -- this proves an
actual file survives or disappears on disk, not a mocked
`run_sandboxed` call.
"""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

import pytest

from kestrel.managers.approval import ApprovalDenied, ApprovalManager, _prompt_stdin
from kestrel.tools.execute import ExecuteArgs, execute

pytestmark = [
    pytest.mark.p019,
    pytest.mark.system,
    pytest.mark.skipif(
        shutil.which("bwrap") is None or os.getenv("GITHUB_ACTIONS") == "true",
        reason="bwrap not available or insufficient sandbox permissions in CI",
    ),
]


def _approval_with_piped_reply(reply: str) -> ApprovalManager:
    """Build an `ApprovalManager` whose `decide_fn` is the real
    `_prompt_stdin` -- rendering the prompt and parsing a reply exactly
    as the real terminal path does -- fed `reply` instead of a real
    keystroke."""
    return ApprovalManager(
        decide_fn=functools.partial(_prompt_stdin, input_fn=lambda _prompt: reply)
    )


def test_piped_no_answer_denies_the_delete_and_the_file_survives(
    tmp_path: Path,
) -> None:
    """Given a real `rm` request through the real approval path with a
    piped "n" answer, when `execute` runs, then `ApprovalDenied` is
    raised, `run_sandboxed` never executes the command, and the target
    file is still on disk."""
    repo_root = tmp_path.resolve()
    target = repo_root / "somefile"
    target.write_text("keep me")

    with pytest.raises(ApprovalDenied):
        execute(
            ExecuteArgs(cmd=("rm", "-f", target.name)),
            repo_root=repo_root,
            approval=_approval_with_piped_reply("n"),
        )

    assert target.exists()


def test_piped_yes_answer_allows_the_delete_and_the_file_is_gone(
    tmp_path: Path,
) -> None:
    """Given a real `rm` request through the real approval path with a
    piped "y" answer, when `execute` runs, then the command is allowed
    through to the real sandbox and the target file is gone."""
    repo_root = tmp_path.resolve()
    target = repo_root / "somefile"
    target.write_text("delete me")

    result = execute(
        ExecuteArgs(cmd=("rm", "-f", target.name)),
        repo_root=repo_root,
        approval=_approval_with_piped_reply("y"),
    )

    assert not target.exists(), f"File was not deleted. Sandbox output:\n{result}"
