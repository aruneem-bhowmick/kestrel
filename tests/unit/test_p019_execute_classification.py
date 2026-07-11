"""Tests for `classify_destructive_action`'s pattern table -- which
commands `execute` gates behind approval, and which run unchecked --
and for `execute`'s own wiring of that classification into a real
`approval.check()` call before `run_sandboxed` ever runs.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from kestrel.managers.approval import (
    ApprovalDecision,
    ApprovalDenied,
    ApprovalManager,
    ApprovalRequest,
)
from kestrel.tools.execute import ExecuteArgs, classify_destructive_action, execute
from kestrel.tools.sandbox import SandboxResult

pytestmark = [pytest.mark.p019, pytest.mark.unit]

# See `test_p016_execute_redteam.py` for why this module is resolved via
# `importlib` rather than `import kestrel.tools.execute as execute_module`:
# `kestrel.tools.__init__` rebinds the `execute` *attribute* on the
# `kestrel.tools` package to the function of the same name.
_execute_module = importlib.import_module("kestrel.tools.execute")


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("cmd", "expected_kind"),
    [
        (("rm", "-rf", "foo"), "delete"),
        (("rm", "foo"), "delete"),
        (("rmdir", "foo"), "delete"),
        (("git", "push", "--force"), "force_push"),
        (("git", "push", "-f"), "force_push"),
        (("git", "push", "origin", "main", "--force"), "force_push"),
        (("chmod", "+x", "foo"), "chmod"),
        (("chmod", "755", "foo"), "chmod"),
    ],
)
def test_classifies_recognized_destructive_commands(
    cmd: tuple[str, ...], expected_kind: str
) -> None:
    """Given a command matching the pattern table, when classified,
    then it produces an `ApprovalRequest` of the expected kind."""
    request = classify_destructive_action(cmd)

    assert request is not None
    assert request.kind == expected_kind


@pytest.mark.sanity
@pytest.mark.parametrize(
    "cmd",
    [
        ("pytest", "-q"),
        ("ls",),
        ("git", "status"),
        ("git", "push"),
        ("git", "push", "origin", "main"),
        (),
    ],
)
def test_unrecognized_commands_are_not_classified(cmd: tuple[str, ...]) -> None:
    """Given a command outside the pattern table -- including a bare
    `git push` with no force flag, and an empty `cmd` -- when
    classified, then it produces no `ApprovalRequest`."""
    assert classify_destructive_action(cmd) is None


def test_classified_request_carries_the_full_command_as_detail() -> None:
    """Given a recognized command, when classified, then the resulting
    request's `detail` is the exact command joined by spaces -- what a
    reviewer needs to make an informed decision, verbatim."""
    request = classify_destructive_action(("rm", "-rf", "build/"))

    assert request is not None
    assert request.detail == "rm -rf build/"
    assert "rm -rf build/" in request.summary


def _stub_run_sandboxed(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Monkeypatch `execute`'s own `run_sandboxed` reference to a stub
    that records every `cmd` it is called with and returns a trivial
    successful result, so a test can prove whether the sandbox was
    reached at all without spawning anything real."""
    calls: list[list[str]] = []

    def _stub(cmd: list[str], **_kwargs: object) -> SandboxResult:
        calls.append(cmd)
        return SandboxResult(stdout="", stderr="", exit_code=0, timed_out=False)

    monkeypatch.setattr(_execute_module, "run_sandboxed", _stub)
    return calls


def test_destructive_command_is_checked_before_the_sandbox_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a command classified as destructive, when `execute` runs,
    then `approval.check` is called with the matching `ApprovalRequest`
    before `run_sandboxed` is ever reached."""
    sandbox_calls = _stub_run_sandboxed(monkeypatch)
    checked: list[ApprovalRequest] = []

    def _decide(request: ApprovalRequest) -> ApprovalDecision:
        checked.append(request)
        return "once"

    approval = ApprovalManager(decide_fn=_decide)

    execute(
        ExecuteArgs(cmd=("rm", "-rf", "foo")), repo_root=tmp_path, approval=approval
    )

    assert len(checked) == 1
    assert checked[0].kind == "delete"
    assert sandbox_calls == [["rm", "-rf", "foo"]]


def test_denied_destructive_command_never_reaches_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a command classified as destructive and denied, when
    `execute` runs, then `ApprovalDenied` propagates and `run_sandboxed`
    is never called."""
    sandbox_calls = _stub_run_sandboxed(monkeypatch)
    approval = ApprovalManager(decide_fn=lambda _request: "deny")

    with pytest.raises(ApprovalDenied):
        execute(
            ExecuteArgs(cmd=("rm", "-rf", "foo")), repo_root=tmp_path, approval=approval
        )

    assert sandbox_calls == []


def test_unclassified_command_never_calls_approval_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a command outside the pattern table, when `execute` runs,
    then `approval.check` is never called and the command reaches the
    sandbox unchecked."""
    sandbox_calls = _stub_run_sandboxed(monkeypatch)

    def _unexpected_decide_fn(request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(f"decide_fn should not be called for {request!r}")

    approval = ApprovalManager(decide_fn=_unexpected_decide_fn)

    execute(ExecuteArgs(cmd=("git", "status")), repo_root=tmp_path, approval=approval)

    assert sandbox_calls == [["git", "status"]]
