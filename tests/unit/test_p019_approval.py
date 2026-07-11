"""Tests for `ApprovalManager`'s allowlist/session-approval semantics
and `_prompt_stdin`'s reply parsing.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from kestrel.managers.approval import (
    ApprovalDecision,
    ApprovalDenied,
    ApprovalManager,
    ApprovalRequest,
    DestructiveKind,
    _prompt_stdin,
)

pytestmark = [pytest.mark.p019, pytest.mark.unit]


def _request(kind: DestructiveKind = "delete") -> ApprovalRequest:
    """Build a representative `ApprovalRequest` for a given `kind`."""
    return ApprovalRequest(kind=kind, summary="delete somefile", detail="rm somefile")


def _unexpected_decide_fn(request: ApprovalRequest) -> ApprovalDecision:
    """Stand in for `decide_fn`, failing the test if it is ever called."""
    raise AssertionError(f"decide_fn should not be called for {request.kind!r}")


def _make_recorder(
    decision: ApprovalDecision,
) -> tuple[Callable[[ApprovalRequest], ApprovalDecision], list[ApprovalRequest]]:
    """Return a `decide_fn` that always returns `decision`, plus the
    list it records every request onto, in call order."""
    calls: list[ApprovalRequest] = []

    def _decide(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return decision

    return _decide, calls


@pytest.mark.sanity
def test_allowlisted_kind_never_calls_decide_fn() -> None:
    """Given a kind present in the allowlist, when checked, then
    `decide_fn` is never invoked -- the request is allowed outright."""
    manager = ApprovalManager(
        allowlist=frozenset({"delete"}), decide_fn=_unexpected_decide_fn
    )

    manager.check(_request(kind="delete"))


@pytest.mark.sanity
def test_once_decision_allows_but_does_not_persist() -> None:
    """Given `decide_fn` returning `"once"`, when checked, then the
    request is allowed, but an identical follow-up request calls
    `decide_fn` again rather than short-circuiting."""
    decide_fn, calls = _make_recorder("once")
    manager = ApprovalManager(decide_fn=decide_fn)

    manager.check(_request(kind="delete"))
    manager.check(_request(kind="delete"))

    assert len(calls) == 2


@pytest.mark.sanity
def test_always_decision_allows_and_persists_for_the_session() -> None:
    """Given `decide_fn` returning `"always"`, when checked, then the
    request is allowed and a second same-kind request short-circuits
    without calling `decide_fn` again."""
    decide_fn, calls = _make_recorder("always")
    manager = ApprovalManager(decide_fn=decide_fn)

    manager.check(_request(kind="delete"))
    manager.check(_request(kind="delete"))

    assert len(calls) == 1


def test_always_decision_does_not_leak_across_kinds() -> None:
    """Given `"always"` recorded for one kind, when a different kind is
    checked, then `decide_fn` is still called for it -- session
    approval is scoped per kind, not global."""
    decide_fn, calls = _make_recorder("always")
    manager = ApprovalManager(decide_fn=decide_fn)

    manager.check(_request(kind="delete"))
    manager.check(_request(kind="chmod"))

    assert len(calls) == 2
    assert {call.kind for call in calls} == {"delete", "chmod"}


@pytest.mark.sanity
def test_deny_decision_raises_approval_denied_naming_the_summary() -> None:
    """Given `decide_fn` returning `"deny"`, when checked, then
    `ApprovalDenied` is raised carrying the request's own `summary`."""
    decide_fn, _calls = _make_recorder("deny")
    manager = ApprovalManager(decide_fn=decide_fn)

    with pytest.raises(ApprovalDenied, match="delete somefile"):
        manager.check(_request(kind="delete"))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("y", "once"),
        ("yes", "once"),
        ("Y", "once"),
        ("YES", "once"),
        ("always", "always"),
        ("ALWAYS", "always"),
        ("n", "deny"),
        ("no", "deny"),
        ("", "deny"),
        ("banana", "deny"),
    ],
)
def test_prompt_stdin_parses_every_accepted_spelling(
    reply: str, expected: ApprovalDecision
) -> None:
    """Given every accepted spelling of a reply (case-insensitive), when
    `_prompt_stdin` reads it via an injected `input_fn`, then it maps
    to the expected `ApprovalDecision`."""
    decision = _prompt_stdin(_request(), input_fn=lambda _prompt: reply)

    assert decision == expected


def test_prompt_stdin_renders_summary_and_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a request, when `_prompt_stdin` runs, then it prints both
    the request's `summary` and its `detail` before reading a reply."""
    request = ApprovalRequest(kind="delete", summary="Delete: rm x", detail="rm x")

    _prompt_stdin(request, input_fn=lambda _prompt: "n")

    captured = capsys.readouterr()
    assert "Delete: rm x" in captured.out
    assert "rm x" in captured.out
