"""Unit tests for `kestrel.doctor._check_tui`: the flight check backing
the same non-interactive-stdout guard `kestrel` (no subcommand) applies
before mounting the cockpit.
"""

from __future__ import annotations

import sys

import pytest

from kestrel.doctor import CheckStatus, _check_tui

pytestmark = [pytest.mark.p043, pytest.mark.unit, pytest.mark.sanity]


def test_reports_ok_when_stdout_is_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given `sys.stdout.isatty()` returns `True`, when the check runs,
    then it reports OK naming the interactive state -- no remedy text,
    since there is nothing to fix."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    result = _check_tui()

    assert result.status is CheckStatus.OK
    assert result.detail == "interactive"


def test_reports_fail_naming_the_run_alternative_when_stdout_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `sys.stdout.isatty()` returns `False` (a piped or redirected
    stdout), when the check runs, then it FAILs with a detail naming both
    the reason and the `kestrel run` invocation to use instead."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    result = _check_tui()

    assert result.status is CheckStatus.FAIL
    assert "not a terminal" in result.detail
    assert "kestrel run" in result.detail
    assert "--repo PATH" in result.detail
