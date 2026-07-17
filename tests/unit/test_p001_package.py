"""Tests for the packaged version constant and the CLI argument-parsing skeleton."""

from __future__ import annotations

import io
import sys
from importlib import metadata

import pytest

from kestrel import __version__
from kestrel.cli import main

pytestmark = [pytest.mark.p001, pytest.mark.unit, pytest.mark.sanity]


def test_version_matches_installed_distribution() -> None:
    """The importable version constant matches the installed distribution metadata."""
    assert __version__ == metadata.version("kestrel-cli")


def test_version_flag_prints_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` prints the version string and returns exit code 0."""
    exit_code = main(["--version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == __version__


def test_no_arguments_refuses_the_cockpit_on_a_non_interactive_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no subcommand, `main` resolves the packaged default
    configuration and registry, then refuses to mount the interactive
    cockpit against a non-interactive stdout, returning exit code 1 and
    printing the documented alternative instead of trying to draw a
    full-screen interface into a pipe."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "kestrel run" in captured.err
