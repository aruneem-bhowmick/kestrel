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


def test_no_arguments_starts_the_repl_and_exits_cleanly_on_eof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no subcommand, the REPL starts against the packaged default
    configuration and registry, and returns exit code 0 as soon as stdin
    is exhausted (mirroring a piped, non-interactive invocation)."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "kestrel" in captured.out


def test_doctor_subcommand_returns_not_yet_implemented() -> None:
    """The ``doctor`` subcommand is accepted by the parser but not yet implemented."""
    exit_code = main(["doctor"])

    assert exit_code == 2
