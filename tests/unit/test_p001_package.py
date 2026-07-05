"""Tests for the packaged version constant and the CLI argument-parsing skeleton."""

from __future__ import annotations

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


def test_no_arguments_returns_not_yet_implemented() -> None:
    """With no subcommand, the REPL path is not yet implemented and exits 2."""
    exit_code = main([])

    assert exit_code == 2


def test_doctor_subcommand_returns_not_yet_implemented() -> None:
    """The ``doctor`` subcommand is accepted by the parser but not yet implemented."""
    exit_code = main(["doctor"])

    assert exit_code == 2
