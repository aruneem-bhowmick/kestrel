"""Acceptance test proving the ``kestrel`` console script is wired end to end."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kestrel import __version__

pytestmark = [pytest.mark.p001, pytest.mark.acceptance]


def _kestrel_executable() -> str:
    """Locate the installed ``kestrel`` console script in the active environment.

    ``uv run pytest`` puts the environment's script directory on ``PATH``, so
    :func:`shutil.which` finds it directly. As a fallback (e.g. when a test
    runner invokes pytest without going through ``uv run``), the script lives
    alongside the interpreter that is currently running, since console
    scripts are installed into the same directory as the Python executable
    of the environment that owns them.
    """
    found = shutil.which("kestrel")
    if found is not None:
        return found

    exe_dir = Path(sys.executable).parent
    for candidate_name in ("kestrel", "kestrel.exe"):
        candidate = exe_dir / candidate_name
        if candidate.exists():
            return str(candidate)

    pytest.fail(
        "kestrel console script not found on PATH or in the environment's "
        "script directory"
    )


def test_console_script_prints_version() -> None:
    """Running the installed ``kestrel`` console script with ``--version``
    prints the package version and exits 0.
    """
    result = subprocess.run(
        [_kestrel_executable(), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == __version__
