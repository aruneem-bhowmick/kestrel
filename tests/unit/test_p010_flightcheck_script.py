"""Tests for the Jetson environment flight-check script's committed shape.

The script's actual pass/fail behavior can only be fully exercised on a
real Jetson board (or, for the arch-independent checks, in CI -- see
scripts/jetson-flightcheck.sh's own --ci-mode flag); what a hermetic unit
test can and should pin is that the script is present, starts in strict
mode, and carries the executable bit it needs to be run directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.p010,
    pytest.mark.unit,
    pytest.mark.sanity,
    pytest.mark.regression,
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "jetson-flightcheck.sh"
_SCRIPT_GIT_PATH = "scripts/jetson-flightcheck.sh"


def test_flightcheck_script_exists_with_bash_strict_mode() -> None:
    """Given the script's committed text, when its first lines are read,
    then it starts with the shebang and strict-mode pragma its safety as
    a re-runnable, fail-loud environment check depends on."""
    assert _SCRIPT_PATH.is_file()
    lines = _SCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in lines[:5]


def test_flightcheck_script_is_committed_with_the_executable_bit() -> None:
    """Given the script's path, when its mode is read from the git index,
    then it reports the executable bit -- checked through git rather than
    `os.access`, since only git's own tracked mode (not the host
    filesystem's permission bits) is meaningful on Windows checkouts."""
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", _SCRIPT_GIT_PATH],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout, f"{_SCRIPT_GIT_PATH} is not tracked by git"
    mode = result.stdout.split()[0]
    assert mode == "100755"
