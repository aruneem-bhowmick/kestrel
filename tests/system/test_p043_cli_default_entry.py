"""System test: the installed console script's default (no-subcommand)
entry point refuses to mount the interactive cockpit against a
non-interactive stdout.

`subprocess.run(capture_output=True, ...)` pipes the child's stdout by
construction, so every invocation here exercises the exact guard
`kestrel.cli.main` applies immediately before it would otherwise import
`kestrel.tui.app.KestrelApp`. That import sits textually after the guard
in `cli.py`'s own final branch, so a clean one-line refusal on stderr --
rather than some raw Textual failure -- is itself proof neither the
import nor the cockpit it would have mounted was ever reached.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p043, pytest.mark.system]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TIMEOUT_S = 30.0
_EXPECTED_STDERR = (
    "kestrel: refusing to start the TUI against a non-interactive "
    'stdout; use `kestrel run "<task>" --repo PATH` instead.\n'
)


def test_default_entry_point_refuses_a_piped_stdout(kestrel_executable: str) -> None:
    """Given the committed system-test fixture config and its credential
    env var set, when `kestrel` runs with no subcommand at all against a
    piped stdin/stdout, then it exits 1, prints nothing to stdout, and
    prints exactly the documented refusal message to stderr -- naming
    the non-interactive `kestrel run` alternative -- rather than letting
    Textual fail against a terminal that was never there.
    """
    env = dict(os.environ)
    env["KESTREL_SYSTEM_TEST_API_KEY"] = "sk-test-system"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [kestrel_executable, "--config", "tests/fixtures/kestrel.system.toml"],
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=_REPO_ROOT,
        timeout=_TIMEOUT_S,
        check=False,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == _EXPECTED_STDERR
