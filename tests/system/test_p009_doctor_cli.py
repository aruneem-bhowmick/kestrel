"""System test: the installed console script's `doctor` subcommand runs
every flight check against a committed fixture config and exits cleanly.

This is the smoke-lane test named in the spec's testing strategy: CI runs
it explicitly (``uv run pytest -m smoke``) as its own step, distinct from
the full suite, so a broken environment surfaces from one narrowly-scoped
command rather than only from a full test run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p009, pytest.mark.system, pytest.mark.smoke]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TIMEOUT_S = 30.0


def test_doctor_cli_reports_nine_lines_and_fails_tui_under_a_piped_stdout(
    kestrel_executable: str,
) -> None:
    """Given the committed system-test fixture config and its credential
    env var set, when `kestrel doctor` runs against it (without --live)
    as a real subprocess with its stdout captured through a pipe, then it
    prints exactly nine aligned lines: five OK checks, `endpoint`/`ollama`
    SKIP, `sandbox` OK on a `bwrap`-equipped runner or FAIL naming the
    reason otherwise, and `tui` FAIL -- a piped stdout is never a real
    terminal, so this check deterministically fails under any subprocess
    harness that captures output this way, which is exactly the failure
    mode it exists to catch. The exit code is always 1, since at least
    one check (`tui`, always; `sandbox`, conditionally) always FAILs
    here."""
    env = dict(os.environ)
    env["KESTREL_SYSTEM_TEST_API_KEY"] = "sk-test-system"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [
            kestrel_executable,
            "doctor",
            "--config",
            "tests/fixtures/kestrel.system.toml",
        ],
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=_REPO_ROOT,
        timeout=_TIMEOUT_S,
        check=False,
    )

    lines = result.stdout.splitlines()
    assert len(lines) == 9

    statuses = [line.split(None, 1)[0] for line in lines]
    actual_sandbox_status = statuses[6]
    assert actual_sandbox_status in ("OK", "FAIL")

    assert result.returncode == 1, result.stderr

    assert statuses == [
        "OK",
        "OK",
        "OK",
        "OK",
        "OK",
        "SKIP",
        actual_sandbox_status,
        "FAIL",
        "SKIP",
    ]

    names = [line.split(None, 2)[1] for line in lines]
    assert names == [
        "python-version",
        "config",
        "registry",
        "default-model",
        "api-key",
        "endpoint",
        "sandbox",
        "tui",
        "ollama",
    ]
    assert "pass --live" in lines[5]
    assert "not a terminal" in lines[7]


def test_doctor_cli_missing_credential_exits_one(kestrel_executable: str) -> None:
    """Given the fixture config but no credential env var set, when
    `kestrel doctor` runs, then it exits 1 and the api-key line reports
    FAIL naming the missing variable."""
    env = dict(os.environ)
    env.pop("KESTREL_SYSTEM_TEST_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [
            kestrel_executable,
            "doctor",
            "--config",
            "tests/fixtures/kestrel.system.toml",
        ],
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=_REPO_ROOT,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL  api-key" in result.stdout
    assert "KESTREL_SYSTEM_TEST_API_KEY" in result.stdout
