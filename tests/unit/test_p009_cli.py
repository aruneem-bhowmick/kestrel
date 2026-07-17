"""Unit tests for the CLI's `doctor` subcommand wiring: flag parsing
(including both orderings of `--config`), exit-code derivation, and that
output actually reaches stdout through the real entry point rather than
only through :func:`kestrel.doctor.run_doctor` called directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import kestrel.doctor as doctor_module
from kestrel.cli import main
from kestrel.tools.sandbox import SandboxResult

pytestmark = [pytest.mark.p009, pytest.mark.unit, pytest.mark.sanity]

_VALID_MODELS_TOML = """\
[[models]]
id = "glm-5.2"
backend = "openrouter"
provider_model = "z-ai/glm-5.2"
api_key_env = "OPENROUTER_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true
"""


def _patch_sandbox_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ``sandbox`` check deterministically ``OK``, so a test
    asserting an all-green exit code does not depend on ``bwrap`` being
    installed wherever this suite happens to run."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: True)
    monkeypatch.setattr(
        doctor_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout="", stderr="", exit_code=0, timed_out=False
        ),
    )


def _patch_tui_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ``tui`` check deterministically ``OK``, so a test
    asserting an all-green exit code does not depend on whatever
    ambient tty state this suite happens to run under."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


def test_doctor_prints_nine_lines_and_exits_zero_when_all_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given a valid config with its credential set, when `doctor` runs
    through the real CLI entry point, then it prints nine lines to
    stdout and returns exit code 0."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    _patch_sandbox_ok(monkeypatch)
    _patch_tui_ok(monkeypatch)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert len(captured.out.splitlines()) == 9


def test_doctor_exits_one_when_any_check_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given a valid config but no credential set, when `doctor` runs,
    then it exits 1 and the printed report names the FAILing check."""
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "api-key" in captured.out


def test_doctor_config_flag_works_before_the_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given ``--config`` precedes ``doctor`` on the command line (the
    top-level flag position), when parsed, then doctor still resolves
    against the named config rather than falling back to defaults."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    _patch_sandbox_ok(monkeypatch)
    _patch_tui_ok(monkeypatch)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    exit_code = main(["--config", str(config_path), "doctor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK    config" in captured.out
    assert str(config_path) in captured.out


def test_doctor_config_flag_works_after_the_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given ``--config`` follows ``doctor`` (the subcommand-scoped
    position used throughout this project's own CI and docs), when
    parsed, then it resolves identically to the flag preceding it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    _patch_sandbox_ok(monkeypatch)
    _patch_tui_ok(monkeypatch)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK    config" in captured.out
    assert str(config_path) in captured.out


def test_doctor_without_live_flag_skips_the_endpoint_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given ``doctor`` runs with no ``--live`` flag, when parsed, then
    the endpoint line reports SKIP with the "pass --live" hint rather
    than attempting a network call."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    _patch_sandbox_ok(monkeypatch)
    _patch_tui_ok(monkeypatch)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SKIP  endpoint" in captured.out
    assert "pass --live" in captured.out


def test_doctor_missing_config_file_fails_gracefully_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``--config`` names a file that does not exist, when `doctor`
    runs, then it exits 1 with a readable FAIL line instead of raising --
    doctor's whole point is diagnosing a broken environment, not adding
    to the pile of tracebacks."""
    missing = tmp_path / "missing.toml"

    exit_code = main(["doctor", "--config", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FAIL  config" in captured.out
