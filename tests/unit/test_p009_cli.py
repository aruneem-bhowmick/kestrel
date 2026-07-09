"""Unit tests for the CLI's `doctor` subcommand wiring: flag parsing
(including both orderings of `--config`), exit-code derivation, and that
output actually reaches stdout through the real entry point rather than
only through :func:`kestrel.doctor.run_doctor` called directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config as kestrel_config
from kestrel.cli import main
from kestrel.registry import loader as registry_loader

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


@pytest.fixture
def user_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user config
    directory, so tests never touch (or depend on) the real home directory.
    """
    return tmp_path_factory.mktemp("userconfig")


@pytest.fixture(autouse=True)
def _isolated_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_config_dir: Path
) -> None:
    """Chdir into an empty directory, clear ``$KESTREL_CONFIG`` and the
    OpenRouter credential, and point both the config and registry
    user-config-dir lookups at an empty temp directory."""
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        kestrel_config.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )
    monkeypatch.setattr(
        registry_loader.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )


def _write_config(tmp_path: Path) -> Path:
    """Write a valid ``kestrel.toml`` + ``models.toml`` pair and return
    the config path."""
    models_file = tmp_path / "models.toml"
    models_file.write_text(_VALID_MODELS_TOML, encoding="utf-8")

    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "glm-5.2"

[paths]
models_file = "{models_file.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def test_doctor_prints_eight_lines_and_exits_zero_when_all_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a valid config with its credential set, when `doctor` runs
    through the real CLI entry point, then it prints eight lines to
    stdout and returns exit code 0."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = _write_config(tmp_path)

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert len(captured.out.splitlines()) == 8


def test_doctor_exits_one_when_any_check_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a valid config but no credential set, when `doctor` runs,
    then it exits 1 and the printed report names the FAILing check."""
    config_path = _write_config(tmp_path)

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "api-key" in captured.out


def test_doctor_config_flag_works_before_the_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``--config`` precedes ``doctor`` on the command line (the
    top-level flag position), when parsed, then doctor still resolves
    against the named config rather than falling back to defaults."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = _write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "doctor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK    config" in captured.out
    assert str(config_path) in captured.out


def test_doctor_config_flag_works_after_the_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``--config`` follows ``doctor`` (the subcommand-scoped
    position used throughout this project's own CI and docs), when
    parsed, then it resolves identically to the flag preceding it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = _write_config(tmp_path)

    exit_code = main(["doctor", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK    config" in captured.out
    assert str(config_path) in captured.out


def test_doctor_without_live_flag_skips_the_endpoint_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``doctor`` runs with no ``--live`` flag, when parsed, then
    the endpoint line reports SKIP with the "pass --live" hint rather
    than attempting a network call."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = _write_config(tmp_path)

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
