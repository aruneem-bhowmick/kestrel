"""Tests for the CLI's REPL-wiring error paths: a bad config, a bad
registry, and an unknown starting model each exit 1 before the REPL loop
ever starts, instead of surfacing as an unhandled traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config as kestrel_config
from kestrel.cli import main
from kestrel.registry import loader as registry_loader

pytestmark = [pytest.mark.p008, pytest.mark.unit, pytest.mark.sanity]


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
    """Chdir into an empty directory, clear ``$KESTREL_CONFIG``, and point
    both the config and registry user-config-dir lookups at an empty temp
    directory, so no test here can be polluted by (or pollute) the
    developer's real machine.
    """
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
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


def test_missing_explicit_config_file_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``--config`` names a file that does not exist, when main
    runs, then it exits 1 and reports the missing path on stderr."""
    missing = tmp_path / "missing.toml"

    exit_code = main(["--config", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert str(missing) in captured.err


def test_broken_registry_file_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given the configured registry file fails validation (a zai entry
    missing its required endpoint), when main runs, then it exits 1 and
    reports the failure on stderr instead of starting the REPL."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(
        """\
[[models]]
id = "broken"
backend = "zai"
provider_model = "glm-5.2"
api_key_env = "ZAI_API_KEY"
context_window = 1000
max_output = 100
usd_per_mtok_input = 1.0
usd_per_mtok_output = 2.0
usd_per_mtok_cached = 0.5
supports_tools = true
supports_cache = false
""",
        encoding="utf-8",
    )
    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text(
        f'[paths]\nmodels_file = "{models_toml.as_posix()}"\n', encoding="utf-8"
    )

    exit_code = main(["--config", str(kestrel_toml)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "endpoint" in captured.err


def test_unknown_model_flag_exits_one_before_the_repl_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given ``--model`` names an id absent from the registry, when main
    runs, then it exits 1 and reports the unknown id (and the available
    ones) on stderr, without ever reaching the REPL loop."""
    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text("", encoding="utf-8")

    exit_code = main(["--config", str(kestrel_toml), "--model", "nope"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unknown model id 'nope'" in captured.err
    assert "glm-5.2" in captured.err
