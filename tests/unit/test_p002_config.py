"""Tests for the layered kestrel.toml configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config

pytestmark = [pytest.mark.p002, pytest.mark.unit]


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
    the user-config-dir lookup at an empty temp directory so every test
    starts with no ambient config layers and cannot pollute (or be
    polluted by) the developer's real machine.
    """
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )


@pytest.mark.sanity
def test_no_config_files_anywhere_returns_builtin_defaults() -> None:
    """Given no config file in any layer, when load_config runs, then it
    returns the built-in defaults and reports no source path."""
    loaded, source = config.load_config()

    assert loaded == config.KestrelConfig()
    assert source is None


@pytest.mark.sanity
def test_cwd_config_wins_over_user_config_dir(
    tmp_path: Path, user_config_dir: Path
) -> None:
    """Given both a ./kestrel.toml and a user-config-dir kestrel.toml, when
    load_config runs, then the cwd file wins and the layers are not merged.
    """
    (user_config_dir / "kestrel.toml").write_text(
        '[general]\ndefault_model = "from-user-dir"\n'
    )
    (tmp_path / "kestrel.toml").write_text('[general]\ndefault_model = "from-cwd"\n')

    loaded, source = config.load_config()

    assert loaded.general.default_model == "from-cwd"
    assert source == tmp_path / "kestrel.toml"


@pytest.mark.sanity
def test_env_var_beats_cwd_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given both $KESTREL_CONFIG and a ./kestrel.toml, when load_config
    runs, then the environment variable's file wins."""
    (tmp_path / "kestrel.toml").write_text('[general]\ndefault_model = "from-cwd"\n')
    env_config = tmp_path / "env-kestrel.toml"
    env_config.write_text('[general]\ndefault_model = "from-env"\n')
    monkeypatch.setenv("KESTREL_CONFIG", str(env_config))

    loaded, source = config.load_config()

    assert loaded.general.default_model == "from-env"
    assert source == env_config


@pytest.mark.sanity
def test_explicit_path_beats_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given both an explicit path and $KESTREL_CONFIG, when load_config
    runs, then the explicit path wins."""
    env_config = tmp_path / "env-kestrel.toml"
    env_config.write_text('[general]\ndefault_model = "from-env"\n')
    monkeypatch.setenv("KESTREL_CONFIG", str(env_config))
    explicit_config = tmp_path / "explicit-kestrel.toml"
    explicit_config.write_text('[general]\ndefault_model = "from-explicit"\n')

    loaded, source = config.load_config(explicit_path=explicit_config)

    assert loaded.general.default_model == "from-explicit"
    assert source == explicit_config


def test_user_config_dir_used_when_no_higher_layer_exists(
    user_config_dir: Path,
) -> None:
    """Given only a user-config-dir kestrel.toml, when load_config runs,
    then that file is read even though it is the lowest file-backed layer.
    """
    (user_config_dir / "kestrel.toml").write_text(
        '[general]\ndefault_model = "from-user-dir"\n'
    )

    loaded, source = config.load_config()

    assert loaded.general.default_model == "from-user-dir"
    assert source == user_config_dir / "kestrel.toml"
