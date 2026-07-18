"""Unit tests for `kestrel.config.SelfCritiqueConfig`: the
`[managers.self_critique]` table's default and its `extra="forbid"`
guard -- the same contract every other nested config table in
`kestrel.config` already carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config

pytestmark = [pytest.mark.p047, pytest.mark.unit]


@pytest.fixture
def user_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user config
    directory, so tests never touch (or depend on) the real home
    directory."""
    return tmp_path_factory.mktemp("userconfig")


@pytest.fixture(autouse=True)
def _isolated_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_config_dir: Path
) -> None:
    """Chdir into an empty directory, clear `$KESTREL_CONFIG`, and point
    the user-config-dir lookup at an empty temp directory so every test
    starts with no ambient config layers and cannot pollute (or be
    polluted by) the developer's real machine."""
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )


@pytest.mark.sanity
def test_self_critique_defaults_to_enabled() -> None:
    """Given no `[managers.self_critique]` table at all, when
    `SelfCritiqueConfig` is constructed, then `enabled` defaults to
    `True` -- self-critique is skippable by config, but on by default.
    """
    assert config.SelfCritiqueConfig().enabled is True


def test_self_critique_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """Given a `[managers.self_critique]` table carrying a key outside
    its schema, when `load_config` runs, then it raises `ConfigError`
    naming that key."""
    (tmp_path / "kestrel.toml").write_text(
        "[managers.self_critique]\nbogus_key = true\n"
    )

    with pytest.raises(config.ConfigError, match="bogus_key"):
        config.load_config()


def test_self_critique_can_be_disabled_via_config(tmp_path: Path) -> None:
    """Given a `[managers.self_critique]` table setting `enabled = false`,
    when `load_config` runs, then the resulting config carries that
    override rather than the built-in default."""
    (tmp_path / "kestrel.toml").write_text(
        "[managers.self_critique]\nenabled = false\n"
    )

    loaded, _ = config.load_config()

    assert loaded.managers.self_critique.enabled is False
