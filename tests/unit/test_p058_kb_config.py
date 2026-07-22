"""Unit tests for `kestrel.config.KbConfig`: the `[kb]` table's
defaults, its `extra="forbid"` guard, its field bounds, and its
partial-override semantics -- the same contract every other nested
config table in `kestrel.config` already carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config

pytestmark = [pytest.mark.p058, pytest.mark.unit]


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
def test_kb_config_defaults_match_the_documented_values() -> None:
    """Given no `[kb]` table at all, when `KbConfig` is constructed, then
    every field defaults to the value this feature's own documentation
    names for it."""
    kb = config.KbConfig()

    assert kb.enabled is True
    assert kb.top_k == 5
    assert kb.global_namespace is False


def test_kb_config_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """Given a `[kb]` table carrying a key outside its schema, when
    `load_config` runs, then it raises `ConfigError` naming that key."""
    (tmp_path / "kestrel.toml").write_text("[kb]\nbogus_key = true\n")

    with pytest.raises(config.ConfigError, match="bogus_key"):
        config.load_config()


def test_kb_config_top_k_below_minimum_raises_config_error(tmp_path: Path) -> None:
    """Given a `[kb]` table setting `top_k` below its own floor of 1,
    when `load_config` runs, then it raises `ConfigError`."""
    (tmp_path / "kestrel.toml").write_text("[kb]\ntop_k = 0\n")

    with pytest.raises(config.ConfigError, match="top_k"):
        config.load_config()


def test_kb_config_top_k_above_maximum_raises_config_error(tmp_path: Path) -> None:
    """Given a `[kb]` table setting `top_k` above its own ceiling of 50,
    when `load_config` runs, then it raises `ConfigError`."""
    (tmp_path / "kestrel.toml").write_text("[kb]\ntop_k = 51\n")

    with pytest.raises(config.ConfigError, match="top_k"):
        config.load_config()


def test_kb_config_partial_override_leaves_other_fields_at_default(
    tmp_path: Path,
) -> None:
    """Given a `kestrel.toml` overriding only `top_k`, when `load_config`
    runs, then that one field takes the override while the other two
    stay at their own defaults -- the same partial-override semantics
    every other nested config table already has."""
    (tmp_path / "kestrel.toml").write_text("[kb]\ntop_k = 10\n")

    loaded, _source = config.load_config()

    assert loaded.kb.top_k == 10
    assert loaded.kb.enabled is True
    assert loaded.kb.global_namespace is False


def test_kb_config_disabling_is_honored_by_load_config(tmp_path: Path) -> None:
    """Given a `[kb]` table setting `enabled = false`, when `load_config`
    runs, then the resulting config carries that override rather than
    the built-in default."""
    (tmp_path / "kestrel.toml").write_text("[kb]\nenabled = false\n")

    loaded, _source = config.load_config()

    assert loaded.kb.enabled is False
