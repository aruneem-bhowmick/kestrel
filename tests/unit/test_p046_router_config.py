"""Unit tests for `kestrel.config.RouterPolicyConfig`/`RouterConfig`:
the `[router.policy]` table's defaults, its `extra="forbid"` guard, and
its partial-override semantics -- the same contract every other nested
config table in `kestrel.config` already carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel import config

pytestmark = [pytest.mark.p046, pytest.mark.unit]


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
    """Chdir into an empty directory, clear ``$KESTREL_CONFIG``, and
    point the user-config-dir lookup at an empty temp directory so
    every test starts with no ambient config layers and cannot pollute
    (or be polluted by) the developer's real machine."""
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )


@pytest.mark.sanity
def test_router_policy_defaults_match_the_documented_values() -> None:
    """Given no `[router.policy]` table at all, when `RouterPolicyConfig`
    is constructed, then every field defaults to the tag this feature's
    own documentation names for it."""
    policy = config.RouterPolicyConfig()

    assert policy.plan == "planner"
    assert policy.execute == "executor"
    assert policy.critique == "cheap"
    assert policy.trivial == "cheap"
    assert policy.embed == "local"


def test_router_policy_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """Given a `[router.policy]` table carrying a key outside its
    schema, when `load_config` runs, then it raises `ConfigError`
    naming that key -- the same `extra="forbid"` guard every other
    config table enforces applies here too."""
    (tmp_path / "kestrel.toml").write_text('[router.policy]\nbogus_key = "cheap"\n')

    with pytest.raises(config.ConfigError, match="bogus_key"):
        config.load_config()


def test_router_policy_rejects_an_unrecognized_tag(tmp_path: Path) -> None:
    """Given a `[router.policy]` field set to a value outside the
    registry's own `Tag` literal, when `load_config` runs, then it
    raises `ConfigError` naming the offending field."""
    (tmp_path / "kestrel.toml").write_text(
        '[router.policy]\ncritique = "not_a_real_tag"\n'
    )

    with pytest.raises(config.ConfigError, match="critique"):
        config.load_config()


def test_router_policy_partial_override_leaves_other_fields_at_default(
    tmp_path: Path,
) -> None:
    """Given a `kestrel.toml` overriding only `critique`, when
    `load_config` runs, then that one field takes the override while
    the other four stay at their own defaults -- the same partial-
    override semantics every other nested config table already has."""
    (tmp_path / "kestrel.toml").write_text('[router.policy]\ncritique = "local"\n')

    loaded, _source = config.load_config()

    assert loaded.router.policy.critique == "local"
    assert loaded.router.policy.plan == "planner"
    assert loaded.router.policy.execute == "executor"
    assert loaded.router.policy.trivial == "cheap"
    assert loaded.router.policy.embed == "local"


def test_router_policy_as_mapping_matches_pydantic_fields() -> None:
    """Given a `RouterPolicyConfig` with one field overridden, when
    `as_mapping()` is called, then the returned mapping carries the
    override alongside every unmodified field's own default."""
    policy = config.RouterPolicyConfig(embed="cheap")

    assert policy.as_mapping() == {
        "plan": "planner",
        "execute": "executor",
        "critique": "cheap",
        "trivial": "cheap",
        "embed": "cheap",
    }
