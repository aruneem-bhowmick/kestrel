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


def test_unknown_key_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    """Given a kestrel.toml with a key outside the schema, when load_config
    runs, then it raises ConfigError naming that key."""
    (tmp_path / "kestrel.toml").write_text('[general]\nbogus_key = "x"\n')

    with pytest.raises(config.ConfigError, match="bogus_key"):
        config.load_config()


def test_wrong_type_raises_config_error_naming_key_and_expectation(
    tmp_path: Path,
) -> None:
    """Given a kestrel.toml where log_level is not one of the recognized
    strings, when load_config runs, then it raises ConfigError naming the
    key and the accepted values."""
    (tmp_path / "kestrel.toml").write_text("[general]\nlog_level = 3\n")

    with pytest.raises(config.ConfigError, match="log_level") as exc_info:
        config.load_config()

    assert "DEBUG" in str(exc_info.value)


def test_malformed_toml_syntax_raises_config_error(tmp_path: Path) -> None:
    """Given a kestrel.toml that is not valid TOML, when load_config runs,
    then it raises ConfigError describing the syntax problem."""
    (tmp_path / "kestrel.toml").write_text("this is not valid toml [[[")

    with pytest.raises(config.ConfigError, match="invalid TOML syntax"):
        config.load_config()


@pytest.mark.regression
def test_secret_shaped_key_is_rejected(tmp_path: Path) -> None:
    """Given a kestrel.toml containing a key that looks like a credential,
    when load_config runs, then it raises ConfigError instructing the user
    to use an environment variable instead -- regardless of case or of the
    exact key name variant used. This guards the no-secrets-on-disk rule
    against regressing.
    """
    (tmp_path / "kestrel.toml").write_text(
        '[general]\nAPI_KEY = "sk-do-not-put-me-here"\n'
    )

    with pytest.raises(
        config.ConfigError, match="Secrets belong in environment variables"
    ):
        config.load_config()


@pytest.mark.regression
def test_secret_shaped_key_inside_array_of_tables_is_rejected(
    tmp_path: Path,
) -> None:
    """Given a secret-shaped key nested inside an array of tables, when
    load_config runs, then it is still caught -- the secret scan walks
    lists, not just top-level tables."""
    (tmp_path / "kestrel.toml").write_text('[[general.servers]]\ntoken = "abc123"\n')

    with pytest.raises(
        config.ConfigError, match="Secrets belong in environment variables"
    ):
        config.load_config()


def test_missing_explicit_path_raises_config_error(tmp_path: Path) -> None:
    """Given an explicit path that does not exist, when load_config runs,
    then it raises ConfigError rather than silently falling back to a
    lower-precedence layer."""
    missing = tmp_path / "does-not-exist.toml"

    with pytest.raises(config.ConfigError, match="not found"):
        config.load_config(explicit_path=missing)


def test_missing_env_var_path_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given an environment variable config path that does not exist, when
    load_config runs, then it raises ConfigError rather than silently falling
    back to a lower-precedence layer."""
    missing = tmp_path / "does-not-exist.toml"
    monkeypatch.setenv("KESTREL_CONFIG", str(missing))

    with pytest.raises(config.ConfigError, match="not found"):
        config.load_config()


def test_managers_approval_allowlist_defaults_to_empty() -> None:
    """Given a kestrel.toml with no [managers.approval] table at all,
    when load_config runs, then the allowlist defaults to an empty
    tuple rather than requiring the table to be spelled out."""
    loaded, _source = config.load_config()

    assert loaded.managers.approval.allowlist == ()


def test_managers_approval_allowlist_round_trips(tmp_path: Path) -> None:
    """Given a kestrel.toml naming a [managers.approval] allowlist, when
    load_config runs, then every listed kind survives intact and in
    order."""
    (tmp_path / "kestrel.toml").write_text(
        '[managers.approval]\nallowlist = ["delete", "chmod"]\n'
    )

    loaded, _source = config.load_config()

    assert loaded.managers.approval.allowlist == ("delete", "chmod")


def test_managers_approval_rejects_an_unrecognized_kind(tmp_path: Path) -> None:
    """Given an allowlist entry that is not one of the five recognized
    DestructiveKinds, when load_config runs, then it raises ConfigError
    naming the offending field and the accepted values, rather than
    silently accepting it."""
    (tmp_path / "kestrel.toml").write_text(
        '[managers.approval]\nallowlist = ["not_a_real_kind"]\n'
    )

    with pytest.raises(
        config.ConfigError, match="managers.approval.allowlist"
    ) as exc_info:
        config.load_config()

    assert "'delete'" in str(exc_info.value)


def test_managers_approval_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """Given a [managers.approval] table carrying a key outside its
    schema, when load_config runs, then it raises ConfigError naming
    that key -- the same extra="forbid" guard every other config table
    enforces applies here too."""
    (tmp_path / "kestrel.toml").write_text('[managers.approval]\nbogus_key = "x"\n')

    with pytest.raises(config.ConfigError, match="bogus_key"):
        config.load_config()


@pytest.mark.regression
def test_managers_approval_secret_shaped_key_is_rejected(tmp_path: Path) -> None:
    """Given a secret-shaped key nested under [managers.approval], when
    load_config runs, then the secret scan still catches it before
    schema validation runs -- the new table doesn't create a blind spot
    in the no-secrets-on-disk guard."""
    (tmp_path / "kestrel.toml").write_text(
        '[managers.approval]\napi_key = "sk-do-not-put-me-here"\n'
    )

    with pytest.raises(
        config.ConfigError, match="Secrets belong in environment variables"
    ):
        config.load_config()
