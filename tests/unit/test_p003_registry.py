"""Tests for the model registry schema, loader, and validators."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.registry import RegistryError, UnknownModelError, load_registry
from kestrel.registry import loader as registry_loader

pytestmark = [pytest.mark.p003, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent
    / "golden"
    / "p003_registry_normalized.golden"
)


def _minimal_entry_toml(model_id: str) -> str:
    """Render a `[[models]]` table with every required field filled in,
    for tests that only care about precedence, not schema content."""
    return (
        "[[models]]\n"
        f'id = "{model_id}"\n'
        'backend = "openrouter"\n'
        'provider_model = "test/model"\n'
        "context_window = 1000\n"
        "max_output = 100\n"
        "usd_per_mtok_input = 1.0\n"
        "usd_per_mtok_output = 2.0\n"
        "usd_per_mtok_cached = 0.5\n"
        "supports_tools = true\n"
        "supports_cache = false\n"
    )


@pytest.fixture
def user_registry_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user config
    directory, so tests never touch (or depend on) the real home directory.
    """
    return tmp_path_factory.mktemp("userregistry")


@pytest.fixture(autouse=True)
def _isolated_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_registry_dir: Path
) -> None:
    """Chdir into an empty directory and point the user-config-dir lookup
    at an empty temp directory, so every test starts with no ambient
    models.toml layers and cannot pollute (or be polluted by) the
    developer's real machine.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        registry_loader.platformdirs,
        "user_config_dir",
        lambda appname: str(user_registry_dir),  # noqa: ARG005
    )


@pytest.mark.sanity
def test_packaged_default_loads_both_ids_with_decimal_rates() -> None:
    """Given no models.toml in any layer, when load_registry runs, then
    the packaged default registry loads with both GLM-5.2 routes and
    Decimal-typed rates."""
    registry = load_registry()

    assert registry.ids() == ["glm-5.2", "glm-5.2-zai"]
    for model_id in registry.ids():
        entry = registry.get(model_id)
        assert isinstance(entry.usd_per_mtok_input, Decimal)
        assert isinstance(entry.usd_per_mtok_output, Decimal)
        assert isinstance(entry.usd_per_mtok_cached, Decimal)


@pytest.mark.sanity
def test_float_rate_loads_as_exact_decimal() -> None:
    """Given a packaged entry with rate `0.60`, when it is loaded, then it
    becomes exactly Decimal("0.6") rather than the binary float's true
    (imprecise) value -- confirming rates round-trip through Decimal
    without ever converting a float straight into Decimal."""
    entry = load_registry().get("glm-5.2")

    assert entry.usd_per_mtok_input == Decimal("0.6")
    assert entry.usd_per_mtok_output == Decimal("2.2")
    assert entry.usd_per_mtok_cached == Decimal("0.11")


@pytest.mark.sanity
def test_cwd_registry_wins_over_user_config_dir(
    tmp_path: Path, user_registry_dir: Path
) -> None:
    """Given both a ./models.toml and a user-config-dir models.toml, when
    load_registry runs, then the cwd file wins and the layers are not
    merged."""
    (user_registry_dir / "models.toml").write_text(_minimal_entry_toml("from-user-dir"))
    (tmp_path / "models.toml").write_text(_minimal_entry_toml("from-cwd"))

    registry = load_registry()

    assert registry.ids() == ["from-cwd"]
    assert registry.source == tmp_path / "models.toml"


@pytest.mark.sanity
def test_explicit_path_beats_cwd_registry(tmp_path: Path) -> None:
    """Given both an explicit path and a ./models.toml, when load_registry
    runs, then the explicit path wins."""
    (tmp_path / "models.toml").write_text(_minimal_entry_toml("from-cwd"))
    explicit_path = tmp_path / "explicit-models.toml"
    explicit_path.write_text(_minimal_entry_toml("from-explicit"))

    registry = load_registry(explicit_path)

    assert registry.ids() == ["from-explicit"]
    assert registry.source == explicit_path


def test_user_config_dir_used_when_no_higher_layer_exists(
    user_registry_dir: Path,
) -> None:
    """Given only a user-config-dir models.toml, when load_registry runs,
    then that file is read even though it is the lowest file-backed layer
    (still above the packaged default)."""
    (user_registry_dir / "models.toml").write_text(_minimal_entry_toml("from-user-dir"))

    registry = load_registry()

    assert registry.ids() == ["from-user-dir"]
    assert registry.source == user_registry_dir / "models.toml"


def test_missing_explicit_path_raises_registry_error(tmp_path: Path) -> None:
    """Given an explicit path that does not exist, when load_registry
    runs, then it raises RegistryError rather than silently falling back
    to a lower-precedence layer."""
    missing = tmp_path / "does-not-exist.toml"

    with pytest.raises(RegistryError, match="not found"):
        load_registry(missing)


def test_malformed_toml_syntax_raises_registry_error(tmp_path: Path) -> None:
    """Given a models.toml that is not valid TOML, when load_registry
    runs, then it raises RegistryError describing the syntax problem."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text("this is not valid toml [[[")

    with pytest.raises(RegistryError, match="invalid TOML syntax"):
        load_registry(registry_path)


@pytest.mark.acceptance
def test_missing_required_field_names_file_entry_and_field(tmp_path: Path) -> None:
    """Given a models.toml entry with an id but missing a required field,
    when load_registry runs, then RegistryError names the file, the
    entry's id, and the missing field -- the "helpful errors on
    misconfiguration" requirement made concrete."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        "[[models]]\n"
        'id = "incomplete"\n'
        'backend = "openrouter"\n'
        'provider_model = "test/model"\n'
        "max_output = 100\n"
        "usd_per_mtok_input = 1.0\n"
        "usd_per_mtok_output = 2.0\n"
        "usd_per_mtok_cached = 0.5\n"
        "supports_tools = true\n"
        "supports_cache = false\n"
    )

    with pytest.raises(RegistryError) as exc_info:
        load_registry(registry_path)

    message = str(exc_info.value)
    assert str(registry_path) in message
    assert "incomplete" in message
    assert "context_window" in message


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    """Given a models.toml entry with a key outside the schema, when
    load_registry runs, then it raises RegistryError naming that key."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        _minimal_entry_toml("extra-field").rstrip("\n") + '\nbogus_key = "x"\n'
    )

    with pytest.raises(RegistryError, match="bogus_key"):
        load_registry(registry_path)


def test_sensitive_field_redaction(tmp_path: Path) -> None:
    """Given an entry with an extra/invalid field containing key/token/secret,
    when validation fails, the error message redacts the raw value."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        _minimal_entry_toml("sensitive-extra").rstrip("\n")
        + '\napi_key = "super-secret-value-123"\n'
    )

    with pytest.raises(RegistryError) as exc_info:
        load_registry(registry_path)

    message = str(exc_info.value)
    assert "api_key" in message
    assert "super-secret-value-123" not in message
    assert "[REDACTED]" in message


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    """Given two entries sharing the same id, when load_registry runs,
    then it raises RegistryError naming the duplicated id."""
    entry = _minimal_entry_toml("dup")
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(entry + "\n" + entry)

    with pytest.raises(RegistryError, match="dup"):
        load_registry(registry_path)


def test_models_field_not_array_of_tables(tmp_path: Path) -> None:
    """Given a models.toml where models is not an array of tables, when
    load_registry runs, then it raises RegistryError."""
    registry_path = tmp_path / "models.toml"

    # Try with a dictionary (table instead of array of tables)
    registry_path.write_text("[models]\nid = 'foo'\n")
    with pytest.raises(RegistryError, match="models must be an array of tables"):
        load_registry(registry_path)

    # Try with a string
    registry_path.write_text("models = 'not-a-list'\n")
    with pytest.raises(RegistryError, match="models must be an array of tables"):
        load_registry(registry_path)


def test_zai_backend_without_endpoint_is_rejected(tmp_path: Path) -> None:
    """Given a "zai" entry with no endpoint, when load_registry runs, then
    it raises RegistryError -- direct backends cannot resolve a base URL
    on their own."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        "[[models]]\n"
        'id = "zai-no-endpoint"\n'
        'backend = "zai"\n'
        'provider_model = "glm-5.2"\n'
        "context_window = 1000\n"
        "max_output = 100\n"
        "usd_per_mtok_input = 1.0\n"
        "usd_per_mtok_output = 2.0\n"
        "usd_per_mtok_cached = 0.5\n"
        "supports_tools = true\n"
        "supports_cache = false\n"
    )

    with pytest.raises(RegistryError, match="endpoint"):
        load_registry(registry_path)


@pytest.mark.regression
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.z.ai/api/coding/paas/v4",
        "https://api.z.ai/api/CODING/paas/v4",
        "https://api.z.ai/API/coding/paas/v4",
        "https://api.z.ai/api/Coding/paas/v4",
    ],
)
def test_coding_plan_endpoint_is_rejected(tmp_path: Path, endpoint: str) -> None:
    """Given a "zai" entry whose endpoint targets the Z.ai Coding-Plan
    route, when load_registry runs, then it is rejected -- the registry
    must never be able to express a Coding-Plan endpoint, since that
    quota is contractually restricted to recognized coding tools and
    Kestrel is a custom application. This guards the ToS guard against
    regressing."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        "[[models]]\n"
        'id = "coding-plan"\n'
        'backend = "zai"\n'
        'provider_model = "glm-5.2"\n'
        f'endpoint = "{endpoint}"\n'
        "context_window = 1000\n"
        "max_output = 100\n"
        "usd_per_mtok_input = 1.0\n"
        "usd_per_mtok_output = 2.0\n"
        "usd_per_mtok_cached = 0.5\n"
        "supports_tools = true\n"
        "supports_cache = false\n"
    )

    with pytest.raises(RegistryError, match="Coding-Plan"):
        load_registry(registry_path)


def test_get_unknown_model_id_lists_available_ids() -> None:
    """Given a loaded registry, when Registry.get is called with an id
    that does not exist, then it raises UnknownModelError listing every
    id that is actually available."""
    registry = load_registry()

    with pytest.raises(UnknownModelError) as exc_info:
        registry.get("nope")

    message = str(exc_info.value)
    assert "nope" in message
    assert "glm-5.2" in message
    assert "glm-5.2-zai" in message


def test_invalid_tag_is_rejected(tmp_path: Path) -> None:
    """Given an entry tagged with a value outside the recognized set,
    when load_registry runs, then it raises RegistryError naming the
    rejected tag."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        _minimal_entry_toml("bad-tag").rstrip("\n") + '\ntags = ["not-a-real-tag"]\n'
    )

    with pytest.raises(RegistryError, match="not-a-real-tag"):
        load_registry(registry_path)


def test_cached_rate_above_input_rate_logs_warning_but_still_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Given an entry whose cached rate exceeds its input rate, when
    load_registry runs, then the entry still loads (a provider is free to
    price cache reads above input) but a warning is logged, since this
    combination is almost always a pricing mistake worth flagging."""
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        "[[models]]\n"
        'id = "pricey-cache"\n'
        'backend = "openrouter"\n'
        'provider_model = "test/model"\n'
        "context_window = 1000\n"
        "max_output = 100\n"
        "usd_per_mtok_input = 1.0\n"
        "usd_per_mtok_output = 2.0\n"
        "usd_per_mtok_cached = 5.0\n"
        "supports_tools = true\n"
        "supports_cache = false\n"
    )

    with caplog.at_level("WARNING", logger="kestrel.registry"):
        registry = load_registry(registry_path)

    assert registry.ids() == ["pricey-cache"]
    assert "exceeds input rate" in caplog.text


@pytest.mark.regression
def test_packaged_default_registry_matches_golden_snapshot() -> None:
    """The packaged default registry, normalized to sorted JSON, must
    match a pinned snapshot byte-for-byte -- an accidental change to the
    schema or the shipped defaults shows up as a diff here instead of
    surfacing later as a silent behavior change."""
    registry = load_registry()

    normalized = json.dumps(
        json.loads(registry.model_dump_json()), indent=2, sort_keys=True
    )

    assert normalized + "\n" == _GOLDEN_FILE.read_text()
