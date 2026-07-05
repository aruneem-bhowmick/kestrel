"""Resolve and parse ``models.toml`` into a validated :class:`Registry`."""

from __future__ import annotations

import logging
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import ValidationError

from kestrel.registry.model import ModelEntry, Registry, RegistryError

logger = logging.getLogger("kestrel.registry")

_REGISTRY_FILENAME = "models.toml"
_PACKAGED_DEFAULT_PACKAGE = "kestrel.data"
_PACKAGED_DEFAULT_RESOURCE = "models.default.toml"


def _user_registry_path() -> Path:
    """Return the per-user registry file path (may or may not exist)."""
    return Path(platformdirs.user_config_dir("kestrel")) / _REGISTRY_FILENAME


def _parse_toml_file(path: Path) -> dict[str, Any]:
    """Parse ``path`` as TOML, wrapping decode failures as :class:`RegistryError`."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"{path}: invalid TOML syntax ({exc})") from exc


def _load_packaged_default() -> dict[str, Any]:
    """Parse the registry bundled with the package as the last-resort layer."""
    resource = resources.files(_PACKAGED_DEFAULT_PACKAGE).joinpath(
        _PACKAGED_DEFAULT_RESOURCE
    )
    return tomllib.loads(resource.read_bytes().decode("utf-8"))


def _render_entry_error(
    exc: ValidationError, *, label: str, entry_id: str | None
) -> str:
    """Render a validation failure naming the entry's position/id and the
    offending field(s), so a misconfigured entry is a one-screen fix."""
    heading = label if entry_id is None else f"{label} (id={entry_id!r})"
    lines = [f"invalid entry at {heading}"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        val_input = error["input"]
        is_sensitive = False
        for part in error["loc"]:
            part_str = str(part).lower()
            if any(sensitive in part_str for sensitive in ("key", "token", "secret")):
                is_sensitive = True
                break
        if is_sensitive:
            val_input = "[REDACTED]"
        lines.append(f"  {location}: {error['msg']} (got: {val_input!r})")
    return "\n".join(lines)


def _build_registry(data: dict[str, Any], *, source: Path | None) -> Registry:
    """Validate every ``[[models]]`` table and assemble the id-keyed
    mapping, rejecting duplicate ids as soon as one is found."""
    file_label = str(source) if source is not None else "packaged default registry"
    raw_entries = data.get("models", [])

    if not isinstance(raw_entries, list) or not all(isinstance(entry, dict) for entry in raw_entries):
        raise RegistryError(f"{file_label}: models must be an array of tables")

    entries: dict[str, ModelEntry] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry_id = raw_entry.get("id") if isinstance(raw_entry, dict) else None
        label = f"{file_label}: models[{index}]"
        try:
            entry = ModelEntry.model_validate(raw_entry)
        except ValidationError as exc:
            raise RegistryError(
                _render_entry_error(exc, label=label, entry_id=entry_id)
            ) from exc

        if entry.id in entries:
            raise RegistryError(f"{label}: duplicate model id '{entry.id}'")

        if entry.usd_per_mtok_cached > entry.usd_per_mtok_input:
            logger.warning(
                "%s: cached rate (%s) exceeds input rate (%s) for entry '%s'",
                file_label,
                entry.usd_per_mtok_cached,
                entry.usd_per_mtok_input,
                entry.id,
            )

        entries[entry.id] = entry

    return Registry(models=entries, source=source)


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate the model registry.

    Precedence (first hit wins): ``path`` (e.g. ``--config``-adjacent
    explicit override) > ``./models.toml`` > ``<platformdirs
    user_config_dir 'kestrel'>/models.toml`` > the packaged default
    (``kestrel/data/models.default.toml``, read via
    :mod:`importlib.resources`). Files never merge across layers --
    exactly one file (or the packaged default) is read. Every entry in
    that file is validated; the first error encountered is raised,
    rendered with the file, the entry's position/id, and the field(s)
    at fault.
    """
    source: Path | None

    if path is not None:
        if not path.is_file():
            raise RegistryError(f"Model registry file not found: {path}")
        source = path
    else:
        cwd_path = Path(_REGISTRY_FILENAME)
        user_path = _user_registry_path()
        if cwd_path.is_file():
            source = cwd_path.resolve()
        elif user_path.is_file():
            source = user_path
        else:
            source = None

    data = _load_packaged_default() if source is None else _parse_toml_file(source)
    return _build_registry(data, source=source)
