"""Load and validate Kestrel's global configuration file.

Configuration is resolved from exactly one file, selected by a fixed
precedence (see :func:`load_config`); files never merge across layers, and
falling back to built-in defaults is always a valid outcome. Because
secrets must never be committed to a file on disk, every loaded file is
scanned for credential-shaped keys before schema validation runs, so a
stray API key produces a targeted error rather than being silently
accepted or misreported as an unrelated schema problem.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import BaseModel, ConfigDict, ValidationError

from kestrel.managers.approval import DestructiveKind

logger = logging.getLogger("kestrel.config")

_CONFIG_FILENAME = "kestrel.toml"
_CONFIG_ENV_VAR = "KESTREL_CONFIG"

_SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|token|secret|password)", re.IGNORECASE)


class GeneralConfig(BaseModel):
    """Cross-cutting settings that apply regardless of the active model.

    Attributes:
        default_model: Registry id used when no ``--model`` flag or
            ``/model`` command has selected another entry.
        log_level: Minimum severity of log records emitted to stderr.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_model: str = "glm-5.2"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PathsConfig(BaseModel):
    """Filesystem overrides for locating other Kestrel data files.

    Attributes:
        models_file: Explicit path to a model registry file, overriding
            the registry loader's own default search.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    models_file: Path | None = None


class ApprovalConfig(BaseModel):
    """Per-repo pre-approval for destructive tool actions.

    Attributes:
        allowlist: `DestructiveKind`s pre-approved for every request in
            a session, so `ApprovalManager.check` never prompts for one
            of these regardless of what the interactive decision
            function would have said.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowlist: tuple[DestructiveKind, ...] = ()


class BudgetConfig(BaseModel):
    """Per-repo USD budget caps and soft-threshold fraction.

    Attributes:
        session_usd: Cap on this task/session's own spend, in USD.
            `None` means no cap for this scope.
        day_usd: Cap on spend across the current UTC day, in USD.
            `None` means no cap for this scope.
        month_usd: Cap on spend across the current UTC month, in USD.
            `None` means no cap for this scope.
        soft_threshold: Fraction of a cap counted as the soft
            (warn/degrade) boundary, e.g. 0.8 means the soft threshold
            trips at 80% of whichever cap it belongs to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_usd: Decimal | None = None
    day_usd: Decimal | None = None
    month_usd: Decimal | None = None
    soft_threshold: Decimal = Decimal("0.8")


class ManagersConfig(BaseModel):
    """Settings for Kestrel's own runtime-state managers.

    Attributes:
        approval: Per-repo configuration for the destructive-action
            approval gate.
        budget: Per-repo configuration for session/day/month USD
            budget caps.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval: ApprovalConfig = ApprovalConfig()
    budget: BudgetConfig = BudgetConfig()


class KestrelConfig(BaseModel):
    """The fully validated, immutable global configuration object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    general: GeneralConfig = GeneralConfig()
    paths: PathsConfig = PathsConfig()
    managers: ManagersConfig = ManagersConfig()


class ConfigError(Exception):
    """Carries file path + underlying pydantic/toml error, rendered
    as a one-screen human message (file, table, key, expected type)."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        """Store the rendered message alongside the file it came from."""
        super().__init__(message)
        self.path = path


def _check_for_secrets(
    node: Any, *, source: Path, key_path: tuple[str, ...] = ()
) -> None:
    """Recursively reject any table key that looks like a credential.

    Walks the raw, not-yet-validated TOML tree (dicts, and lists of dicts
    for array-of-tables) so a secret-shaped key is caught before schema
    validation runs -- a secret-specific message should win over a generic
    "unknown field" complaint when both would otherwise apply.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if _SECRET_KEY_PATTERN.search(key):
                dotted = ".".join((*key_path, key))
                raise ConfigError(
                    f"{source}: key '{dotted}' looks like a secret. Secrets "
                    "belong in environment variables, not kestrel.toml.",
                    path=source,
                )
            _check_for_secrets(value, source=source, key_path=(*key_path, key))
    elif isinstance(node, list):
        for item in node:
            _check_for_secrets(item, source=source, key_path=key_path)


def _parse_toml_file(path: Path) -> dict[str, Any]:
    """Parse ``path`` as TOML, wrapping decode failures as :class:`ConfigError`."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML syntax ({exc})", path=path) from exc


def _render_validation_error(exc: ValidationError, source: Path) -> str:
    """Render a pydantic validation failure as a one-screen message naming
    the file, the offending table/key, and what was expected there."""
    lines = [f"{source}: invalid configuration"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def _build_config(data: dict[str, Any], *, source: Path) -> KestrelConfig:
    """Validate a parsed TOML tree into a :class:`KestrelConfig`, wrapping
    any pydantic validation failure as a :class:`ConfigError`."""
    try:
        return KestrelConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_render_validation_error(exc, source), path=source) from exc


def _user_config_path() -> Path:
    """Return the per-user config file path (may or may not exist)."""
    return Path(platformdirs.user_config_dir("kestrel")) / _CONFIG_FILENAME


def load_config(
    explicit_path: Path | None = None,
) -> tuple[KestrelConfig, Path | None]:
    """Precedence (first hit wins): explicit_path (--config) >
    $KESTREL_CONFIG > ./kestrel.toml > <platformdirs user_config_dir
    'kestrel'>/kestrel.toml > built-in defaults (returns (config, None)).
    Files do NOT merge across layers; exactly one file is read.
    """
    source: Path | None
    layer: str

    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(
                f"Config file not found: {explicit_path}", path=explicit_path
            )
        source, layer = explicit_path, "--config"
    else:
        source, layer = None, ""

        env_value = os.environ.get(_CONFIG_ENV_VAR)
        if env_value:
            env_path = Path(env_value)
            if not env_path.is_file():
                raise ConfigError(f"Config file not found: {env_path}", path=env_path)
            source, layer = env_path, f"${_CONFIG_ENV_VAR}"

        if source is None:
            cwd_path = Path(_CONFIG_FILENAME)
            if cwd_path.is_file():
                source, layer = cwd_path.resolve(), "./kestrel.toml"

        if source is None:
            user_path = _user_config_path()
            if user_path.is_file():
                source, layer = user_path, "user config directory"

    if source is None:
        logger.debug("kestrel.toml resolved via builtin defaults")
        return KestrelConfig(), None

    data = _parse_toml_file(source)
    _check_for_secrets(data, source=source)
    validated = _build_config(data, source=source)
    logger.debug("kestrel.toml resolved via %s: %s", layer, source)
    return validated, source
