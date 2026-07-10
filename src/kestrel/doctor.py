"""Environment and configuration flight checks for ``kestrel doctor``.

Every check here answers one yes/no question about whether the current
environment can actually run Kestrel: is the interpreter new enough, does
the configuration file parse, does the model registry it points at parse,
does the configured default model exist in that registry, and is its
credential present. Checks 3 through 6 form a dependency chain -- each one
needs the object the previous one produced -- so a failure partway through
is reported once, at its source, and every check downstream of it is
marked skipped rather than re-deriving (and re-reporting) the same root
cause under a different name.

Nothing here repairs a broken environment; every check is read-only, and
the optional live endpoint probe is the only check that ever reaches the
network -- and only when the caller explicitly opts in, since it is the
only check with a real (if tiny, budget-capped) cost.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kestrel.config import ConfigError, KestrelConfig, load_config
from kestrel.provider.errors import AuthError, ProviderError
from kestrel.provider.litellm_client import LiteLLMClient, _require_api_key
from kestrel.registry.loader import load_registry
from kestrel.registry.model import (
    ModelEntry,
    Registry,
    RegistryError,
    UnknownModelError,
)
from kestrel.repl import sanitize_terminal
from kestrel.tools.sandbox import (
    SandboxUnavailableError,
    bwrap_available,
    run_sandboxed,
)

_MIN_PYTHON: tuple[int, int] = (3, 12)
_LIVE_PROBE_MESSAGE = "ping"
_LIVE_PROBE_MAX_TOKENS = 1
_LIVE_PROBE_TIMEOUT_S = 30.0
_SANDBOX_SMOKE_TIMEOUT_S = 5.0

_CHECK_NAMES: tuple[str, ...] = (
    "python-version",
    "config",
    "registry",
    "default-model",
    "api-key",
    "endpoint",
    "sandbox",
    "ollama",
)
_STATUS_WIDTH = max(len(status) for status in ("OK", "FAIL", "SKIP"))
_NAME_WIDTH = max(len(name) for name in _CHECK_NAMES)


class CheckStatus(StrEnum):
    """One flight check's outcome."""

    OK = "OK"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One flight check's name, outcome, and one-line explanation.

    Attributes:
        name: The check's stable identifier (e.g. ``"config"``), one of
            :data:`_CHECK_NAMES`.
        status: Whether the check passed, failed, or was skipped.
        detail: A one-line explanation of the outcome -- on ``FAIL``, the
            remedy; never a secret value, even for checks that concern
            credentials.
    """

    name: str
    status: CheckStatus
    detail: str


def _check_python_version(version_info: tuple[int, int]) -> CheckResult:
    """Check the running interpreter is at least Kestrel's minimum version."""
    if version_info >= _MIN_PYTHON:
        return CheckResult(
            "python-version", CheckStatus.OK, f"{version_info[0]}.{version_info[1]}"
        )
    return CheckResult(
        "python-version",
        CheckStatus.FAIL,
        f"requires >= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}, "
        f"found {version_info[0]}.{version_info[1]}",
    )


def _check_config(
    config_path: Path | None,
) -> tuple[CheckResult, KestrelConfig | None]:
    """Load the configuration, pairing the check result with the config
    itself so later checks can build on it without loading it twice."""
    try:
        config, source = load_config(config_path)
    except ConfigError as exc:
        return CheckResult("config", CheckStatus.FAIL, str(exc)), None
    detail = str(source) if source is not None else "built-in defaults"
    return CheckResult("config", CheckStatus.OK, detail), config


def _check_registry(config: KestrelConfig) -> tuple[CheckResult, Registry | None]:
    """Load the model registry ``config`` points at, pairing the check
    result with the registry itself so later checks can build on it."""
    try:
        registry = load_registry(config.paths.models_file)
    except RegistryError as exc:
        return CheckResult("registry", CheckStatus.FAIL, str(exc)), None
    count = len(registry.ids())
    noun = "model" if count == 1 else "models"
    return CheckResult("registry", CheckStatus.OK, f"{count} {noun}"), registry


def _check_default_model(
    config: KestrelConfig, registry: Registry
) -> tuple[CheckResult, ModelEntry | None]:
    """Resolve ``config``'s default model id against ``registry``."""
    try:
        entry = registry.get(config.general.default_model)
    except UnknownModelError as exc:
        return CheckResult("default-model", CheckStatus.FAIL, str(exc)), None
    return CheckResult("default-model", CheckStatus.OK, entry.id), entry


def _check_api_key(entry: ModelEntry) -> CheckResult:
    """Confirm the default model's credential environment variable is set
    and non-empty, without ever reading (or reporting) its value."""
    try:
        _require_api_key(entry)
    except AuthError as exc:
        return CheckResult("api-key", CheckStatus.FAIL, str(exc))
    return CheckResult("api-key", CheckStatus.OK, entry.api_key_env or "")


async def _drain_probe(registry: Registry, entry: ModelEntry) -> None:
    """Send one budget-capped completion to confirm the backend answers,
    discarding whatever it replies with."""
    client = LiteLLMClient(registry)
    async for _event in client.complete(
        [{"role": "user", "content": _LIVE_PROBE_MESSAGE}],
        None,
        entry.id,
        "high",
        stream=True,
        max_tokens=_LIVE_PROBE_MAX_TOKENS,
    ):
        pass


async def _probe_endpoint(registry: Registry, entry: ModelEntry) -> None:
    """Run :func:`_drain_probe` under a hard wall-clock bound.

    The client's own per-request timeout is tuned for a full REPL turn,
    not a diagnostic that is supposed to answer in seconds -- and, for a
    streaming call, a generic request timeout does not reliably bound the
    time between individual chunks. Wrapping the whole probe in
    :func:`asyncio.wait_for` guarantees ``kestrel doctor --live`` reports
    a result instead of hanging the terminal on an unresponsive or
    slow-drip backend.
    """
    await asyncio.wait_for(_drain_probe(registry, entry), timeout=_LIVE_PROBE_TIMEOUT_S)


def _check_endpoint(registry: Registry, entry: ModelEntry) -> CheckResult:
    """Run the live reachability probe against the default model.

    Only ever called when the caller passed ``live=True`` and every
    upstream check succeeded -- callers gate this themselves so a blocked
    or opted-out probe never reaches this function at all.
    """
    try:
        asyncio.run(_probe_endpoint(registry, entry))
    except TimeoutError:
        return CheckResult(
            "endpoint",
            CheckStatus.FAIL,
            f"no response within {_LIVE_PROBE_TIMEOUT_S:.0f}s",
        )
    except ProviderError as exc:
        return CheckResult("endpoint", CheckStatus.FAIL, f"{type(exc).__name__}: {exc}")
    return CheckResult("endpoint", CheckStatus.OK, f"{entry.backend}/{entry.id}")


def _check_sandbox() -> CheckResult:
    """Confirm `bwrap` is on `PATH` and a smoke invocation of it inside
    a real sandbox exits `0`, exercising the exact code path `execute`
    itself uses rather than merely checking the binary's presence.

    Unconditional -- unlike checks 2 through 6, this never depends on
    `config` or the registry resolving first, so it always runs and
    never reports `SKIP`.
    """
    if not bwrap_available():
        return CheckResult("sandbox", CheckStatus.FAIL, "bwrap not found on PATH")
    try:
        result = run_sandboxed(
            ["true"], repo_root=Path.cwd(), timeout_s=_SANDBOX_SMOKE_TIMEOUT_S
        )
    except SandboxUnavailableError as exc:
        return CheckResult("sandbox", CheckStatus.FAIL, str(exc))
    if result.exit_code != 0:
        detail = f"smoke invocation exited {result.exit_code}: {result.stderr.strip()}"
        return CheckResult("sandbox", CheckStatus.FAIL, detail)
    return CheckResult("sandbox", CheckStatus.OK, "bwrap")


def run_doctor(config_path: Path | None, *, live: bool) -> list[CheckResult]:
    """Run every flight check, in order, and return all eight results.

    Checks, in order:

    1. ``python-version`` -- the running interpreter is >= 3.12.
    2. ``config`` -- :func:`kestrel.config.load_config` succeeds (a
       missing file at every layer is a valid outcome too, resolving to
       built-in defaults); the detail names the resolved source.
    3. ``registry`` -- :func:`kestrel.registry.loader.load_registry`
       succeeds for the path ``config`` names; the detail counts entries.
    4. ``default-model`` -- ``config``'s configured default model id
       resolves against the loaded registry.
    5. ``api-key`` -- the resolved default model's ``api_key_env`` is set
       and non-empty; the detail names the variable, never its value.
    6. ``endpoint`` -- only when ``live=True``: a one-token-capped
       completion against the default model confirms the backend
       answers. Skipped with ``"pass --live"`` when ``live=False``.
    7. ``sandbox`` -- ``bwrap`` is on ``PATH`` and a smoke invocation of
       it exits ``0``; ``FAIL`` names whichever of those two failed.
    8. ``ollama`` -- always skipped; the Ollama backend does not exist in
       this codebase yet.

    Checks 2 through 6 form a dependency chain: the first one to fail
    records its own ``FAIL``, and every check after it in the chain
    reports ``SKIP`` naming that same original check (not whichever check
    immediately precedes it), so the root cause is never obscured by a
    chain of "blocked by the previous line" indirection. Checks 7 and 8
    are unconditional and never join this chain.
    """
    results: list[CheckResult] = [
        _check_python_version((sys.version_info.major, sys.version_info.minor))
    ]
    blocking: str | None = None

    config_result, config = _check_config(config_path)
    results.append(config_result)
    if config_result.status is CheckStatus.FAIL:
        blocking = config_result.name

    registry: Registry | None = None
    if blocking is not None:
        results.append(
            CheckResult("registry", CheckStatus.SKIP, f"blocked by: {blocking}")
        )
    else:
        assert config is not None
        registry_result, registry = _check_registry(config)
        results.append(registry_result)
        if registry_result.status is CheckStatus.FAIL:
            blocking = registry_result.name

    entry: ModelEntry | None = None
    if blocking is not None:
        results.append(
            CheckResult("default-model", CheckStatus.SKIP, f"blocked by: {blocking}")
        )
    else:
        assert config is not None
        assert registry is not None
        default_model_result, entry = _check_default_model(config, registry)
        results.append(default_model_result)
        if default_model_result.status is CheckStatus.FAIL:
            blocking = default_model_result.name

    if blocking is not None:
        results.append(
            CheckResult("api-key", CheckStatus.SKIP, f"blocked by: {blocking}")
        )
    else:
        assert entry is not None
        api_key_result = _check_api_key(entry)
        results.append(api_key_result)
        if api_key_result.status is CheckStatus.FAIL:
            blocking = api_key_result.name

    if blocking is not None:
        results.append(
            CheckResult("endpoint", CheckStatus.SKIP, f"blocked by: {blocking}")
        )
    elif not live:
        results.append(CheckResult("endpoint", CheckStatus.SKIP, "pass --live"))
    else:
        assert registry is not None
        assert entry is not None
        results.append(_check_endpoint(registry, entry))

    results.append(_check_sandbox())
    results.append(
        CheckResult("ollama", CheckStatus.SKIP, "the Ollama backend is not implemented")
    )

    return results


def format_check_line(result: CheckResult) -> str:
    """Render one aligned, terminal-safe line for a single check result.

    ``detail`` is run through :func:`kestrel.repl.sanitize_terminal` before
    printing -- it can carry a user-controlled filesystem path (an
    unresolved ``--config`` argument, for instance), and doctor output is
    routinely captured verbatim into CI logs.
    """
    return (
        f"{result.status.value:<{_STATUS_WIDTH}}  "
        f"{result.name:<{_NAME_WIDTH}}  "
        f"{sanitize_terminal(result.detail)}"
    )


def render_report(results: Sequence[CheckResult]) -> str:
    """Render every check result as the exact text ``kestrel doctor`` prints."""
    return "\n".join(format_check_line(result) for result in results) + "\n"


def all_checks_passed(results: Sequence[CheckResult]) -> bool:
    """Whether every result is ``OK`` or ``SKIP`` -- no ``FAIL`` among them."""
    return all(result.status is not CheckStatus.FAIL for result in results)
