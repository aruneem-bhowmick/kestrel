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

Nothing here repairs a broken environment; every check is read-only.
Two checks reach the network, both only when the caller explicitly opts
in with ``--live``: the endpoint probe (a tiny, budget-capped
completion) and the ``ollama`` check (a one-word, zero-cost embedding
call against the local Ollama backend). The ``resources`` check reads
this process's and Ollama's own resident memory and never reaches the
network at all.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kestrel.config import ConfigError, KestrelConfig, load_config
from kestrel.kb.embeddings import EmbeddingError, OllamaEmbeddingClient
from kestrel.managers.resource_guard import (
    check_resources,
    measure_kestrel_rss_mb,
    measure_ollama_rss_mb,
)
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
from kestrel.router.policy import resolve_model_id
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
    "tui",
    "ollama",
    "resources",
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
    except (SandboxUnavailableError, OSError) as exc:
        return CheckResult("sandbox", CheckStatus.FAIL, str(exc))
    if result.exit_code != 0:
        detail = f"smoke invocation exited {result.exit_code}: {result.stderr.strip()}"
        return CheckResult("sandbox", CheckStatus.FAIL, detail)
    return CheckResult("sandbox", CheckStatus.OK, "bwrap")


def _check_tui() -> CheckResult:
    """Confirm stdout is a real terminal.

    `kestrel` with no subcommand mounts a full-screen Textual cockpit,
    which has nowhere to draw into against a piped or redirected
    stdout -- this check answers, ahead of time, the same question
    `kestrel.cli.main`'s own guard asks right before it would otherwise
    try. Unconditional, exactly like `_check_sandbox`: it never depends
    on `config` or the registry resolving first, so it always runs and
    never reports `SKIP`.
    """
    if sys.stdout.isatty():
        return CheckResult("tui", CheckStatus.OK, "interactive")
    return CheckResult(
        "tui",
        CheckStatus.FAIL,
        "stdout is not a terminal; 'kestrel' (no subcommand) requires an "
        "interactive terminal -- use 'kestrel run \"<task>\" --repo PATH' "
        "instead",
    )


async def _drain_embedding_probe(registry: Registry, entry: ModelEntry) -> None:
    """Send one single-word embedding call to confirm Ollama answers,
    discarding the returned vector."""
    client = OllamaEmbeddingClient(registry)
    await client.embed(["ping"], model_id=entry.id)


async def _probe_ollama(registry: Registry, entry: ModelEntry) -> None:
    """Run :func:`_drain_embedding_probe` under the identical hard
    wall-clock bound :func:`_probe_endpoint` already applies to the
    chat-completion probe, for the same reason: an unresponsive backend
    must produce a bounded ``FAIL``, not a hung terminal.
    """
    await asyncio.wait_for(
        _drain_embedding_probe(registry, entry), timeout=_LIVE_PROBE_TIMEOUT_S
    )


def _check_ollama(
    registry: Registry, config: KestrelConfig, *, live: bool
) -> CheckResult:
    """Confirm the router's ``"embed"`` task class resolves to a reachable
    Ollama backend.

    Skipped with ``"pass --live"`` when ``live=False``, mirroring
    ``endpoint``'s own opt-in gate -- this check reaches a real network
    endpoint too. Otherwise resolves the ``"embed"`` task class against
    ``config.router.policy`` and probes it: a policy with no reachable
    ``"local"``-tagged entry falls back to the configured default model
    (never raising on its own), but a fallback that is not itself an
    Ollama-backed entry surfaces as an ``EmbeddingError`` from
    ``OllamaEmbeddingClient.embed``'s own backend check, which this
    function reports as ``FAIL`` rather than letting it escape.
    """
    if not live:
        return CheckResult("ollama", CheckStatus.SKIP, "pass --live")
    model_id = resolve_model_id(
        "embed",
        registry=registry,
        policy=config.router.policy.as_mapping(),
        fallback_model_id=config.general.default_model,
    )
    entry = registry.get(model_id)
    try:
        asyncio.run(_probe_ollama(registry, entry))
    except TimeoutError:
        return CheckResult(
            "ollama",
            CheckStatus.FAIL,
            f"no response within {_LIVE_PROBE_TIMEOUT_S:.0f}s",
        )
    except EmbeddingError as exc:
        return CheckResult("ollama", CheckStatus.FAIL, str(exc))
    return CheckResult("ollama", CheckStatus.OK, entry.id)


def _check_resources() -> CheckResult:
    """Report Kestrel's and Ollama's combined resident memory.

    Never ``FAIL``s -- a resource check that could itself crash
    ``kestrel doctor`` on an unusual platform (no ``/proc``, an
    unreadable status file) would defeat its own purpose, so both
    underlying measurements already degrade to a safe default rather
    than raising. A combined-RSS warning is folded into the detail
    string rather than changing ``status``: this check is advisory,
    matching the "warn," never "block," nature of a memory ceiling.
    """
    status = check_resources(
        kestrel_rss_mb=measure_kestrel_rss_mb(),
        ollama_rss_mb=measure_ollama_rss_mb(),
    )
    detail = f"kestrel={status.kestrel_rss_mb:.0f}MB"
    detail += (
        f" ollama={status.ollama_rss_mb:.0f}MB"
        if status.ollama_rss_mb is not None
        else " ollama=unknown"
    )
    detail += f" / ceiling={status.ceiling_mb:.0f}MB"
    if status.warning is not None:
        detail += f" -- {status.warning}"
    return CheckResult("resources", CheckStatus.OK, detail)


def run_doctor(config_path: Path | None, *, live: bool) -> list[CheckResult]:
    """Run every flight check, in order, and return all ten results.

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
    8. ``tui`` -- stdout is a real terminal, the same precondition
       `kestrel` (no subcommand) itself requires before mounting the
       cockpit.
    9. ``ollama`` -- only when ``live=True``: a one-word embedding call
       against the router's resolved ``"embed"`` model confirms Ollama
       answers. Skipped with ``"pass --live"`` when ``live=False``.
    10. ``resources`` -- Kestrel's and Ollama's combined resident memory
        against a fixed ceiling; always ``OK``, a high-memory reading is
        folded into the detail as an advisory warning rather than a
        failure.

    Checks 2 through 6 form a dependency chain: the first one to fail
    records its own ``FAIL``, and every check after it in the chain
    reports ``SKIP`` naming that same original check (not whichever check
    immediately precedes it), so the root cause is never obscured by a
    chain of "blocked by the previous line" indirection. Checks 7 and 8
    are unconditional and never join this chain. Check 9 rejoins the
    chain -- it needs ``registry``/``config`` resolved exactly like
    ``endpoint`` does -- reporting ``SKIP`` naming the original blocker
    when checks 2 through 5 failed upstream. Check 10 is unconditional
    and never reports anything but ``OK``.
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
    results.append(_check_tui())

    if blocking is not None:
        results.append(
            CheckResult("ollama", CheckStatus.SKIP, f"blocked by: {blocking}")
        )
    else:
        assert config is not None
        assert registry is not None
        results.append(_check_ollama(registry, config, live=live))

    results.append(_check_resources())

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
