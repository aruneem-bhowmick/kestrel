"""Measure and classify Kestrel's own memory footprint against a ceiling.

Kestrel runs alongside Ollama on the same small, fixed-memory Jetson
board it targets, so a runaway session competing with the aux embedding
model for the same pool of RAM is a real failure mode -- not a
theoretical one -- and it fails ungracefully (the OOM killer, not a
clean error) if nothing warns first. `check_resources` is a pure
classifier over two already-measured RSS figures and a ceiling; the two
`measure_*` functions are the readings it classifies, kept separate so
each can be tested (and faked, by pointing `_PROC_ROOT` at a fake tree)
without the other. `unload_aux_model` is unrelated to the warning path
-- it is the one place this module reaches the network, freeing
Ollama's own GPU memory on request rather than warning about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

import litellm

from kestrel.registry.model import ModelEntry

# The Jetson Orin Nano board this project targets ships 8GB of shared
# CPU/GPU memory; after the OS and Ollama's own GPU-resident model share
# reserve their slice, roughly 5.2GB remains usable by Kestrel itself.
# This is a starting default, not a hardcoded assumption baked into the
# classification logic -- a caller measuring a different board is free
# to pass its own `ceiling_mb`.
_DEFAULT_CEILING_MB: Final[float] = 5200.0
_DEFAULT_WARN_THRESHOLD: Final[Decimal] = Decimal("0.85")

_KIB_PER_MB: Final[float] = 1024.0

_PROC_ROOT: Final[Path] = Path("/proc")
_OLLAMA_COMM_NAME: Final[str] = "ollama"
_OLLAMA_CMDLINE_SUBSTRING: Final[str] = "ollama serve"

_UNLOAD_KEEP_ALIVE: Final[int] = 0


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    """One resource-guard snapshot's own classification.

    Attributes:
        kestrel_rss_mb: This process's own resident memory, in MB.
        ollama_rss_mb: The `ollama` server process's own resident
            memory, in MB, or `None` when it could not be found or
            measured.
        ceiling_mb: The configured memory ceiling being checked against.
        warning: A one-line message when combined RSS is at or past the
            warn threshold of `ceiling_mb`; `None` otherwise.
    """

    kestrel_rss_mb: float
    ollama_rss_mb: float | None
    ceiling_mb: float
    warning: str | None


def _format_warning(
    combined_mb: float,
    ceiling_mb: float,
    warn_threshold: Decimal,
    *,
    ollama_measured: bool,
) -> str:
    """Render the one-line warning naming the combined figure and ceiling.

    Appends a note that Ollama's own usage is excluded when it could not
    be measured, so the warning never implies a false sense of precision
    about a number it did not actually see.
    """
    message = (
        f"combined RSS {combined_mb:.0f}MB is at or above "
        f"{warn_threshold:.0%} of the {ceiling_mb:.0f}MB ceiling"
    )
    if not ollama_measured:
        message += " (Ollama's own usage could not be measured and is excluded)"
    return message


def check_resources(
    *,
    kestrel_rss_mb: float,
    ollama_rss_mb: float | None,
    ceiling_mb: float = _DEFAULT_CEILING_MB,
    warn_threshold: Decimal = _DEFAULT_WARN_THRESHOLD,
) -> ResourceStatus:
    """Classify combined Kestrel+Ollama RSS against `ceiling_mb`.

    Pure and side-effect-free: the combined figure is `kestrel_rss_mb +
    (ollama_rss_mb or 0)`, and `warning` is set exactly when that
    combined figure is at or past `ceiling_mb * warn_threshold` -- an
    understated warning (one that excludes an unmeasured Ollama process
    entirely) is safer than one that guesses and gets it wrong.
    """
    combined_mb = kestrel_rss_mb + (ollama_rss_mb or 0.0)
    threshold_mb = ceiling_mb * float(warn_threshold)
    warning: str | None = None
    if combined_mb >= threshold_mb:
        warning = _format_warning(
            combined_mb,
            ceiling_mb,
            warn_threshold,
            ollama_measured=ollama_rss_mb is not None,
        )
    return ResourceStatus(
        kestrel_rss_mb=kestrel_rss_mb,
        ollama_rss_mb=ollama_rss_mb,
        ceiling_mb=ceiling_mb,
        warning=warning,
    )


def _looks_like_ollama(pid_dir: Path) -> bool:
    """Whether `pid_dir` (a `/proc/<pid>` entry) is an Ollama server process.

    Matches on `/proc/<pid>/comm` naming it exactly `"ollama"`, or on
    `/proc/<pid>/cmdline` containing `"ollama serve"` once its null-byte
    argument separators are normalized to spaces -- covering both a
    plain `ollama serve` invocation and one launched through a wrapper
    script whose own `comm` differs.
    """
    comm_path = pid_dir / "comm"
    try:
        comm = comm_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        comm = ""
    if comm == _OLLAMA_COMM_NAME:
        return True
    cmdline_path = pid_dir / "cmdline"
    try:
        raw_cmdline = cmdline_path.read_bytes()
    except OSError:
        return False
    cmdline = raw_cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return _OLLAMA_CMDLINE_SUBSTRING in cmdline


def _read_vmrss_mb(pid_dir: Path) -> float | None:
    """Parse the `VmRSS` line out of `/proc/<pid>/status`, in MB.

    Returns `None` -- never raises -- when the status file cannot be
    read (a permissions issue, or the process exiting between being
    found and being read) or does not carry a well-formed `VmRSS` line.
    """
    status_path = pid_dir / "status"
    try:
        lines = status_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return float(fields[1]) / _KIB_PER_MB
        except ValueError:
            return None
    return None


def measure_kestrel_rss_mb() -> float:
    """This process's own current resident memory, in MB.

    Reads `/proc/self/status`'s own `VmRSS` line through the same
    `_read_vmrss_mb` helper `measure_ollama_rss_mb` uses for Ollama's
    process -- `/proc/self` is the kernel's own always-current symlink
    to the calling process's `/proc/<pid>` entry. `VmRSS` names
    *current* resident memory, unlike `resource.getrusage()`'s own
    `ru_maxrss` field, which reports a high-water mark that never falls
    even after memory is freed and would leave a resource-ceiling
    warning stuck long after the pressure that triggered it is gone.
    Degrades to `0.0` -- never raises -- on any platform without a
    `/proc` filesystem, the same outcome `_read_vmrss_mb` already
    returns for a process it cannot read.
    """
    return _read_vmrss_mb(_PROC_ROOT / "self") or 0.0


def measure_ollama_rss_mb() -> float | None:
    """The first `/proc`-listed process that looks like Ollama's own
    resident memory, in MB.

    Returns `None` when no such process is found, `/proc` does not
    exist at all (any non-Linux platform), or the matching process's
    status file could not be read -- every one of those is a "could not
    measure" outcome a caller should treat identically, never a crash.
    """
    if not _PROC_ROOT.is_dir():
        return None
    try:
        pid_dirs = sorted(
            (entry for entry in _PROC_ROOT.iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
    except OSError:
        return None
    for pid_dir in pid_dirs:
        if _looks_like_ollama(pid_dir):
            return _read_vmrss_mb(pid_dir)
    return None


class UnloadAuxError(Exception):
    """`unload_aux_model` could not reach Ollama, or Ollama answered with
    a non-2xx status. `str(self)` names the remedy."""


async def unload_aux_model(*, entry: ModelEntry) -> None:
    """Ask Ollama to immediately unload `entry`'s model from GPU memory.

    POSTs `{"model": entry.provider_model, "keep_alive": 0}` to
    `{entry.endpoint}/api/generate` -- Ollama's own documented
    immediate-unload idiom: a generate request with `keep_alive: 0` and
    no `prompt` unloads the named model without generating anything.
    Reuses `litellm.module_level_aclient`, the same shared async HTTP
    client litellm's own Ollama embedding route already drives
    internally, rather than adding a direct `httpx` dependency to this
    module for one call site.

    Raises:
        UnloadAuxError: the request could not be sent (Ollama
            unreachable, a network failure) or the response status is
            not in `[200, 300)`.
    """
    try:
        await litellm.module_level_aclient.post(
            url=f"{entry.endpoint}/api/generate",
            json={"model": entry.provider_model, "keep_alive": _UNLOAD_KEEP_ALIVE},
        )
    except Exception as exc:
        raise UnloadAuxError(
            f"could not unload {entry.id!r} from Ollama at {entry.endpoint!r}: {exc}"
        ) from exc
