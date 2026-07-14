"""A `bubblewrap`-contained subprocess runner.

`run_sandboxed` is the one seam through which Kestrel ever executes an
arbitrary command on a caller's behalf: every invocation runs under
`bwrap`, confined to a read-only view of the filesystem plus read-write
access to `repo_root`, a fresh scratch directory, and any explicitly
named extra paths, with its network namespace unshared unless the
caller opts in. This module knows nothing about tool schemas, framing,
or the agent loop -- `kestrel.tools.execute` is its first caller, and
`kestrel.doctor` calls straight into it for the `sandbox` flight check,
so both share the exact same containment path rather than the doctor
check merely asserting a binary's presence.

`bwrap` is required; there is no fallback to an unsandboxed subprocess
or to another containment tool when it is missing. `bwrap_available`
exists precisely so a caller can check that before deciding what to do
about its absence, and `run_sandboxed` itself refuses to run anything
at all rather than quietly executing uncontained.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_BWRAP_BIN: Final[str] = "bwrap"


class SandboxUnavailableError(Exception):
    """`bwrap` is not on `PATH`, or cannot be invoked at all.

    Raised by :func:`run_sandboxed` before it attempts to spawn
    anything -- a caller never gets back a `SandboxResult` for a run
    that could not even start.
    """


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """One sandboxed command's captured outcome.

    Attributes:
        stdout: Everything the command wrote to standard output,
            decoded as UTF-8 with invalid bytes replaced.
        stderr: Everything the command wrote to standard error, decoded
            the same way.
        exit_code: The command's exit status, or `-1` when `timed_out`
            is `True` (the command never exited on its own).
        timed_out: Whether the command was killed for exceeding its
            `timeout_s` bound rather than exiting on its own.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def bwrap_available() -> bool:
    """Whether `bwrap` can be found on `PATH`.

    This is a presence check only -- it does not confirm `bwrap` can
    actually create a sandbox on the current kernel (e.g. unprivileged
    user namespaces disabled), which only a real invocation can prove.
    """
    return shutil.which(_BWRAP_BIN) is not None


def _decode_partial(value: str | bytes | None) -> str:
    """Normalize a `subprocess.TimeoutExpired` attribute -- `None`,
    already-decoded `str`, or raw `bytes` depending on platform and
    Python version -- into the `str` this module always returns."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _build_bwrap_argv(
    cmd: Sequence[str],
    *,
    repo_root: Path,
    scratch_dir: Path,
    allow_network: bool,
    extra_rw_paths: Sequence[Path],
) -> list[str]:
    """Build the full `bwrap` argv for `cmd`: a read-only bind of the
    whole filesystem root, a freshly mounted `/dev` layered on top of
    it, read-write binds of `repo_root`, `scratch_dir`, and each of
    `extra_rw_paths`, the sandboxed process dying if its parent does,
    the network namespace unshared unless `allow_network`, and the
    working directory set to `repo_root` -- followed by `cmd` itself,
    untouched, so no shell ever interprets it.

    `--dev /dev` is load-bearing, not cosmetic: `--ro-bind / /` alone
    carries over whatever `/dev` entries the host's root happens to
    expose at that path, which is not guaranteed to include a working
    `/dev/urandom` in every environment. Any sandboxed command that
    starts a CPython interpreter -- `pytest`, most obviously -- calls
    `getrandom()`/reads `/dev/urandom` to seed hash randomization at
    startup, and dies immediately with `_Py_HashRandomization_Init:
    failed to get random numbers to initialize Python` when that isn't
    available; a plain command like `true` never touches it and always
    looks fine. `--dev /dev` gives every sandboxed process bwrap's own
    minimal, always-functional device set instead of depending on the
    host's."""
    argv = [
        _BWRAP_BIN,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--bind",
        str(repo_root),
        str(repo_root),
        "--bind",
        str(scratch_dir),
        str(scratch_dir),
    ]
    for path in extra_rw_paths:
        argv += ["--bind", str(path), str(path)]
    argv.append("--die-with-parent")
    if not allow_network:
        argv.append("--unshare-net")
    argv += ["--chdir", str(repo_root)]
    argv += list(cmd)
    return argv


def run_sandboxed(
    cmd: Sequence[str],
    *,
    repo_root: Path,
    timeout_s: float,
    allow_network: bool = False,
    extra_rw_paths: Sequence[Path] = (),
) -> SandboxResult:
    """Run `cmd` (an argv list -- never a shell string, so no shell
    metacharacter is ever interpreted) under a `bwrap` sandbox scoped to
    `repo_root` plus a fresh scratch directory.

    The sandbox binds the entire filesystem read-only, then layers
    read-write binds of `repo_root`, the scratch directory, and each of
    `extra_rw_paths` on top; the sandboxed process is killed if this
    process exits first; and the network namespace is unshared -- no
    network access at all -- unless `allow_network` is `True`. The
    command runs with `repo_root` as its working directory.

    Raises:
        SandboxUnavailableError: `bwrap` is not on `PATH`. Nothing is
            spawned in this case.

    A command that exceeds `timeout_s` is killed and reported back as
    `SandboxResult(timed_out=True, exit_code=-1, ...)`, carrying
    whatever partial stdout/stderr was captured before the kill --
    this is an expected, reportable outcome for a caller driving an
    agent loop, not a raised exception.
    """
    if not bwrap_available():
        raise SandboxUnavailableError(f"{_BWRAP_BIN} not found on PATH")

    with tempfile.TemporaryDirectory() as scratch:
        argv = _build_bwrap_argv(
            cmd,
            repo_root=repo_root,
            scratch_dir=Path(scratch),
            allow_network=allow_network,
            extra_rw_paths=extra_rw_paths,
        )
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                stdout=_decode_partial(exc.stdout),
                stderr=_decode_partial(exc.stderr),
                exit_code=-1,
                timed_out=True,
            )

    return SandboxResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        timed_out=False,
    )
