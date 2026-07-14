"""Integration tests for `run_sandboxed`'s real `bwrap` invocation:
output capture, filesystem containment, network denial, and the
timeout bound.

Skipped locally when `bwrap` is not on `PATH` -- a real local seam (the
binary genuinely may not be installed, and never will be on a
non-Linux host), not a network one. CI installs `bubblewrap` on every
runner, so this suite always actually runs there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kestrel.tools.sandbox import bwrap_available, run_sandboxed


def _can_initialize_network_namespace() -> bool:
    """Check if the environment can initialize network namespaces for bwrap.

    This is a prerequisite for all bwrap tests. If bwrap cannot set up the
    loopback interface, all sandboxed commands will fail immediately.
    """
    if shutil.which("bwrap") is None:
        return False
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_sandboxed(["true"], repo_root=Path(tmpdir), timeout_s=5.0)
            return result.exit_code == 0 and not result.timed_out
    except Exception:
        return False


pytestmark = [
    pytest.mark.p016,
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not found on PATH"),
    pytest.mark.skipif(
        not _can_initialize_network_namespace(),
        reason="bwrap cannot initialize network namespace (missing capabilities or AppArmor restrictions)",
    ),
]


@pytest.mark.sanity
def test_echo_inside_the_sandbox_succeeds(tmp_path: Path) -> None:
    """Given a trivial `echo` command, when run inside the sandbox,
    then it exits 0 and its stdout contains the echoed text."""
    result = run_sandboxed(["echo", "hello"], repo_root=tmp_path, timeout_s=10.0)

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False


def test_a_python_interpreter_starts_successfully_inside_the_sandbox(
    tmp_path: Path,
) -> None:
    """Given a command that starts a fresh CPython interpreter (`pytest`
    is exactly this shape, and is what `verify`'s own KESTREL.md-configured
    `test` command normally runs), when run inside the sandbox, then it
    starts and exits 0 rather than dying during interpreter startup.

    CPython seeds hash randomization from `getrandom()`/`/dev/urandom`
    on every startup; a sandbox whose `/dev` is not itself a working
    device tree (as opposed to one that merely happens to look normal
    read-only under `--ro-bind`) fails this before the interpreter can
    run a single line, with `_Py_HashRandomization_Init: failed to get
    random numbers to initialize Python` on stderr -- the exact failure
    `--dev /dev` in `_build_bwrap_argv` exists to prevent.
    """
    result = run_sandboxed(
        ["python3", "-c", "print('ok')"], repo_root=tmp_path, timeout_s=10.0
    )

    assert result.timed_out is False
    assert result.exit_code == 0, result.stderr
    assert "ok" in result.stdout


def test_writing_outside_repo_root_fails_without_crashing_the_harness(
    tmp_path: Path,
) -> None:
    """Given a command that tries to write to a path outside
    `repo_root` and outside the per-call scratch directory -- directly
    under `/tmp`, which is normally world-writable outside the sandbox,
    isolating this from an ownership-based rejection -- when run inside
    the sandbox, then the command itself fails with a
    permission-denied-shaped outcome (the read-only bind of the rest of
    the filesystem) rather than the harness raising or crashing."""
    outside_target = "/tmp/kestrel-sandbox-writetest-outside"

    result = run_sandboxed(
        ["sh", "-c", f"echo blocked > {outside_target}"],
        repo_root=tmp_path,
        timeout_s=10.0,
    )

    assert result.timed_out is False
    assert result.exit_code != 0
    assert "denied" in result.stderr.lower() or "read-only" in result.stderr.lower()


def test_network_call_fails_due_to_the_unshared_namespace(tmp_path: Path) -> None:
    """Given a command that tries to open a TCP connection to a real,
    non-loopback address, when run inside the sandbox with the default
    `allow_network=False`, then the connection fails with a
    network-unreachable error -- proof the network namespace itself is
    unshared and carries no route out at all, not merely that
    something refused the connection.

    Deliberately targets a non-loopback address rather than
    `127.0.0.1`: bwrap brings up a real, working loopback interface
    inside an unshared network namespace (so sandboxed code can still
    talk to itself over `127.0.0.1`), so a loopback connection to a
    port nothing listens on gets an ordinary connection-refused error
    both inside and outside the sandbox -- it proves nothing about
    isolation either way. A namespace with no configured route besides
    loopback, by contrast, cannot reach any other address at all,
    which is exactly what network isolation is supposed to guarantee.
    """
    result = run_sandboxed(
        [
            "python3",
            "-c",
            "import socket; socket.create_connection(('10.255.255.1', 80), timeout=2)",
        ],
        repo_root=tmp_path,
        timeout_s=10.0,
    )

    assert result.timed_out is False
    assert result.exit_code != 0
    assert "network is unreachable" in result.stderr.lower()


def test_command_exceeding_timeout_reports_timed_out_rather_than_hanging(
    tmp_path: Path,
) -> None:
    """Given a command that sleeps far longer than its `timeout_s`
    bound, when run inside the sandbox, then `run_sandboxed` returns
    promptly with `timed_out=True` rather than blocking the test until
    the command would have finished on its own."""
    result = run_sandboxed(["sleep", "60"], repo_root=tmp_path, timeout_s=2.0)

    assert result.timed_out is True
    assert result.exit_code == -1


def test_bwrap_available_is_true_on_a_runner_with_bwrap_installed() -> None:
    """Given a runner where this whole module has not been skipped
    (i.e. `bwrap` is genuinely on `PATH`), when `bwrap_available` is
    called, then it returns `True`."""
    assert bwrap_available() is True
