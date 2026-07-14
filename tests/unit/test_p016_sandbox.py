"""Unit tests for `kestrel.tools.sandbox`'s pure logic: `bwrap`
availability, the unavailable-raises guard, argv construction, and
timeout handling via a monkeypatched `subprocess.run` -- none of these
need a real `bwrap` binary, unlike `tests/integration/test_p016_sandbox.py`'s
real-containment proofs.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

from kestrel.tools.sandbox import (
    SandboxUnavailableError,
    bwrap_available,
    run_sandboxed,
)

pytestmark = [pytest.mark.p016, pytest.mark.unit]

# `kestrel.tools.__init__` imports several sibling tool modules by name,
# which does not rebind `sandbox` itself -- but resolving the module via
# `importlib` rather than a bare `import ... as` keeps this test immune
# to that kind of shadowing regardless, the same defensive habit
# `test_p015_search_timeout.py` established for `search`.
_sandbox_module = importlib.import_module("kestrel.tools.sandbox")


@pytest.mark.sanity
def test_bwrap_available_reflects_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given `shutil.which` reporting `bwrap` present or absent, when
    `bwrap_available` is called, then it returns exactly that."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/bwrap")
    assert bwrap_available() is True

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert bwrap_available() is False


def test_run_sandboxed_raises_when_bwrap_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `bwrap` reported absent, when `run_sandboxed` is called,
    then `SandboxUnavailableError` is raised naming it, and no
    subprocess is ever spawned."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: False)

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        """Stand in for `subprocess.run`, failing the test if reached."""
        raise AssertionError("subprocess.run should not be called when bwrap is absent")

    monkeypatch.setattr(subprocess, "run", _unexpected_call)

    with pytest.raises(SandboxUnavailableError, match="bwrap"):
        run_sandboxed(["true"], repo_root=tmp_path, timeout_s=5.0)


def test_run_sandboxed_timeout_returns_timed_out_result_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `bwrap` invocation that does not finish within
    `timeout_s`, when run, then `run_sandboxed` returns a
    `SandboxResult` with `timed_out=True` and `exit_code=-1`, carrying
    whatever partial stdout/stderr the timeout captured, rather than
    letting `subprocess.TimeoutExpired` escape."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: True)

    def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Stand in for `subprocess.run`, raising a timeout with partial
        `str` output attached, as `Popen.communicate` can produce."""
        raise subprocess.TimeoutExpired(
            cmd=["bwrap"], timeout=5.0, output="partial out", stderr="partial err"
        )

    monkeypatch.setattr(subprocess, "run", _timeout)

    result = run_sandboxed(["sleep", "999"], repo_root=tmp_path, timeout_s=5.0)

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.stdout == "partial out"
    assert result.stderr == "partial err"


def test_run_sandboxed_timeout_with_no_partial_output_returns_empty_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a timeout whose exception carries no captured output at
    all (`stdout`/`stderr` are `None`), when run, then the resulting
    `SandboxResult` reports empty strings rather than `None` or a raw
    `AttributeError`."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: True)

    def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Stand in for `subprocess.run`, raising a timeout with no
        captured output attached at all."""
        raise subprocess.TimeoutExpired(cmd=["bwrap"], timeout=5.0)

    monkeypatch.setattr(subprocess, "run", _timeout)

    result = run_sandboxed(["sleep", "999"], repo_root=tmp_path, timeout_s=5.0)

    assert result.timed_out is True
    assert result.stdout == ""
    assert result.stderr == ""


def test_run_sandboxed_timeout_with_raw_bytes_output_decodes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a timeout whose exception carries partial output as raw
    `bytes` (the shape `Popen.communicate` produces on some platforms
    when a caller has not requested text decoding at that layer), when
    run, then the resulting `SandboxResult` decodes it as UTF-8 rather
    than returning the raw bytes object or raising."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: True)

    def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Stand in for `subprocess.run`, raising a timeout with partial
        raw `bytes` output attached instead of already-decoded `str`."""
        raise subprocess.TimeoutExpired(
            cmd=["bwrap"], timeout=5.0, output=b"raw out", stderr=b"raw err"
        )

    monkeypatch.setattr(subprocess, "run", _timeout)

    result = run_sandboxed(["sleep", "999"], repo_root=tmp_path, timeout_s=5.0)

    assert result.stdout == "raw out"
    assert result.stderr == "raw err"


def test_run_sandboxed_success_returns_the_completed_process_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a `bwrap` invocation that exits normally, when run, then
    the resulting `SandboxResult` carries its stdout, stderr, and exit
    code with `timed_out=False`."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: True)

    def _fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """Stand in for `subprocess.run`, returning a canned successful
        `echo` result after checking the sandboxed argv starts with
        `bwrap`."""
        assert argv[0] == "bwrap"
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="hello\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = run_sandboxed(["echo", "hello"], repo_root=tmp_path, timeout_s=5.0)

    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.timed_out is False


def _capture_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    """Monkeypatch `bwrap_available` to `True` and `subprocess.run` to
    record every argv it is called with (returning a trivial successful
    result), so a test can inspect exactly what `run_sandboxed` builds
    without spawning anything real."""
    monkeypatch.setattr(_sandbox_module, "bwrap_available", lambda: True)
    calls: list[list[str]] = []

    def _record(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Stand in for `subprocess.run`, recording `argv` and returning
        a trivial successful result."""
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _record)
    return calls


def test_argv_contains_the_default_containment_flags_and_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given no `allow_network` or `extra_rw_paths` override, when run,
    then the `bwrap` argv ro-binds the filesystem root, mounts a fresh
    `/dev`, binds `repo_root` read-write, unshares the network
    namespace, dies with its parent, sets `repo_root` as the working
    directory, and appends the command's own argv untouched at the
    end."""
    calls = _capture_argv(monkeypatch)

    run_sandboxed(["echo", "hi"], repo_root=tmp_path, timeout_s=5.0)

    (argv,) = calls
    assert argv[0] == "bwrap"
    assert "--ro-bind" in argv
    assert argv[argv.index("--ro-bind") + 1 : argv.index("--ro-bind") + 3] == ["/", "/"]
    assert "--dev" in argv
    assert argv[argv.index("--dev") + 1] == "/dev"
    # Not just present, but layered directly on top of the root bind:
    # bwrap applies later filesystem sources on top of earlier ones, so
    # `--dev /dev` must follow `--ro-bind / /` to actually take effect
    # rather than being shadowed by it.
    assert argv.index("--dev") == argv.index("--ro-bind") + 3
    assert "--bind" in argv
    bind_index = argv.index("--bind")
    assert argv[bind_index + 1 : bind_index + 3] == [str(tmp_path), str(tmp_path)]
    assert "--die-with-parent" in argv
    assert "--unshare-net" in argv
    assert argv[argv.index("--chdir") + 1] == str(tmp_path)
    assert argv[-2:] == ["echo", "hi"]


def test_argv_omits_unshare_net_when_allow_network_is_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `allow_network=True`, when run, then the `bwrap` argv does
    not carry `--unshare-net` -- the network-off default is only ever
    lifted by an explicit opt-in."""
    calls = _capture_argv(monkeypatch)

    run_sandboxed(["true"], repo_root=tmp_path, timeout_s=5.0, allow_network=True)

    (argv,) = calls
    assert "--unshare-net" not in argv


def test_argv_adds_a_bind_pair_for_each_extra_rw_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `extra_rw_paths` naming additional writable directories,
    when run, then the `bwrap` argv carries one `--bind src dst` pair
    per extra path, on top of the default repo-root and scratch binds."""
    calls = _capture_argv(monkeypatch)
    extra = tmp_path / "extra"
    extra.mkdir()

    run_sandboxed(["true"], repo_root=tmp_path, timeout_s=5.0, extra_rw_paths=[extra])

    (argv,) = calls
    bind_targets = [argv[i + 1] for i, token in enumerate(argv) if token == "--bind"]
    assert str(extra) in bind_targets
