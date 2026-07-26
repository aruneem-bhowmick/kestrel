"""Unit tests for the resource guard's pure RSS classification and its
platform-reading helpers.

`check_resources` is pure and needs no filesystem or process state, so
its cases run everywhere. `measure_kestrel_rss_mb` and
`measure_ollama_rss_mb` read real platform state (`resource.getrusage`
and `/proc` respectively); the former is skipped on a non-POSIX
platform, and the latter is exercised against a fake `/proc` tree built
under `tmp_path` rather than the real one, so these tests never depend
on -- or interfere with -- whatever is actually running on the machine.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.managers.resource_guard as resource_guard_module
from kestrel.managers.resource_guard import (
    ResourceStatus,
    check_resources,
    measure_kestrel_rss_mb,
    measure_ollama_rss_mb,
)

pytestmark = [pytest.mark.p064, pytest.mark.unit]


# --- check_resources: pure classification ------------------------------------


@pytest.mark.sanity
def test_combined_rss_well_under_threshold_has_no_warning() -> None:
    """Given combined kestrel+ollama RSS far below the warn threshold,
    when classified, then no warning is produced."""
    status = check_resources(
        kestrel_rss_mb=100.0,
        ollama_rss_mb=200.0,
        ceiling_mb=1000.0,
        warn_threshold=Decimal("0.85"),
    )

    assert status == ResourceStatus(
        kestrel_rss_mb=100.0, ollama_rss_mb=200.0, ceiling_mb=1000.0, warning=None
    )


@pytest.mark.sanity
def test_combined_rss_at_threshold_warns_naming_both_figures() -> None:
    """Given combined kestrel+ollama RSS exactly at the warn threshold,
    when classified, then the warning names both the combined figure and
    the ceiling."""
    status = check_resources(
        kestrel_rss_mb=425.0,
        ollama_rss_mb=425.0,
        ceiling_mb=1000.0,
        warn_threshold=Decimal("0.85"),
    )

    assert status.warning is not None
    assert "850" in status.warning
    assert "1000" in status.warning


@pytest.mark.sanity
def test_combined_rss_over_threshold_warns() -> None:
    """Given combined kestrel+ollama RSS above the warn threshold, when
    classified, then a warning is produced."""
    status = check_resources(
        kestrel_rss_mb=900.0,
        ollama_rss_mb=200.0,
        ceiling_mb=1000.0,
        warn_threshold=Decimal("0.85"),
    )

    assert status.warning is not None


@pytest.mark.sanity
def test_unmeasured_ollama_below_threshold_uses_kestrel_rss_alone() -> None:
    """Given `ollama_rss_mb` is `None` and kestrel's own RSS alone sits
    under the threshold, when classified, then no warning is produced --
    the combined figure is kestrel's own reading alone."""
    status = check_resources(
        kestrel_rss_mb=100.0,
        ollama_rss_mb=None,
        ceiling_mb=1000.0,
        warn_threshold=Decimal("0.85"),
    )

    assert status.warning is None


@pytest.mark.sanity
def test_unmeasured_ollama_over_threshold_warns_naming_it_unmeasured() -> None:
    """Given `ollama_rss_mb` is `None` but kestrel's own RSS alone already
    crosses the threshold, when classified, then the warning names
    Ollama's own usage as unmeasured rather than silently excluding it."""
    status = check_resources(
        kestrel_rss_mb=900.0,
        ollama_rss_mb=None,
        ceiling_mb=1000.0,
        warn_threshold=Decimal("0.85"),
    )

    assert status.warning is not None
    assert "could not be measured" in status.warning


def test_check_resources_uses_default_ceiling_and_threshold() -> None:
    """Given no explicit ceiling or threshold, when classified, then the
    module's own defaults (a 5200MB ceiling, an 85% warn threshold)
    apply -- a caller measuring a Jetson Orin Nano's own default board
    never has to pass either explicitly."""
    status = check_resources(kestrel_rss_mb=100.0, ollama_rss_mb=None)

    assert status.ceiling_mb == 5200.0
    assert status.warning is None


# --- measure_kestrel_rss_mb ---------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="resource.getrusage is POSIX-only")
def test_measure_kestrel_rss_mb_returns_a_positive_float() -> None:
    """Given the test process itself is running, when its own resident
    memory is measured, then the result is a positive float -- this
    process is always resident somewhere."""
    rss = measure_kestrel_rss_mb()

    assert isinstance(rss, float)
    assert rss > 0.0


# --- measure_ollama_rss_mb: a fake /proc tree ---------------------------------


def _write_fake_process(
    proc_root: Path,
    pid: str,
    *,
    comm: str | None = None,
    cmdline: bytes | None = None,
    vmrss_kb: int | None = None,
) -> None:
    """Create one fake `/proc/<pid>` entry under `proc_root`.

    Any of `comm`, `cmdline`, or `vmrss_kb` left unset simply omits that
    file, standing in for a real process that never exposed it (or one
    that exited before it could be read).
    """
    pid_dir = proc_root / pid
    pid_dir.mkdir()
    if comm is not None:
        (pid_dir / "comm").write_text(f"{comm}\n", encoding="utf-8")
    if cmdline is not None:
        (pid_dir / "cmdline").write_bytes(cmdline)
    if vmrss_kb is not None:
        (pid_dir / "status").write_text(
            f"Name:\t{comm or ''}\nVmRSS:\t  {vmrss_kb} kB\n", encoding="utf-8"
        )


def test_measure_ollama_rss_mb_finds_a_process_named_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a fake `/proc` tree with a process whose `comm` is exactly
    "ollama", when measured, then its own `VmRSS` is returned in MB."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    _write_fake_process(tmp_path, "123", comm="ollama", vmrss_kb=2048)

    assert measure_ollama_rss_mb() == pytest.approx(2.0)


def test_measure_ollama_rss_mb_finds_a_process_by_cmdline_substring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a fake `/proc` tree with a process whose `comm` does not
    match but whose `cmdline` contains "ollama serve" once its null-byte
    argument separators are read, when measured, then its own `VmRSS` is
    still found."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    _write_fake_process(
        tmp_path,
        "456",
        comm="wrapper-script",
        cmdline=b"/usr/local/bin/ollama\x00serve\x00",
        vmrss_kb=4096,
    )

    assert measure_ollama_rss_mb() == pytest.approx(4.0)


def test_measure_ollama_rss_mb_returns_none_when_no_process_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a fake `/proc` tree with only unrelated processes, when
    measured, then `None` is returned."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    _write_fake_process(
        tmp_path, "789", comm="python", cmdline=b"python\x00app.py", vmrss_kb=1024
    )

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_returns_none_when_neither_comm_nor_cmdline_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a process directory with neither a `comm` nor a `cmdline`
    file (an unusual, half-populated `/proc` entry), when measured, then
    it is treated as a non-match rather than raising."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    (tmp_path / "555").mkdir()

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_returns_none_when_proc_root_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `/proc` does not exist at all (any non-Linux platform), when
    measured, then `None` is returned rather than raising."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path / "nonexistent")

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_returns_none_when_status_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a matching process directory with no `status` file (as if
    the process exited between being found and being read), when
    measured, then `None` is returned rather than raising."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    _write_fake_process(tmp_path, "999", comm="ollama")

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_matches_by_cmdline_when_comm_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a process directory with no `comm` file at all (some
    processes never expose one) but a matching `cmdline`, when measured,
    then the cmdline match still finds it -- a missing `comm` file falls
    through to the cmdline check rather than raising."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    pid_dir = tmp_path / "111"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"ollama\x00serve\x00")
    (pid_dir / "status").write_text("VmRSS:\t  1024 kB\n", encoding="utf-8")

    assert measure_ollama_rss_mb() == pytest.approx(1.0)


def test_measure_ollama_rss_mb_returns_none_for_a_vmrss_line_missing_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a matching process whose `status` file's own `VmRSS` line
    carries no numeric field at all, when measured, then `None` is
    returned rather than raising an `IndexError`."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    pid_dir = tmp_path / "222"
    pid_dir.mkdir()
    (pid_dir / "comm").write_text("ollama\n", encoding="utf-8")
    (pid_dir / "status").write_text("VmRSS:\n", encoding="utf-8")

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_returns_none_for_a_non_numeric_vmrss_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a matching process whose `status` file's own `VmRSS` line
    carries a non-numeric value, when measured, then `None` is returned
    rather than raising a `ValueError`."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    pid_dir = tmp_path / "333"
    pid_dir.mkdir()
    (pid_dir / "comm").write_text("ollama\n", encoding="utf-8")
    (pid_dir / "status").write_text("VmRSS:\tmany kB\n", encoding="utf-8")

    assert measure_ollama_rss_mb() is None


def test_measure_ollama_rss_mb_returns_none_when_status_has_no_vmrss_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a matching process whose `status` file carries no `VmRSS`
    line at all, when measured, then `None` is returned."""
    monkeypatch.setattr(resource_guard_module, "_PROC_ROOT", tmp_path)
    pid_dir = tmp_path / "444"
    pid_dir.mkdir()
    (pid_dir / "comm").write_text("ollama\n", encoding="utf-8")
    (pid_dir / "status").write_text("Name:\tollama\n", encoding="utf-8")

    assert measure_ollama_rss_mb() is None
