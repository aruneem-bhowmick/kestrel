"""Integration tests for `verify` against a real `bwrap` sandbox: a
real `pytest` invocation configured through KESTREL.md, passing and
failing, plus the mandatory red-team proof that a hostile file's
content, once echoed to a configured command's stdout, cannot corrupt
`verify`'s own returned frame or the persisted report around it.

Skipped locally when `bwrap` is not on `PATH` (matching
`tests/integration/test_p016_sandbox.py`'s precedent) -- a real local
seam, not a network one. CI installs `bubblewrap` on every runner, so
this suite always actually runs there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kestrel.kestrel_md import load_kestrel_md
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.sandbox import run_sandboxed
from kestrel.tools.verify import VerifyArgs, run_verification, verify

_HOSTILE_CASE_ID = "readme_ignore_previous_instructions"


def _can_initialize_network_namespace() -> bool:
    """Check whether this environment can actually run a sandboxed
    command at all -- a prerequisite shared with every other `bwrap`
    integration suite in this project (see `test_p016_sandbox.py`)."""
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
    pytest.mark.p025,
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not found on PATH"),
    pytest.mark.skipif(
        not _can_initialize_network_namespace(),
        reason="bwrap cannot initialize network namespace (missing capabilities or AppArmor restrictions)",
    ),
]

_TASK_ID = "task-sandbox"
_TURN_ID = 1


def _write_kestrel_md(repo_root: Path, *, test_command: str) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "KESTREL.md").write_text(
        f'```kestrel-verify\ntest = "{test_command}"\n```\n', encoding="utf-8"
    )


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


@pytest.fixture(scope="session")
def hostile_file_case() -> InjectionCase:
    """The corpus case used to prove a hostile file's content, once
    echoed by a configured command, cannot corrupt `verify`'s own
    framing. Looked up lazily so a lookup failure surfaces as a normal
    test error tied to whichever test requested it."""
    return _find_case(_HOSTILE_CASE_ID)


def test_a_real_passing_pytest_run_reports_passed(tmp_path: Path) -> None:
    """Given a KESTREL.md configuring `test = "pytest -q"` against a
    fixture repo with one passing test, when verified, then the report
    is `passed=True`."""
    _write_kestrel_md(tmp_path, test_command="pytest -q")
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    kestrel_md = load_kestrel_md(tmp_path)
    assert kestrel_md is not None

    report = run_verification(
        kestrel_md,
        only=["test"],
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        timeout_s=60.0,
    )

    assert report.passed is True
    assert report.commands[0].exit_code == 0


def test_a_real_failing_pytest_run_reports_the_failure_text(tmp_path: Path) -> None:
    """Given a KESTREL.md configuring `test = "pytest -q"` against a
    fixture repo with one failing test, when verified, then the report
    is `passed=False` and the command's own `stdout` carries the real
    pytest failure output."""
    _write_kestrel_md(tmp_path, test_command="pytest -q")
    (tmp_path / "test_sample.py").write_text(
        "def test_fails():\n    assert False, 'deliberate failure'\n",
        encoding="utf-8",
    )
    kestrel_md = load_kestrel_md(tmp_path)
    assert kestrel_md is not None

    report = run_verification(
        kestrel_md,
        only=["test"],
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        timeout_s=60.0,
    )

    assert report.passed is False
    assert report.commands[0].exit_code != 0
    assert "deliberate failure" in report.commands[0].stdout


@pytest.mark.redteam
def test_hostile_echoed_file_content_cannot_corrupt_verifys_own_framing(
    tmp_path: Path, hostile_file_case: InjectionCase
) -> None:
    """Given a fixture file whose content is one of the injection
    corpus's hostile payloads, and a KESTREL.md configuring `lint` to
    `cat` that file straight to stdout, when verified, then `verify`'s
    own returned result still starts and ends with the real frame
    markers -- unable to acquire the hostile payload at all, since
    `verify` never returns raw per-command output, only a rendered
    pass/fail summary -- and the persisted report on disk still carries
    the real payload text, unbroken, for a human or later tool to
    inspect.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "HOSTILE.md").write_text(hostile_file_case.payload, encoding="utf-8")
    _write_kestrel_md(tmp_path, test_command="true")
    (tmp_path / "KESTREL.md").write_text(
        '```kestrel-verify\nlint = "cat HOSTILE.md"\ntest = "true"\n```\n',
        encoding="utf-8",
    )

    result = verify(
        VerifyArgs(only=("lint",)),
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
    )

    assert result.startswith("<<<UNTRUSTED:tool_stdout:verify>>>")
    assert result.endswith("<<<END_UNTRUSTED>>>")
    for marker in hostile_file_case.forbidden_markers:
        assert result.count(marker) == 1

    persisted_path = (
        tmp_path / ".kestrel" / "artifacts" / f"verification-{_TASK_ID}-{_TURN_ID}.md"
    )
    persisted_text = persisted_path.read_text(encoding="utf-8")
    assert hostile_file_case.payload.strip() in persisted_text
