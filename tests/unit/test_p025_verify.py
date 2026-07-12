"""Tests for the `verify` tool: running configured lint/build/test
commands to completion regardless of an earlier failure, the `only`
filter, timeout reporting, markdown rendering, artifact persistence,
`report_sink` bookkeeping, and argument parsing.

`run_sandboxed` is stubbed throughout rather than requiring a real
`bwrap` invocation -- the properties under test here are `verify`'s own
selection, ordering, rendering, and persistence logic, which are
independent of whether a command's outcome came from a real sandboxed
process or a stand-in one. See `tests/integration/test_p025_verify_sandbox.py`
for the real-sandbox proof.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from kestrel.kestrel_md import KestrelMd, load_kestrel_md
from kestrel.tools.sandbox import SandboxResult
from kestrel.tools.verify import (
    VerificationCommandResult,
    VerificationReport,
    VerifyArgs,
    VerifyError,
    parse_verify_args,
    persist_verification_report,
    render_verification_markdown,
    run_verification,
    verify,
)

pytestmark = [pytest.mark.p025, pytest.mark.unit]

# See `test_p016_execute_redteam.py` for why this module is resolved via
# `importlib` rather than `import kestrel.tools.verify as verify_module`:
# `kestrel.tools.__init__` rebinds the `verify` *attribute* on the
# `kestrel.tools` package to the function of the same name, shadowing the
# submodule itself once the package has finished importing.
_verify_module = importlib.import_module("kestrel.tools.verify")

_TASK_ID = "task-1"
_TURN_ID = 3


def _write_kestrel_md(repo_root: Path, content: str) -> Path:
    """Write `content` as UTF-8 bytes to `repo_root / "KESTREL.md"`,
    creating `repo_root` as needed, and return the written path."""
    repo_root.mkdir(parents=True, exist_ok=True)
    path = repo_root / "KESTREL.md"
    path.write_bytes(content.encode("utf-8"))
    return path


def _all_commands_kestrel_md(repo_root: Path) -> KestrelMd:
    """Write and load a KESTREL.md configuring all three verify
    commands with distinct, easily-identified command strings."""
    _write_kestrel_md(
        repo_root,
        (
            "```kestrel-verify\n"
            'lint = "run-lint"\n'
            'build = "run-build"\n'
            'test = "run-test"\n'
            "```\n"
        ),
    )
    loaded = load_kestrel_md(repo_root)
    assert loaded is not None
    return loaded


def _stub_sandboxed_by_command(
    monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, SandboxResult]
) -> list[str]:
    """Replace `verify`'s own `run_sandboxed` reference with a stub that
    maps the shell command string (the third element of the `["sh",
    "-c", command]` argv `run_verification` builds) to a canned
    `SandboxResult` from `outcomes`. Returns the list of command strings
    invoked, in call order, so a test can assert every configured
    command actually ran (not short-circuited)."""
    calls: list[str] = []

    def _stub(cmd: list[str], **_kwargs: object) -> SandboxResult:
        assert cmd[:2] == ["sh", "-c"]
        command = cmd[2]
        calls.append(command)
        return outcomes[command]

    monkeypatch.setattr(_verify_module, "run_sandboxed", _stub)
    return calls


def _ok(stdout: str = "") -> SandboxResult:
    return SandboxResult(stdout=stdout, stderr="", exit_code=0, timed_out=False)


def _failed(exit_code: int = 1, stderr: str = "") -> SandboxResult:
    return SandboxResult(stdout="", stderr=stderr, exit_code=exit_code, timed_out=False)


def _timed_out() -> SandboxResult:
    return SandboxResult(stdout="", stderr="", exit_code=-1, timed_out=True)


@pytest.mark.sanity
def test_all_commands_succeed_reports_passed_with_three_results_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a KESTREL.md configuring all three commands, all of which
    succeed, when verified, then `passed` is True and `commands` holds
    exactly three results in lint/build/test order."""
    kestrel_md = _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _ok(), "run-test": _ok()},
    )

    report = run_verification(
        kestrel_md,
        only=None,
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
    )

    assert report.passed is True
    assert [result.name for result in report.commands] == ["lint", "build", "test"]
    assert all(result.exit_code == 0 for result in report.commands)


def test_a_failing_command_does_not_short_circuit_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given lint failing with a non-zero exit code, when verified, then
    `passed` is False but build and test still ran -- the model needs
    every configured check's own result to fix them together."""
    kestrel_md = _all_commands_kestrel_md(tmp_path)
    calls = _stub_sandboxed_by_command(
        monkeypatch,
        {
            "run-lint": _failed(exit_code=1, stderr="lint error"),
            "run-build": _ok(),
            "run-test": _ok(),
        },
    )

    report = run_verification(
        kestrel_md,
        only=None,
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
    )

    assert report.passed is False
    assert calls == ["run-lint", "run-build", "run-test"]
    lint_result = report.commands[0]
    assert lint_result.exit_code == 1
    assert lint_result.stderr == "lint error"


def test_a_timed_out_command_marks_the_report_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given build timing out, when verified, then that result's
    `timed_out` is True and the overall report is `passed=False`."""
    kestrel_md = _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _timed_out(), "run-test": _ok()},
    )

    report = run_verification(
        kestrel_md,
        only=None,
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
    )

    assert report.passed is False
    build_result = next(r for r in report.commands if r.name == "build")
    assert build_result.timed_out is True
    assert build_result.exit_code == -1


@pytest.mark.sanity
def test_only_restricts_the_run_to_the_named_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given all three commands configured, when verified with
    `only=["test"]`, then only `test` runs."""
    kestrel_md = _all_commands_kestrel_md(tmp_path)
    calls = _stub_sandboxed_by_command(monkeypatch, {"run-test": _ok()})

    report = run_verification(
        kestrel_md,
        only=["test"],
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
    )

    assert calls == ["run-test"]
    assert [result.name for result in report.commands] == ["test"]
    assert report.passed is True


@pytest.mark.sanity
def test_only_naming_an_unconfigured_command_raises_naming_it(
    tmp_path: Path,
) -> None:
    """Given a KESTREL.md configuring only lint and test, when verified
    with `only=["build"]`, then `VerifyError` names `build` as not
    configured."""
    _write_kestrel_md(
        tmp_path,
        '```kestrel-verify\nlint = "run-lint"\ntest = "run-test"\n```\n',
    )
    kestrel_md = load_kestrel_md(tmp_path)
    assert kestrel_md is not None

    with pytest.raises(VerifyError, match="build"):
        run_verification(
            kestrel_md,
            only=["build"],
            repo_root=tmp_path,
            task_id=_TASK_ID,
            turn_id=_TURN_ID,
        )


@pytest.mark.sanity
def test_no_kestrel_md_raises(tmp_path: Path) -> None:
    """Given a repo root with no KESTREL.md, when `verify` is called,
    then `VerifyError` is raised -- there is nothing to run."""
    with pytest.raises(VerifyError):
        verify(VerifyArgs(), repo_root=tmp_path, task_id=_TASK_ID, turn_id=_TURN_ID)


@pytest.mark.sanity
def test_kestrel_md_with_no_verify_block_raises(tmp_path: Path) -> None:
    """Given a KESTREL.md with prose but no kestrel-verify block, when
    `verify` is called, then `VerifyError` is raised -- there are no
    commands to run."""
    _write_kestrel_md(tmp_path, "# Conventions\n\nBe kind to the tests.\n")

    with pytest.raises(VerifyError):
        verify(VerifyArgs(), repo_root=tmp_path, task_id=_TASK_ID, turn_id=_TURN_ID)


def test_render_verification_markdown_names_every_exit_code_and_truncates() -> None:
    """Given a report with one large-stdout command, when rendered as
    markdown, then every command's exit code and timeout flag appear
    and the oversized stream is capped with a truncation note (reusing
    `execute`'s own cap constant)."""
    large_stdout = "a" * (64 * 1024 + 10)
    report = VerificationReport(
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        commands=(
            VerificationCommandResult(
                name="lint",
                command="run-lint",
                exit_code=0,
                timed_out=False,
                stdout="",
                stderr="",
            ),
            VerificationCommandResult(
                name="test",
                command="run-test",
                exit_code=1,
                timed_out=False,
                stdout=large_stdout,
                stderr="",
            ),
        ),
        passed=False,
    )

    rendered = render_verification_markdown(report)

    assert "exit_code: 0" in rendered
    assert "exit_code: 1" in rendered
    assert "timed_out: False" in rendered
    assert "... [truncated: 10 more bytes omitted]" in rendered
    assert large_stdout not in rendered


@pytest.mark.sanity
def test_verify_returns_a_result_framed_as_untrusted_tool_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a passing configuration, when `verify` is called, then its
    returned string is framed with `source="tool_stdout"` and
    `origin="verify"`."""
    _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _ok(), "run-test": _ok()},
    )

    result = verify(
        VerifyArgs(), repo_root=tmp_path, task_id=_TASK_ID, turn_id=_TURN_ID
    )

    assert result.startswith("<<<UNTRUSTED:tool_stdout:verify>>>")
    assert result.endswith("<<<END_UNTRUSTED>>>")
    assert "PASSED" in result


def test_report_sink_defaulting_to_none_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given no `report_sink` argument, when `verify` is called, then it
    completes normally -- a caller uninterested in the report is never
    forced to supply a list."""
    _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _ok(), "run-test": _ok()},
    )

    verify(VerifyArgs(), repo_root=tmp_path, task_id=_TASK_ID, turn_id=_TURN_ID)


def test_report_sink_receives_exactly_one_matching_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a mutable `report_sink` list, when `verify` is called, then
    exactly one `VerificationReport` matching the call's own outcome is
    appended to it."""
    _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _ok(), "run-test": _ok()},
    )
    sink: list[VerificationReport] = []

    verify(
        VerifyArgs(),
        repo_root=tmp_path,
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        report_sink=sink,
    )

    assert len(sink) == 1
    assert sink[0].task_id == _TASK_ID
    assert sink[0].turn_id == _TURN_ID
    assert sink[0].passed is True


def test_persist_verification_report_writes_the_rendered_markdown(
    tmp_path: Path,
) -> None:
    """Given a report, when persisted, then the exact rendered markdown
    is written to `.kestrel/artifacts/verification-<task>-<turn>.md`,
    creating that directory tree since it does not yet exist."""
    report = VerificationReport(
        task_id=_TASK_ID,
        turn_id=_TURN_ID,
        commands=(
            VerificationCommandResult(
                name="lint",
                command="run-lint",
                exit_code=0,
                timed_out=False,
                stdout="ok",
                stderr="",
            ),
        ),
        passed=True,
    )
    assert not (tmp_path / ".kestrel").exists()

    written = persist_verification_report(report, repo_root=tmp_path)

    expected_path = (
        tmp_path / ".kestrel" / "artifacts" / f"verification-{_TASK_ID}-{_TURN_ID}.md"
    )
    assert written == expected_path
    assert written.read_text(encoding="utf-8") == render_verification_markdown(report)


def test_verify_persists_a_report_readable_at_the_documented_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a passing configuration, when `verify` is called, then a
    verification report markdown file exists at the documented artifact
    path afterward."""
    _all_commands_kestrel_md(tmp_path)
    _stub_sandboxed_by_command(
        monkeypatch,
        {"run-lint": _ok(), "run-build": _ok(), "run-test": _ok()},
    )

    verify(VerifyArgs(), repo_root=tmp_path, task_id=_TASK_ID, turn_id=_TURN_ID)

    expected_path = (
        tmp_path / ".kestrel" / "artifacts" / f"verification-{_TASK_ID}-{_TURN_ID}.md"
    )
    assert expected_path.is_file()
    assert "PASSED" in expected_path.read_text(encoding="utf-8")


# -- Argument parsing -------------------------------------------------


@pytest.mark.sanity
def test_parse_verify_args_with_no_fields_defaults_only_to_none() -> None:
    """Given an empty arguments object, when parsed, then `only` is
    `None` -- every configured command runs."""
    assert parse_verify_args("{}") == VerifyArgs(only=None)


@pytest.mark.sanity
def test_parse_verify_args_accepts_a_valid_only_list() -> None:
    """Given `only` naming two valid command names, when parsed, then
    they survive as a tuple in the given order."""
    args = parse_verify_args(json.dumps({"only": ["lint", "test"]}))

    assert args.only == ("lint", "test")


def test_parse_verify_args_rejects_invalid_json() -> None:
    """Given arguments text that is not valid JSON, when parsed, then
    `VerifyError` is raised naming the parse failure rather than a raw
    `json.JSONDecodeError` escaping."""
    with pytest.raises(VerifyError, match="invalid JSON"):
        parse_verify_args("not json")


def test_parse_verify_args_rejects_a_non_object_payload() -> None:
    """Given a JSON array instead of an object, when parsed, then
    `VerifyError` names the expected shape."""
    with pytest.raises(VerifyError, match="expected a JSON object"):
        parse_verify_args("[]")


def test_parse_verify_args_rejects_an_unexpected_field() -> None:
    """Given a field `VERIFY_SCHEMA` does not declare, when parsed, then
    `VerifyError` names it."""
    with pytest.raises(VerifyError, match="timeout_s"):
        parse_verify_args(json.dumps({"timeout_s": 30}))


def test_parse_verify_args_rejects_only_that_is_not_an_array() -> None:
    """Given `only` given as a bare string instead of an array, when
    parsed, then `VerifyError` names the defect."""
    with pytest.raises(VerifyError, match="'only'"):
        parse_verify_args(json.dumps({"only": "lint"}))


def test_parse_verify_args_rejects_an_unknown_command_name_in_only() -> None:
    """Given `only` naming a command outside lint/build/test, when
    parsed, then `VerifyError` names the unknown entry."""
    with pytest.raises(VerifyError, match="deploy"):
        parse_verify_args(json.dumps({"only": ["deploy"]}))
