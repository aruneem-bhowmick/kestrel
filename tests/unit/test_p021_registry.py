"""Tests for `kestrel.tools.registry`: the fixed schema list every
provider call is offered, and `dispatch`'s routing of a `ToolCallEvent`
to its bound tool -- unknown tool names, malformed arguments, each real
tool's own successful path, each tool's own typed error surfacing as a
framed result instead of an exception, and the `**context` keyword
filtering that lets one caller pass a superset dict for every tool.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest

import kestrel.tools.registry as registry_module
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import ToolSchema
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.edit_file import EDIT_FILE_SCHEMA, EditFileArgs, edit_file
from kestrel.tools.execute import EXECUTE_SCHEMA, ExecuteArgs, execute
from kestrel.tools.read_file import READ_FILE_SCHEMA, ReadFileArgs, read_file
from kestrel.tools.registry import all_schemas, dispatch
from kestrel.tools.sandbox import SandboxResult
from kestrel.tools.search import SEARCH_SCHEMA, SearchArgs, search
from kestrel.tools.verify import VERIFY_SCHEMA, VerifyArgs, verify

pytestmark = [pytest.mark.p021, pytest.mark.unit]

# See `test_p019_execute_classification.py` for why `execute` and
# `search` are resolved via `importlib.import_module` rather than a
# plain attribute reference on their own submodules:
# `kestrel.tools.__init__` rebinds the `execute` and `search`
# *attributes* on the `kestrel.tools` package to the functions of the
# same name, shadowing the submodules themselves.
_execute_mod = importlib.import_module("kestrel.tools.execute")
_verify_mod = importlib.import_module("kestrel.tools.verify")


def _stub_run_sandboxed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `execute`'s own `run_sandboxed` reference with a stub
    that never touches a real `bwrap` sandbox, returning a fixed,
    successful `SandboxResult` instead."""

    def _stub(cmd: list[str], **_kwargs: object) -> SandboxResult:
        return SandboxResult(stdout="ok", stderr="", exit_code=0, timed_out=False)

    monkeypatch.setattr(_execute_mod, "run_sandboxed", _stub)


def _stub_verify_run_sandboxed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `verify`'s own `run_sandboxed` reference with a stub
    that never touches a real `bwrap` sandbox, returning a fixed,
    successful `SandboxResult` instead."""

    def _stub(cmd: list[str], **_kwargs: object) -> SandboxResult:
        return SandboxResult(stdout="ok", stderr="", exit_code=0, timed_out=False)

    monkeypatch.setattr(_verify_mod, "run_sandboxed", _stub)


def _stub_rg(
    monkeypatch: pytest.MonkeyPatch, *, stdout: str, returncode: int, stderr: str = ""
) -> None:
    """Replace `search`'s own `subprocess.run` reference with a stub
    that never shells out to a real `rg`, returning a fixed
    `CompletedProcess` instead."""

    def _stub(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr(subprocess, "run", _stub)


@pytest.mark.sanity
def test_all_schemas_returns_five_schemas_in_fixed_order_by_identity() -> None:
    """Given the registry, when `all_schemas()` is called, then it
    returns exactly the five tool schemas, in the fixed
    read/search/execute/edit/verify order, each the very same object as
    the owning tool module's own `*_SCHEMA` constant."""
    expected = (
        READ_FILE_SCHEMA,
        SEARCH_SCHEMA,
        EXECUTE_SCHEMA,
        EDIT_FILE_SCHEMA,
        VERIFY_SCHEMA,
    )

    schemas = all_schemas()

    assert schemas == expected
    assert all(actual is want for actual, want in zip(schemas, expected))


@pytest.mark.sanity
def test_dispatch_on_unknown_tool_name_returns_a_framed_result_not_an_exception(
    tmp_path: Path,
) -> None:
    """Given a `ToolCallEvent` naming a tool the registry does not
    know, when dispatched, then it returns a `ToolResult` naming the
    unrecognized tool rather than raising."""
    event = ToolCallEvent(id="call-1", name="frobnicate", arguments_json="{}")

    result = dispatch(event, repo_root=tmp_path)

    assert result.tool_call_id == "call-1"
    assert "frobnicate" in result.content


@pytest.mark.sanity
def test_dispatch_with_malformed_arguments_json_returns_a_framed_result(
    tmp_path: Path,
) -> None:
    """Given a known tool name but arguments_json that is not valid
    JSON, when dispatched, then it returns a `ToolResult` carrying the
    parser's own error message rather than raising."""
    event = ToolCallEvent(id="call-2", name="read_file", arguments_json="not json")

    result = dispatch(event, repo_root=tmp_path)

    assert result.tool_call_id == "call-2"
    assert "invalid JSON" in result.content


@pytest.mark.sanity
def test_dispatch_read_file_matches_calling_the_tool_directly(tmp_path: Path) -> None:
    """Given valid `read_file` arguments, when dispatched, then the
    result's content is exactly what calling `read_file` directly would
    return for the same arguments."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    event = ToolCallEvent(
        id="call-3",
        name="read_file",
        arguments_json=json.dumps({"path": "greet.py"}),
    )

    result = dispatch(event, repo_root=tmp_path)
    direct = read_file(ReadFileArgs(path="greet.py"), repo_root=tmp_path)

    assert result.tool_call_id == "call-3"
    assert result.content == direct


@pytest.mark.sanity
def test_dispatch_search_matches_calling_the_tool_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given valid `search` arguments and a stubbed `rg` invocation,
    when dispatched, then the result's content is exactly what calling
    `search` directly would return for the same arguments."""
    _stub_rg(monkeypatch, stdout="greet.py:1:print('hi')\n", returncode=0)
    event = ToolCallEvent(
        id="call-4",
        name="search",
        arguments_json=json.dumps({"pattern": "hi"}),
    )

    result = dispatch(event, repo_root=tmp_path)
    direct = search(SearchArgs(pattern="hi"), repo_root=tmp_path)

    assert result.tool_call_id == "call-4"
    assert result.content == direct


@pytest.mark.sanity
def test_dispatch_execute_matches_calling_the_tool_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given valid `execute` arguments, a stubbed sandbox, and an
    `ApprovalManager` passed through `**context`, when dispatched, then
    the result's content is exactly what calling `execute` directly
    would return for the same arguments."""
    _stub_run_sandboxed(monkeypatch)
    approval = ApprovalManager(decide_fn=lambda _request: "once")
    event = ToolCallEvent(
        id="call-5",
        name="execute",
        arguments_json=json.dumps({"cmd": ["echo", "hi"]}),
    )

    result = dispatch(event, repo_root=tmp_path, approval=approval)
    direct = execute(
        ExecuteArgs(cmd=("echo", "hi")), repo_root=tmp_path, approval=approval
    )

    assert result.tool_call_id == "call-5"
    assert result.content == direct


@pytest.mark.sanity
def test_dispatch_edit_file_matches_calling_the_tool_directly(tmp_path: Path) -> None:
    """Given valid `edit_file` arguments (a dry run, so both the
    dispatched and the direct call can target the same untouched file)
    and an `UndoManager`/`turn_id`/`task_id` passed through `**context`,
    when dispatched, then the result's content is exactly what calling
    `edit_file` directly would return for the same arguments."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    event = ToolCallEvent(
        id="call-6",
        name="edit_file",
        arguments_json=json.dumps(
            {"path": "greet.py", "old": "hi", "new": "world", "dry_run": True}
        ),
    )

    result = dispatch(event, repo_root=tmp_path, undo=undo, turn_id=1, task_id="t-1")
    direct = edit_file(
        EditFileArgs(path="greet.py", old="hi", new="world", dry_run=True),
        repo_root=tmp_path,
        undo=undo,
        turn_id=1,
        task_id="t-1",
    )

    assert result.tool_call_id == "call-6"
    assert result.content == direct


@pytest.mark.sanity
def test_dispatch_verify_matches_calling_the_tool_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a KESTREL.md configuring one command, a stubbed sandbox,
    and `task_id`/`turn_id` passed through `**context`, when dispatched,
    then the result's content is exactly what calling `verify` directly
    would return for the same arguments."""
    (tmp_path / "KESTREL.md").write_bytes(b'```kestrel-verify\nlint = "true"\n```\n')
    _stub_verify_run_sandboxed(monkeypatch)
    event = ToolCallEvent(id="call-7", name="verify", arguments_json="{}")

    result = dispatch(event, repo_root=tmp_path, task_id="t-1", turn_id=1)
    direct = verify(VerifyArgs(), repo_root=tmp_path, task_id="t-1", turn_id=1)

    assert result.tool_call_id == "call-7"
    assert result.content == direct


def test_dispatch_catches_read_file_error_from_the_executor(tmp_path: Path) -> None:
    """Given `read_file` arguments naming a file that does not exist,
    when dispatched, then the executor's own `ReadFileError` is caught
    and framed as the result rather than escaping `dispatch`."""
    event = ToolCallEvent(
        id="call-8",
        name="read_file",
        arguments_json=json.dumps({"path": "missing.py"}),
    )

    result = dispatch(event, repo_root=tmp_path)

    assert "no such file" in result.content


def test_dispatch_catches_search_error_from_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `search` arguments and a stubbed `rg` invocation that
    exits with a real error status, when dispatched, then the
    executor's own `SearchError` is caught and framed as the result
    rather than escaping `dispatch`."""
    _stub_rg(monkeypatch, stdout="", returncode=2, stderr="regex parse error")
    event = ToolCallEvent(
        id="call-9",
        name="search",
        arguments_json=json.dumps({"pattern": "("}),
    )

    result = dispatch(event, repo_root=tmp_path)

    assert "regex parse error" in result.content


def test_dispatch_catches_execute_error_from_the_parser(tmp_path: Path) -> None:
    """Given `execute` arguments with an empty `cmd` array, when
    dispatched, then the parser's own `ExecuteError` is caught and
    framed as the result rather than escaping `dispatch`."""
    event = ToolCallEvent(
        id="call-10",
        name="execute",
        arguments_json=json.dumps({"cmd": []}),
    )

    result = dispatch(event, repo_root=tmp_path)

    assert "cmd" in result.content


def test_dispatch_catches_edit_file_error_from_the_executor(tmp_path: Path) -> None:
    """Given `edit_file` arguments naming an anchor absent from the
    target file, when dispatched, then the executor's own
    `EditFileError` is caught and framed as the result rather than
    escaping `dispatch`."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    event = ToolCallEvent(
        id="call-11",
        name="edit_file",
        arguments_json=json.dumps(
            {"path": "greet.py", "old": "no-such-anchor", "new": "x"}
        ),
    )

    result = dispatch(event, repo_root=tmp_path, undo=undo, turn_id=1, task_id="t-1")

    assert "anchor not found" in result.content


def test_dispatch_catches_verify_error_from_the_executor(tmp_path: Path) -> None:
    """Given a repo root with no KESTREL.md, when `verify` is
    dispatched, then the executor's own `VerifyError` is caught and
    framed as the result rather than escaping `dispatch`."""
    event = ToolCallEvent(id="call-12", name="verify", arguments_json="{}")

    result = dispatch(event, repo_root=tmp_path, task_id="t-1", turn_id=1)

    assert "KESTREL.md" in result.content


def test_dispatch_raises_when_context_omits_an_executors_required_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a fake tool binding (constructed only in this test) whose
    executor requires a keyword-only argument beyond `repo_root`, when
    dispatched with a `context` that does not supply it, then the
    resulting `TypeError` propagates unchanged -- `dispatch` never
    silently omits a required argument instead of failing loudly."""

    def _needs_a_special_kwarg(args: object, *, repo_root: Path, special: str) -> str:
        raise AssertionError("should never run: the required kwarg is missing")

    fake_binding = registry_module._ToolBinding(
        schema=ToolSchema(
            name="fake_tool",
            description="A test double never offered to a real model.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        parse_args=lambda arguments_json: arguments_json,
        execute=_needs_a_special_kwarg,
        error_type=RuntimeError,
    )
    monkeypatch.setitem(registry_module._BY_NAME, "fake_tool", fake_binding)
    event = ToolCallEvent(id="call-13", name="fake_tool", arguments_json="{}")

    with pytest.raises(TypeError):
        dispatch(event, repo_root=tmp_path)
