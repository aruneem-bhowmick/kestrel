"""Red-team proof that a hostile command's captured stdout still comes
back through `execute` wrapped by the real frame markers -- a
sandboxed command's own output cannot smuggle itself out of its own
frame.

Monkeypatches `run_sandboxed` rather than requiring a real `bwrap`
invocation: the property under test is `execute`'s own rendering and
framing of whatever a command produced, which is independent of
whether that output came from a real sandboxed process or a stand-in
one, and this mandatory case must run everywhere -- including a
`bwrap`-less developer machine or CI runner -- never skipped the way
`tests/integration/test_p016_sandbox.py`'s real-containment cases are.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from kestrel.managers.approval import ApprovalManager
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.execute import ExecuteArgs, execute
from kestrel.tools.sandbox import SandboxResult

pytestmark = [pytest.mark.p016, pytest.mark.unit, pytest.mark.redteam]

_HOSTILE_CASE_ID = "fake_destructive_action_success"

# See `test_p015_search_timeout.py` for why this module is resolved via
# `importlib` rather than `import kestrel.tools.execute as execute_module`:
# `kestrel.tools.__init__` rebinds the `execute` *attribute* on the
# `kestrel.tools` package to the function of the same name.
_execute_module = importlib.import_module("kestrel.tools.execute")


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
def hostile_stdout_case() -> InjectionCase:
    """The corpus case used to prove `execute`'s framing survives a
    real prompt-injection payload arriving as a command's stdout.
    Looked up lazily (rather than at module import time) so a lookup
    failure surfaces as a normal test error tied to whichever test
    requested it."""
    return _find_case(_HOSTILE_CASE_ID)


def test_hostile_command_stdout_still_carries_the_real_frame_markers(
    tmp_path: Path, hostile_stdout_case: InjectionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a sandboxed command whose stdout is one of the injection
    corpus's hostile payloads, when run through `execute`, then the
    result still starts with the real opening header, still ends with
    the real closing delimiter, and none of the case's forbidden
    markers appear more than the one time `frame_untrusted` itself
    emits them."""
    monkeypatch.setattr(
        _execute_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout=hostile_stdout_case.payload, stderr="", exit_code=0, timed_out=False
        ),
    )

    framed = execute(
        ExecuteArgs(cmd=("cat", "malicious-output.txt")),
        repo_root=tmp_path,
        approval=ApprovalManager(),
    )

    assert framed.startswith("<<<UNTRUSTED:tool_stdout:")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert hostile_stdout_case.payload in framed
    for marker in hostile_stdout_case.forbidden_markers:
        assert framed.count(marker) == 1


def test_execute_truncates_large_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a command producing stdout and stderr that exceed the 64KB cap,
    when run, then the result contains truncated streams with truncation notes,
    while exit_code and timed_out are correctly preserved."""
    # 64KB + 10 bytes
    large_size = 64 * 1024 + 10
    large_stdout = "a" * large_size
    large_stderr = "b" * large_size

    monkeypatch.setattr(
        _execute_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout=large_stdout,
            stderr=large_stderr,
            exit_code=42,
            timed_out=True,
        ),
    )

    framed = execute(
        ExecuteArgs(cmd=("dummy",)), repo_root=tmp_path, approval=ApprovalManager()
    )

    # We expect:
    # exit_code: 42
    # timed_out: True
    # stdout:
    # <64KB of 'a'>
    # ... [truncated: 10 more bytes omitted]
    # stderr:
    # <64KB of 'b'>
    # ... [truncated: 10 more bytes omitted]
    assert "exit_code: 42" in framed
    assert "timed_out: True" in framed
    assert (
        "stdout:\n" + ("a" * 65536) + "\n... [truncated: 10 more bytes omitted]"
        in framed
    )
    assert (
        "stderr:\n" + ("b" * 65536) + "\n... [truncated: 10 more bytes omitted]"
        in framed
    )


def test_execute_does_not_truncate_small_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a command producing stdout and stderr within the 64KB cap,
    when run, then the streams are returned exactly without truncation notes."""
    small_stdout = "hello stdout"
    small_stderr = "hello stderr"

    monkeypatch.setattr(
        _execute_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout=small_stdout,
            stderr=small_stderr,
            exit_code=0,
            timed_out=False,
        ),
    )

    framed = execute(
        ExecuteArgs(cmd=("dummy",)), repo_root=tmp_path, approval=ApprovalManager()
    )

    assert "exit_code: 0" in framed
    assert "timed_out: False" in framed
    assert "stdout:\nhello stdout" in framed
    assert "stderr:\nhello stderr" in framed
    assert "truncated" not in framed


def test_execute_handles_multibyte_truncation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a stream with a multibyte character split across the 64KB boundary,
    when truncated, then it drops the partial character without crashing and
    reports the correct number of omitted bytes."""
    # 65535 'a' characters (1 byte each) + '𠜎' (4 bytes)
    # Total UTF-8 encoded length: 65539 bytes
    large_stdout = ("a" * 65535) + "𠜎"

    monkeypatch.setattr(
        _execute_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout=large_stdout,
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )

    framed = execute(
        ExecuteArgs(cmd=("dummy",)), repo_root=tmp_path, approval=ApprovalManager()
    )

    # The 4-byte character is omitted entirely because it was cut in half,
    # so we should have 65535 'a's and 4 omitted bytes.
    assert (
        "stdout:\n" + ("a" * 65535) + "\n... [truncated: 4 more bytes omitted]"
        in framed
    )
