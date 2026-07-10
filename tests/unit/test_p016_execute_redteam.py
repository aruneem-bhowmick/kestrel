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
        ExecuteArgs(cmd=("cat", "malicious-output.txt")), repo_root=tmp_path
    )

    assert framed.startswith("<<<UNTRUSTED:tool_stdout:")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert hostile_stdout_case.payload in framed
    for marker in hostile_stdout_case.forbidden_markers:
        assert framed.count(marker) == 1
