"""System test: `LoopDeps.kb`, built for real by
`kestrel.task_setup.build_task_deps` with `[kb].enabled` at its default,
reaches a dispatched tool call's own `context` when the whole chain is
driven through a real `run_task` call -- not just `_dispatch_tool_call`
directly, the way `tests/unit/test_p059_loop_deps_kb.py` proves the
same wiring in isolation.

Reuses `test_p022_loop_scripted_task.py`'s own mock-server harness, with
a small test-only stub tool (declaring a keyword-only `kb` parameter,
which no real tool does yet) standing in for whichever real tool
eventually reads it. Self-critique is disabled for this scenario so the
only model call the task itself makes is its own scripted turn -- this
suite is about the knowledge-base wiring, not the self-critique routing
already covered elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import kestrel.tools.registry as registry_module
from kestrel.agent.loop import TerminationReason, run_task
from kestrel.config import KestrelConfig, ManagersConfig, SelfCritiqueConfig
from kestrel.kb.service import KbService
from kestrel.provider.base import ToolSchema
from kestrel.registry.loader import load_registry
from kestrel.task_setup import build_task_deps
from kestrel.tools.registry import _ToolBinding

pytestmark = [pytest.mark.p059, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_KB_PROBE = _CASSETTES / "toolcall_kb_probe.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_STUB_TOOL_NAME = "kb_probe"


class _StubArgs:
    """Empty argument object -- the stub tool's own parser always
    returns one of these, regardless of what a call's own JSON names."""


class _StubToolError(Exception):
    """The stub tool's own error type -- declared to satisfy
    `_ToolBinding`'s own shape, never actually raised."""


def _parse_stub_args(raw: str) -> _StubArgs:
    """Ignore `raw` entirely and return a fresh `_StubArgs` -- this stub
    tool takes no real arguments."""
    return _StubArgs()


_STUB_SCHEMA = ToolSchema(
    name=_STUB_TOOL_NAME,
    description="test-only stub declaring a keyword-only kb parameter",
    parameters={"type": "object", "properties": {}},
)


@pytest.fixture
def stub_kb_calls(monkeypatch: pytest.MonkeyPatch) -> list[KbService | None]:
    """Register `_STUB_TOOL_NAME` in the shared tool registry for the
    duration of one test, its executor appending whatever it receives
    as `kb` to the returned list -- see
    `tests/unit/test_p059_loop_deps_kb.py`'s own identical fixture for
    the full rationale. `monkeypatch` restores the original
    `_TOOLS`/`_BY_NAME` once the test ends.
    """
    received: list[KbService | None] = []

    def _stub_execute(
        args: _StubArgs, *, repo_root: Path, kb: KbService | None = None
    ) -> str:
        """Record `kb` and return a fixed, otherwise-uninteresting
        result string."""
        received.append(kb)
        return "stub-ok"

    binding = _ToolBinding(
        _STUB_SCHEMA, _parse_stub_args, _stub_execute, _StubToolError
    )
    monkeypatch.setattr(registry_module, "_TOOLS", (*registry_module._TOOLS, binding))
    monkeypatch.setattr(
        registry_module,
        "_BY_NAME",
        {**registry_module._BY_NAME, _STUB_TOOL_NAME: binding},
    )
    return received


async def test_a_real_run_task_call_threads_build_task_deps_kb_into_dispatch(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    stub_kb_calls: list[KbService | None],
) -> None:
    """Given deps built by `build_task_deps` against the packaged
    default registry, with `[kb].enabled` at its default and
    self-critique disabled, when the task runs against a mock server
    scripted to call the stub tool then declare itself done, then the
    stub tool's own executor received a real, non-`None` `KbService` --
    proving the full chain from `build_task_deps` through `LoopDeps.kb`
    to a dispatched tool call's own `context` genuinely holds, not just
    each of its links in isolation.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_KB_PROBE, _DONE_CASSETTE]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = load_registry()
    config = KestrelConfig(
        managers=ManagersConfig(self_critique=SelfCritiqueConfig(enabled=False))
    )
    setup = build_task_deps(
        config=config,
        registry=registry,
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="sys-kb-1",
    )
    assert setup.deps.kb is not None

    result = await run_task("probe the knowledge base", setup.deps, task_id="sys-kb-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert stub_kb_calls == [setup.deps.kb]
