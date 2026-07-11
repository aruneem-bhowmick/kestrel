"""System test: the agent loop driven end to end against a real
`LiteLLMClient`, a real mock chat-completions server replaying a
scripted three-turn cassette sequence, and a real fixture repository --
the loop reads a file, runs a command through the actual `bwrap`
sandbox, then stops, and the file's own content genuinely reaches the
second model call's request body.

Skipped locally when `bwrap` is not on `PATH`, exactly like
`tests/integration/test_p016_sandbox.py`: the binary genuinely may not
be installed, and never will be on a non-Linux host. CI installs
`bubblewrap` on every runner, so this suite always actually runs there.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.sandbox import bwrap_available

pytestmark = [
    pytest.mark.p022,
    pytest.mark.system,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EXECUTE_PYTEST = _CASSETTES / "toolcall_execute_pytest.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassettes' own `model` field."""
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={"glm-5.2": entry}, source=None)


async def test_loop_reads_a_file_runs_a_command_then_completes(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo with one small Python module and an empty
    test file, and a mock server scripted to reply with a `read_file`
    tool call, then an `execute` tool call, then a plain no-more-tools
    reply, when the task runs, then the loop reads the module, runs the
    scripted command for real through `bwrap`, and ends TASK_COMPLETE
    after three turns -- with the read file's own content genuinely
    present in the second model call's captured request body.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_greet.py").write_text("", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _TOOLCALL_EXECUTE_PYTEST,
            _DONE_CASSETTE,
        ],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    deps = LoopDeps(
        client=client,
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
    )

    result = await run_task(
        "read src/greet.py, then run the test suite", deps, task_id="sys-loop-1"
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    assert len(captured) == 3
    assert _FILE_MARKER.encode("utf-8") in captured[1]
