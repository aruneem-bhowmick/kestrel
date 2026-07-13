"""System test: a task's own turns are durably journaled as it runs
against a real `LiteLLMClient` and a real mock chat-completions server,
and a second, independent `LoopDeps`/`resume_task` call -- standing in
for a fresh process picking the task back up -- reconstructs that
journal and drives the task to completion, with the final history
containing every message from both the original and the resumed run's
own turns, in order.

No sandboxed tool is involved (the scripted turns are `read_file` and a
plain stop, neither of which touch `bwrap`), so unlike the verification-
gate and execute-backed system suites, this one has no platform
dependency to skip on.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import (
    LoopDeps,
    LoopLimits,
    TerminationReason,
    resume_task,
    run_task,
)
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.session import SessionManager, load_session
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p029, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
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


async def test_a_task_halted_at_a_turn_cap_resumes_in_a_fresh_process_and_completes(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a task whose own `deps.session` is set, run with
    `max_turns=1` against a mock server scripted to request a
    `read_file` tool call, when the task runs, then it stops TURN_CAP
    after exactly one journaled turn. Given a second, independent
    `LoopDeps` (standing in for a fresh process) and `resume_task`
    instead of `run_task` against a fresh mock server scripted to reply
    with a plain no-more-tools reply, when resumed, then the task
    completes TASK_COMPLETE after exactly one more real model call, and
    the final history contains every message from both runs, in order.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    first_captured: list[bytes] = []
    first_base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE], capture=first_captured
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", first_base_url)

    registry = _registry()
    session = SessionManager(repo_root=tmp_path, task_id="sys-p029-1")
    first_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        session=session,
        limits=LoopLimits(max_turns=1),
    )

    first_result = await run_task(
        "read src/greet.py, then run the test suite", first_deps, task_id="sys-p029-1"
    )

    assert first_result.reason == TerminationReason.TURN_CAP
    assert first_result.turns_used == 1
    assert len(first_captured) == 1

    state = load_session(tmp_path, "sys-p029-1")
    assert state.turns_used == 1
    # the seeded user task, the assistant's tool-call turn, and its tool
    # result -- turn 1's own journaled delta covers the seed too, since
    # nothing had been persisted before it.
    assert len(state.history) == 3
    assert state.history[0]["role"] == "user"

    second_captured: list[bytes] = []
    second_base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE], capture=second_captured
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", second_base_url)

    resumed_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        limits=LoopLimits(max_turns=10),
    )

    resumed_result = await resume_task("sys-p029-1", resumed_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 2
    assert len(second_captured) == 1

    tool_messages = [m for m in resumed_result.history if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert _FILE_MARKER in tool_messages[0]["content"]
    assistant_messages = [m for m in resumed_result.history if m["role"] == "assistant"]
    assert len(assistant_messages) == 2
    assert resumed_result.total_usd > first_result.total_usd


async def test_resume_task_on_an_unknown_task_id_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Given no session journal at all for a task id, when `resume_task`
    is called, then `FileNotFoundError` propagates unchanged."""
    registry = _registry()
    deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
    )

    with pytest.raises(FileNotFoundError):
        await resume_task("no-such-task", deps)
