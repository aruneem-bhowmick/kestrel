"""System test: a task running against a real `LiteLLMClient` and a real
mock chat-completions server halts `BUDGET_HALT` once its hard session
cap trips, with every turn up to and including the tripping one already
durably journaled, and a second, independent `LoopDeps`/`resume_task`
call -- standing in for a fresh process picking the task back up with a
raised cap -- completes it against the remaining scripted cassette.

Mirrors `test_p029_session_persist_resume.py`'s own shape (real client,
real mock server, real `SessionManager`), adding a `BudgetManager` to
`LoopDeps` so the halt/resume path is proven against a real model call
and a real tool dispatch rather than a scripted fake.
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
from kestrel.managers.budget import BudgetLimits, BudgetManager
from kestrel.managers.session import SessionManager, load_session
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p031, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"

# Each `_TOOLCALL_READ_FILE` turn bills prompt_tokens=55, completion_tokens=12
# at glm-5.2's own $0.60/$2.20-per-Mtok rates: (55*0.60 + 12*2.20) / 1e6 =
# $0.0000594 per turn -- two turns cross a $0.0001 session cap on the second
# one without crossing it on the first.
_SESSION_CAP = Decimal("0.0001")


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


async def test_a_hard_session_cap_halts_the_task_and_a_raised_cap_resumes_it(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a task whose `deps.budget` carries a hard session cap that
    the second of two scripted `read_file` turns crosses, when the task
    runs, then it stops `BUDGET_HALT` after exactly two real model
    calls -- never attempting a third. Given a second, independent
    `LoopDeps` (standing in for a fresh process) with the cap raised and
    `resume_task` instead of `run_task` against a fresh mock server
    scripted to reply with a plain no-more-tools reply, when resumed,
    then the task completes `TASK_COMPLETE` after exactly one more real
    model call, continuing the turn counter rather than resetting it.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    first_captured: list[bytes] = []
    first_base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE, _TOOLCALL_READ_FILE],
        capture=first_captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", first_base_url)

    registry = _registry()
    session = SessionManager(repo_root=tmp_path, task_id="sys-p031-1")
    first_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        session=session,
        budget=BudgetManager(limits=BudgetLimits(session_usd=_SESSION_CAP)),
        limits=LoopLimits(max_turns=10),
    )

    first_result = await run_task(
        "read src/greet.py, then run the test suite", first_deps, task_id="sys-p031-1"
    )

    assert first_result.reason == TerminationReason.BUDGET_HALT
    assert first_result.turns_used == 2
    assert len(first_captured) == 2

    state = load_session(tmp_path, "sys-p031-1")
    assert state.turns_used == 2

    second_captured: list[bytes] = []
    second_base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE], capture=second_captured
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", second_base_url)

    resumed_session = SessionManager(repo_root=tmp_path, task_id="sys-p031-1")
    resumed_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        session=resumed_session,
        budget=BudgetManager(limits=BudgetLimits(session_usd=Decimal("100"))),
        limits=LoopLimits(max_turns=10),
    )

    resumed_result = await resume_task("sys-p031-1", resumed_deps)

    assert resumed_result.reason == TerminationReason.TASK_COMPLETE
    assert resumed_result.turns_used == 3
    assert len(second_captured) == 1

    final_state = load_session(tmp_path, "sys-p031-1")
    assert final_state.turns_used == 3

    tool_messages = [m for m in resumed_result.history if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert _FILE_MARKER in tool_messages[0]["content"]
    assert resumed_result.total_usd > first_result.total_usd
