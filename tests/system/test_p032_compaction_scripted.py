"""System test: compaction driven end to end against a real
`LiteLLMClient` and a real mock chat-completions server -- proving an
actual HTTP round trip carries the fold, not just that
`kestrel.agent.compaction.compact_history` behaves correctly in
isolation.

A small `context_window` test registry entry is engineered so that a
single real tool-calling turn's own billed prompt size already crosses
70% of it. The pre-check ahead of the very next turn therefore attempts
a compaction fold immediately -- but with `history` still only three
messages long (the seed task plus one turn's own assistant/tool pair),
that first attempt is `compact_history`'s own documented no-op (nothing
older than the default four-message kept tail exists yet to fold, so no
model call is made). Only once a second tool-calling turn has grown
`history` past that four-message mark does the pre-check ahead of the
third turn produce a real fold, consuming the mock server's own
dedicated compaction cassette before the third turn's real Think call
ever fires.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.compaction import _COMPACTION_SYSTEM_PROMPT
from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p032, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_COMPACTION_CASSETTE = _CASSETTES / "compaction_summary.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"

# Each `_TOOLCALL_READ_FILE` turn bills prompt_tokens=55; a context_window
# of 70 puts that turn's own ratio at 55/70 ≈ 0.786, comfortably past the
# 70% compaction threshold after either real turn.
_CONTEXT_WINDOW = 70


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassettes' own `model` field, with a deliberately small
    `context_window`."""
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=_CONTEXT_WINDOW,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={"glm-5.2": entry}, source=None)


async def test_a_real_compaction_call_folds_history_before_the_closing_turn(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given two scripted `read_file` turns that push the tiny test
    registry entry's ratio past 70%, followed by the mock server's own
    compaction cassette and a closing no-more-tools reply, when the task
    runs, then it makes exactly four real HTTP calls -- two tool turns,
    one compaction fold, and the closing turn -- completes
    `TASK_COMPLETE` with `turns_used == 3` (the fold is never counted as
    a turn), and the closing turn's own request body carries the
    compaction's rendered summary text, proving the fold reached the
    wire rather than only living in an in-memory list.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _TOOLCALL_READ_FILE,
            _COMPACTION_CASSETTE,
            _DONE_CASSETTE,
        ],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

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

    result = await run_task(
        "read src/greet.py, then run the test suite", deps, task_id="sys-p032-1"
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    assert len(captured) == 4

    compaction_request = json.loads(captured[2])
    compaction_system_message = compaction_request["messages"][0]
    assert compaction_system_message["role"] == "system"
    assert compaction_system_message["content"] == _COMPACTION_SYSTEM_PROMPT

    closing_request = json.loads(captured[3])
    closing_messages = closing_request["messages"]
    assert any("Summary:" in message.get("content", "") for message in closing_messages)

    tool_messages = [m for m in result.history if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert _FILE_MARKER in tool_messages[0]["content"]
