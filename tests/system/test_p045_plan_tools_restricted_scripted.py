"""System test: a task's `LoopDeps.available_tools` restriction, driven
end to end against a real `LiteLLMClient` and a real mock
chat-completions server replaying a scripted cassette that requests a
now-restricted tool -- proving a refused call neither crashes nor hangs
the loop, that the target file is left byte-identical, and that the
refusal text itself is what actually lands in the folded-in tool-role
message, not just a value asserted in isolation the way
`test_p045_dispatch_call_refusal.py`'s unit suite does.
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

pytestmark = [pytest.mark.p045, pytest.mark.system, pytest.mark.redteam]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_GREET_ANCHOR = "# TODO: implement greet\n"


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


async def test_a_restricted_call_is_refused_and_the_task_still_completes(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `deps.available_tools` excluding `edit_file`, a fixture repo
    whose `greet.py` still carries the anchor an `edit_file` call would
    otherwise replace, and a mock server scripted to reply with that
    `edit_file` call followed by a plain no-more-tools reply, when the
    task runs, then the loop refuses the call without dispatching it,
    the file on disk is left byte-identical, the task still reaches
    TASK_COMPLETE after exactly two turns, and the refusal text is what
    the tool-role message folded into history for that call actually
    carries.
    """
    (tmp_path / "greet.py").write_text(_GREET_ANCHOR, encoding="utf-8")
    original_bytes = (tmp_path / "greet.py").read_bytes()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_EDIT_GREET, _DONE_CASSETTE],
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
        available_tools=frozenset({"read_file", "search"}),
    )

    result = await run_task(
        "implement greet in greet.py", deps, task_id="sys-p045-plan-restrict"
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert len(captured) == 2
    assert (tmp_path / "greet.py").read_bytes() == original_bytes

    tool_messages = [message for message in result.history if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "is not available in this mode" in tool_messages[0]["content"]
    assert "edit_file" in tool_messages[0]["content"]
