"""System test: the byte-stable cache prefix driven end to end against a
real `LiteLLMClient` and a real mock chat-completions server -- proving
the leading system message an actual HTTP request sends is identical
across two turns of one task, not just that the builder function is
stable in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.kestrel_md import load_kestrel_md
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p027, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"


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


async def test_leading_system_message_is_byte_identical_across_turns(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo with a `KESTREL.md` and a mock server
    scripted to reply with a `read_file` tool call followed by a plain
    no-more-tools reply, when the task runs, then the leading system
    message inside the second request's captured body renders to the
    exact same JSON, byte for byte, as the first request's -- the real
    prefix a cache-capable backend would see is stable, not just the
    builder function that assembles it.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "KESTREL.md").write_text(
        "# Conventions\n\nKeep changes small.\n", encoding="utf-8"
    )
    kestrel_md = load_kestrel_md(tmp_path)
    assert kestrel_md is not None

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE, _DONE_CASSETTE],
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
        kestrel_md=kestrel_md,
    )

    result = await run_task("read src/greet.py, then stop", deps, task_id="sys-p027-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert len(captured) == 2

    first_system_message = json.loads(captured[0])["messages"][0]
    second_system_message = json.loads(captured[1])["messages"][0]

    assert first_system_message["role"] == "system"
    assert second_system_message["role"] == "system"
    assert json.dumps(first_system_message, sort_keys=True) == json.dumps(
        second_system_message, sort_keys=True
    )
    assert kestrel_md.raw_text in first_system_message["content"]
