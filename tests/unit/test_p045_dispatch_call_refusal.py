"""Unit tests for `kestrel.agent.loop._dispatch_tool_call`'s tool-access
guard: a call naming a tool outside `LoopDeps.available_tools` is
refused before the shared dispatcher ever runs, while `None` (every
tool allowed, the pre-existing default) and a name inside the allowlist
both dispatch exactly as they did before this guard existed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.loop import LoopDeps
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StreamEvent, ToolCallEvent
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.framing import frame_untrusted

pytestmark = [pytest.mark.p045, pytest.mark.unit]

_MODEL_ID = "glm-5.2"


class _UnusedClient:
    """Stands in for `ProviderClient` in a `LoopDeps` built only to
    exercise `_dispatch_tool_call` directly -- that function never
    reads `deps.client`, so calling this would be a test-authoring
    mistake."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Never called; raises if it somehow is."""
        raise AssertionError("_dispatch_tool_call must never call deps.client")
        yield  # pragma: no cover -- makes this an async generator


def _registry() -> Registry:
    """A single-entry `Registry`, matching this suite's siblings."""
    entry = ModelEntry(
        id=_MODEL_ID,
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
    return Registry(models={_MODEL_ID: entry}, source=None)


def _build_deps(
    repo_root: Path, *, undo: UndoManager, available_tools: frozenset[str] | None
) -> LoopDeps:
    """Assemble a `LoopDeps` bundle scoped to `repo_root`, varying only
    `available_tools` across this suite's cases."""
    return LoopDeps(
        client=_UnusedClient(),
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=undo,
        meter=CostMeter(),
        available_tools=available_tools,
    )


def _edit_greet_event(call_id: str) -> ToolCallEvent:
    """One `edit_file` call replacing `greet.py`'s `"hi"` with
    `"world"` -- shared across every case in this suite so only
    `available_tools` varies between them."""
    return ToolCallEvent(
        id=call_id,
        name="edit_file",
        arguments_json=json.dumps({"path": "greet.py", "old": "hi", "new": "world"}),
    )


def test_a_call_outside_the_allowlist_is_refused_and_the_file_is_untouched(
    tmp_path: Path,
) -> None:
    """Given `deps.available_tools` naming a set that excludes
    `edit_file`, when an `edit_file` call is dispatched, then it returns
    a framed refusal naming the excluded tool and the allowed set,
    exactly as `frame_untrusted` would produce it, and the target file
    on disk is untouched -- the guard fires before `dispatch` ever runs,
    so no executor, undo entry, or approval prompt is produced."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    deps = _build_deps(tmp_path, undo=undo, available_tools=frozenset({"read_file"}))
    event = _edit_greet_event("call-1")

    result = loop_module._dispatch_tool_call(event, deps=deps, turn_id=1, task_id="t-1")

    expected = frame_untrusted(
        "'edit_file' is not available in this mode; only ['read_file'] may be called.",
        source="tool_stderr",
        origin="edit_file",
    )
    assert result.tool_call_id == "call-1"
    assert result.content == expected
    assert (tmp_path / "greet.py").read_text(encoding="utf-8") == "print('hi')\n"


@pytest.mark.sanity
def test_available_tools_left_unset_dispatches_normally(tmp_path: Path) -> None:
    """Given `deps.available_tools=None` (the pre-existing default),
    when an `edit_file` call is dispatched, then it runs exactly as it
    did before this guard existed -- a regression pin proving the new
    guard doesn't break the unset case."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    deps = _build_deps(tmp_path, undo=undo, available_tools=None)
    event = _edit_greet_event("call-2")

    result = loop_module._dispatch_tool_call(event, deps=deps, turn_id=1, task_id="t-1")

    assert "is not available in this mode" not in result.content
    assert (tmp_path / "greet.py").read_text(encoding="utf-8") == "print('world')\n"


def test_a_call_naming_a_tool_in_the_allowlist_dispatches_normally(
    tmp_path: Path,
) -> None:
    """Given `deps.available_tools` naming exactly the called tool,
    when that call is dispatched, then it runs normally -- proving the
    guard is an allowlist gating everything outside the named set, not
    a denylist that happens to also block this one tool."""
    (tmp_path / "greet.py").write_text("print('hi')\n", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    deps = _build_deps(tmp_path, undo=undo, available_tools=frozenset({"edit_file"}))
    event = _edit_greet_event("call-3")

    result = loop_module._dispatch_tool_call(event, deps=deps, turn_id=1, task_id="t-1")

    assert "is not available in this mode" not in result.content
    assert (tmp_path / "greet.py").read_text(encoding="utf-8") == "print('world')\n"
