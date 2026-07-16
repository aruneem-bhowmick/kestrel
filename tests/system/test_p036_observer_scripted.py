"""System test: `LoopDeps.observer`'s hooks fire in the same shape this
suite's unit tests already prove, but driven by a real `LiteLLMClient`
against a real mock chat-completions server and a real `bwrap` sandbox
rather than a scripted fake client -- reusing
`test_p022_loop_scripted_task.py`'s own three-turn cassette sequence
(read a file, run a command, then stop) so the same fixture proves both
the loop's own behavior and its observer wiring.

Skipped locally when `bwrap` is not on `PATH`, exactly like
`test_p022_loop_scripted_task.py`: the binary genuinely may not be
installed, and never will be on a non-Linux host. CI installs
`bubblewrap` on every runner, so this suite always actually runs there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, LoopResult, TerminationReason, run_task
from kestrel.cost.meter import CostMeter, TurnCost
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.registry import ToolResult
from kestrel.tools.sandbox import bwrap_available
from kestrel.tools.verify import VerificationReport

pytestmark = [
    pytest.mark.p036,
    pytest.mark.system,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EXECUTE_PYTEST = _CASSETTES / "toolcall_execute_pytest.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"


@dataclass
class RecordingObserver:
    """A `LoopObserver` recording every call it receives, in arrival
    order, as `(method_name, payload)` pairs -- the same shape
    `test_p036_loop_observer_hooks.py`'s own test double uses, so a
    call-sequence assertion here reads identically to that suite's."""

    calls: list[tuple[str, object]] = field(default_factory=list)

    def on_turn_started(self, *, turn_id: int, active_model_id: str) -> None:
        """Record this call."""
        self.calls.append(
            (
                "on_turn_started",
                {"turn_id": turn_id, "active_model_id": active_model_id},
            )
        )

    def on_text_delta(self, text: str) -> None:
        """Record this call."""
        self.calls.append(("on_text_delta", text))

    def on_tool_call_started(self, call: ToolCallEvent) -> None:
        """Record this call."""
        self.calls.append(("on_tool_call_started", call))

    def on_tool_call_finished(self, call: ToolCallEvent, result: ToolResult) -> None:
        """Record this call."""
        self.calls.append(("on_tool_call_finished", (call, result)))

    def on_verification(self, report: VerificationReport) -> None:
        """Record this call."""
        self.calls.append(("on_verification", report))

    def on_turn_finished(
        self, *, turn_id: int, turn_cost: TurnCost, active_model_id: str
    ) -> None:
        """Record this call."""
        self.calls.append(
            (
                "on_turn_finished",
                {
                    "turn_id": turn_id,
                    "turn_cost": turn_cost,
                    "active_model_id": active_model_id,
                },
            )
        )

    def on_termination(self, result: LoopResult) -> None:
        """Record this call."""
        self.calls.append(("on_termination", result))

    def names(self) -> list[str]:
        """Just the method names, in arrival order."""
        return [name for name, _ in self.calls]


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


async def test_observer_hooks_fire_in_order_around_a_real_streamed_sandboxed_task(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the same fixture repo and scripted cassette sequence
    `test_p022_loop_scripted_task.py` drives (read a file, run a
    command through a real `bwrap` sandbox, then stop), when the task
    runs with a `RecordingObserver` wired in, then it sees three
    `on_turn_started`/`on_turn_finished` pairs, one `on_tool_call_started`/
    `on_tool_call_finished` pair per tool call with neither pair
    interleaving the other, and exactly one `on_termination` call --
    the task's very last -- reporting `TASK_COMPLETE`.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_greet.py").write_text("", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _TOOLCALL_EXECUTE_PYTEST,
            _DONE_CASSETTE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    observer = RecordingObserver()
    deps = LoopDeps(
        client=client,
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        observer=observer,
    )

    result = await run_task(
        "read src/greet.py, then run the test suite", deps, task_id="sys-p036-1"
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3

    names = observer.names()
    assert names.count("on_turn_started") == 3
    assert names.count("on_turn_finished") == 3
    assert names.count("on_termination") == 1
    assert names[-1] == "on_termination"
    termination_result = observer.calls[-1][1]
    assert isinstance(termination_result, LoopResult)
    assert termination_result.reason == TerminationReason.TASK_COMPLETE

    tool_events = [
        entry for entry in observer.calls if entry[0].startswith("on_tool_call")
    ]
    assert [name for name, _ in tool_events] == [
        "on_tool_call_started",
        "on_tool_call_finished",
        "on_tool_call_started",
        "on_tool_call_finished",
    ]
    first_started_call = tool_events[0][1]
    first_finished_call, _ = tool_events[1][1]
    second_started_call = tool_events[2][1]
    second_finished_call, _ = tool_events[3][1]
    assert first_started_call.id == first_finished_call.id
    assert second_started_call.id == second_finished_call.id
    assert first_started_call.name == "read_file"
    assert second_started_call.name == "execute"
