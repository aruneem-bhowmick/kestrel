"""Injectable, no-op-by-default hooks a running task's tool-calling loop
calls into at seven fixed points, so a task's own state transitions --
a turn starting, an incremental chunk of streamed text arriving, a
tool call starting or finishing, a fresh verification report landing,
a turn's own priced cost settling, and the task itself ending -- are
externally observable without `kestrel.agent.loop` knowing anything
about who, if anyone, is watching.

Every `LoopObserver` method is called synchronously and inline, on the
exact coroutine driving the task, so an implementation that blocks or
raises stalls -- or crashes -- the task itself; an implementation must
stay fast and exception-free. `NullLoopObserver` -- every method a
no-op -- is `LoopDeps.observer`'s own default, so every existing caller
that never sets `observer` sees zero behavior change from this
module's existence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from kestrel.cost.meter import TurnCost
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport

if TYPE_CHECKING:
    from kestrel.agent.loop import LoopResult


class LoopObserver(Protocol):
    """Callbacks `_drive` invokes at defined points in a running task,
    purely for external visibility -- no method's return value is
    read, and none influences control flow. Every method is called
    synchronously, inline, on the same coroutine driving the task, so
    an implementation that blocks or raises stalls (or crashes) the
    task itself; keep every method fast and exception-free.
    """

    def on_turn_started(self, *, turn_id: int, active_model_id: str) -> None:
        """A new turn is about to send its own model call, on
        `active_model_id` -- which may differ from the task's starting
        `LoopDeps.model_id` once a budget degrade has switched it."""
        ...

    def on_text_delta(self, text: str) -> None:
        """One incremental chunk of the assistant's own streamed text,
        in arrival order, as the current turn's model call streams."""
        ...

    def on_tool_call_started(self, call: ToolCallEvent) -> None:
        """`call` is about to be dispatched through the shared tool
        registry."""
        ...

    def on_tool_call_finished(self, call: ToolCallEvent, result: ToolResult) -> None:
        """`call` has finished dispatching, with `result` as its
        outcome -- fired whether the call succeeded, was refused by
        the approval gate, or named an unrecognized tool."""
        ...

    def on_verification(self, report: VerificationReport) -> None:
        """The tool call most recently reported via
        `on_tool_call_finished` recorded `report`, a fresh
        `VerificationReport`, as its own effect."""
        ...

    def on_turn_finished(
        self, *, turn_id: int, turn_cost: TurnCost, active_model_id: str
    ) -> None:
        """The turn identified by `turn_id` has had its own usage
        priced into `turn_cost` and journaled, when a session is
        configured for this task."""
        ...

    def on_termination(self, result: "LoopResult") -> None:
        """The task has ended; `result` is the exact `LoopResult` its
        caller is about to receive."""
        ...


class NullLoopObserver:
    """Every method is a no-op. `LoopDeps`'s own default -- every
    existing caller of `run_task`/`resume_task` that never sets
    `observer` sees zero behavior change from this module's
    existence."""

    def on_turn_started(self, *, turn_id: int, active_model_id: str) -> None:
        """Do nothing."""

    def on_text_delta(self, text: str) -> None:
        """Do nothing."""

    def on_tool_call_started(self, call: ToolCallEvent) -> None:
        """Do nothing."""

    def on_tool_call_finished(self, call: ToolCallEvent, result: ToolResult) -> None:
        """Do nothing."""

    def on_verification(self, report: VerificationReport) -> None:
        """Do nothing."""

    def on_turn_finished(
        self, *, turn_id: int, turn_cost: TurnCost, active_model_id: str
    ) -> None:
        """Do nothing."""

    def on_termination(self, result: "LoopResult") -> None:
        """Do nothing."""


NULL_OBSERVER: Final[LoopObserver] = NullLoopObserver()
