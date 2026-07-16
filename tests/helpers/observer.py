"""`RecordingObserver`: a `LoopObserver` test double that captures every
call it receives, in arrival order, for assertion -- shared between the
unit and system suites covering `LoopDeps.observer`'s wiring so both
read against the same call-recording shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kestrel.agent.loop import LoopResult
from kestrel.cost.meter import TurnCost
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport


@dataclass
class RecordingObserver:
    """A `LoopObserver` recording every call it receives, in arrival
    order, as `(method_name, payload)` pairs. `payload` is a `dict` of
    keyword arguments for a keyword-only method, or the positional
    argument(s) themselves otherwise -- whichever shape a test can
    assert against directly."""

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
        """Just the method names, in arrival order -- the shape a test
        asserts against when only call sequencing matters, not payload
        detail."""
        return [name for name, _ in self.calls]
