"""Unit tests for `kestrel.agent.observer`'s standalone contract:
`NullLoopObserver` -- `LoopDeps.observer`'s own default -- is callable
at every one of `LoopObserver`'s seven points with no exception, and
returns `None` from each, regardless of what it is handed.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.agent.observer import NULL_OBSERVER, NullLoopObserver
from kestrel.cost.meter import TurnCost
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.registry import ToolResult
from kestrel.tools.verify import VerificationReport

pytestmark = [pytest.mark.p036, pytest.mark.unit, pytest.mark.sanity]


def _tool_call() -> ToolCallEvent:
    """A minimal `ToolCallEvent`, standing in for whatever a real turn
    might have requested."""
    return ToolCallEvent(id="call-1", name="read_file", arguments_json="{}")


def _tool_result() -> ToolResult:
    """A minimal `ToolResult`, standing in for whatever a real dispatch
    might have returned."""
    return ToolResult(tool_call_id="call-1", content="ok")


def _turn_cost() -> TurnCost:
    """A minimal `TurnCost`, standing in for whatever a real turn might
    have been priced at."""
    return TurnCost(
        model_id="glm-5.2",
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        usd=Decimal("0.000010"),
    )


def _verification_report() -> VerificationReport:
    """A minimal, empty-commands `VerificationReport`, standing in for
    whatever a real `verify` call might have recorded."""
    return VerificationReport(task_id="t-1", turn_id=1, commands=(), passed=True)


def _loop_result() -> LoopResult:
    """A minimal `LoopResult`, standing in for whatever a real task
    might have ended with."""
    return LoopResult(
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0"),
        history=(),
    )


def test_null_observer_every_method_is_callable_and_returns_none() -> None:
    """Given a fresh `NullLoopObserver`, when every one of its seven
    methods is called with a representative argument, then each
    returns `None` without raising."""
    observer = NullLoopObserver()

    assert observer.on_turn_started(turn_id=1, active_model_id="glm-5.2") is None
    assert observer.on_text_delta("hello") is None
    assert observer.on_tool_call_started(_tool_call()) is None
    assert observer.on_tool_call_finished(_tool_call(), _tool_result()) is None
    assert observer.on_verification(_verification_report()) is None
    assert (
        observer.on_turn_finished(
            turn_id=1, turn_cost=_turn_cost(), active_model_id="glm-5.2"
        )
        is None
    )
    assert observer.on_termination(_loop_result()) is None


def test_null_observer_constant_is_the_same_shared_instance() -> None:
    """`NULL_OBSERVER` -- the module-level constant `LoopDeps.observer`
    defaults to -- is itself a `NullLoopObserver`, so a caller never
    needs to construct one just to get the no-op default."""
    assert isinstance(NULL_OBSERVER, NullLoopObserver)
