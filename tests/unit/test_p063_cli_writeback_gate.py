"""Unit tests for `kestrel.cli._propose_and_commit_learnings`'s own
no-op gate: whether the knowledge base is wired at all, and whether the
just-finished task actually completed cleanly, decide whether a
writeback proposal call is even attempted. Exercised against a fake
`propose_learnings` recording its own call count, so a count of zero
proves the gate short-circuited before ever reaching a model call, not
merely that nothing ended up committed.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.cli as cli_module
from kestrel.agent.loop import LoopDeps, LoopResult, TerminationReason
from kestrel.config import KestrelConfig
from kestrel.cost.meter import CostMeter
from kestrel.kb.writeback import ProposedLearning
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.budget import BudgetLimits
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import Registry
from kestrel.task_setup import TaskSetup

pytestmark = [pytest.mark.p063, pytest.mark.unit, pytest.mark.sanity]


class _UntouchedKbSentinel:
    """Stands in for a real `KbService` in the "kb is wired but the
    task did not complete" scenario below -- the gate under test
    returns before ever reading any of its attributes, so a bare
    sentinel proves that, whereas a real `KbService` would silently
    tolerate being touched and prove nothing."""


def _result(reason: TerminationReason) -> LoopResult:
    """A `LoopResult` carrying `reason`, otherwise arbitrary -- every
    field but `reason` is irrelevant to the gate this suite exercises."""
    return LoopResult(
        reason=reason, turns_used=1, total_usd=Decimal("0.01"), history=()
    )


def _setup(tmp_path: Path, *, kb: object | None) -> TaskSetup:
    """A minimal, real `TaskSetup` scoped to `tmp_path`, varying only
    `deps.kb` -- every other collaborator is real but never actually
    exercised by the no-op gate this suite covers."""
    registry = Registry(models={}, source=None)
    deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        kb=kb,  # type: ignore[arg-type]
    )
    return TaskSetup(
        deps=deps,
        undo=deps.undo,
        meter=deps.meter,
        budget_limits=BudgetLimits(),
        spent_day_usd=Decimal("0"),
        spent_month_usd=Decimal("0"),
    )


@pytest.fixture(autouse=True)
def _fail_if_propose_learnings_is_ever_called(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Replace `kestrel.cli.propose_learnings` with a fake recording its
    own call count, yielded for a test to assert against -- every
    scenario in this module expects that count to stay zero."""
    calls: list[int] = []

    async def _fake_propose_learnings(
        *args: object, **kwargs: object
    ) -> tuple[ProposedLearning, ...]:
        """Record that it was called, then answer as if nothing durable
        happened -- never reached by any scenario in this module."""
        calls.append(1)
        return ()

    monkeypatch.setattr(cli_module, "propose_learnings", _fake_propose_learnings)
    return calls


def test_noop_when_kb_is_none(
    tmp_path: Path, _fail_if_propose_learnings_is_ever_called: list[int]
) -> None:
    """Given `setup.deps.kb is None` (the knowledge base disabled) and a
    task that otherwise completed cleanly, when
    `_propose_and_commit_learnings` runs, then `propose_learnings` is
    never called -- a disabled knowledge base short-circuits before any
    model call is even attempted."""
    setup = _setup(tmp_path, kb=None)

    cli_module._propose_and_commit_learnings(
        _result(TerminationReason.TASK_COMPLETE),
        task_id="t-1",
        setup=setup,
        config=KestrelConfig(),
        registry=setup.deps.registry,
    )

    assert _fail_if_propose_learnings_is_ever_called == []


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.TURN_CAP,
        TerminationReason.TOKEN_CAP,
        TerminationReason.WALL_CLOCK_CAP,
        TerminationReason.CONTEXT_OVERFLOW,
        TerminationReason.USER_STOP,
        TerminationReason.BUDGET_HALT,
    ],
)
def test_noop_when_reason_is_not_task_complete(
    tmp_path: Path,
    reason: TerminationReason,
    _fail_if_propose_learnings_is_ever_called: list[int],
) -> None:
    """Given a knowledge base that is wired (`setup.deps.kb` is not
    `None`) but a task that ended on any reason other than
    `TASK_COMPLETE`, when `_propose_and_commit_learnings` runs, then
    `propose_learnings` is never called -- a task that did not finish
    cleanly has nothing durable to propose, regardless of whether the
    knowledge base itself is available."""
    setup = _setup(tmp_path, kb=_UntouchedKbSentinel())

    cli_module._propose_and_commit_learnings(
        _result(reason),
        task_id="t-1",
        setup=setup,
        config=KestrelConfig(),
        registry=setup.deps.registry,
    )

    assert _fail_if_propose_learnings_is_ever_called == []
