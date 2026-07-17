"""Unit tests for `kestrel.task_setup.build_task_deps`: every parameter
left at its default produces the exact same `LoopDeps` shape `cli.py`'s
own pre-extraction construction always did, and overriding any one
parameter is reflected in the returned `TaskSetup` without disturbing
the rest -- all hermetic and network-free, since nothing here ever
drives a real task (see `tests/system/test_p038_tui_conversation_stream.py`
for that).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopLimits
from kestrel.agent.observer import NULL_OBSERVER
from kestrel.config import ApprovalConfig, BudgetConfig, KestrelConfig, ManagersConfig
from kestrel.kestrel_md import KestrelMd, VerifyCommands
from kestrel.managers.approval import ApprovalDecision, ApprovalDenied, ApprovalRequest
from kestrel.managers.budget import BudgetLimits, BudgetStatus
from kestrel.managers.session import SessionManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry
from kestrel.task_setup import TaskSetup, build_task_deps

pytestmark = [pytest.mark.p038, pytest.mark.unit, pytest.mark.sanity]

_MODEL_ID = "glm-5.2"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry, standing in for
    whatever registry a real caller would have already loaded."""
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


@pytest.mark.sanity
def test_defaults_build_the_pre_extraction_loop_deps_shape(tmp_path: Path) -> None:
    """Given every optional parameter left at its default, when a task's
    deps are built, then the returned bundle matches `cli.py`'s own
    pre-extraction construction exactly: no verification gate, the
    null observer, an uncapped budget, a fresh cost meter, and the
    caller's own registry/model/repo threaded straight through."""
    registry = _registry()
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=registry,
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert isinstance(setup, TaskSetup)
    assert setup.deps.require_verification is False
    assert setup.deps.observer is NULL_OBSERVER
    assert setup.deps.model_id == _MODEL_ID
    assert setup.deps.repo_root == tmp_path
    assert setup.deps.kestrel_md is None
    assert setup.deps.limits == LoopLimits()
    assert isinstance(setup.deps.client, LiteLLMClient)
    assert isinstance(setup.deps.session, SessionManager)
    assert setup.deps.meter is setup.meter
    assert setup.deps.undo is setup.undo
    assert setup.deps.meter.turns == ()
    assert setup.budget_limits == BudgetLimits()
    assert setup.spent_day_usd == Decimal(0)
    assert setup.spent_month_usd == Decimal(0)

    # An uncapped `BudgetManager` (every field of `BudgetLimits()` is
    # `None`/its own default) never trips, regardless of how much spend
    # a check is given.
    event = setup.deps.budget.check(  # type: ignore[union-attr]
        spent_session=Decimal("999999"),
        spent_day=Decimal("999999"),
        spent_month=Decimal("999999"),
    )
    assert event.status is BudgetStatus.OK


def test_overriding_limits_is_reflected_in_deps(tmp_path: Path) -> None:
    """Given a non-default `LoopLimits`, when a task's deps are built,
    then `deps.limits` is that exact instance, not the default."""
    limits = LoopLimits(max_turns=3, max_total_tokens=1_000, max_wall_clock_s=30.0)
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        limits=limits,
    )

    assert setup.deps.limits == limits


def test_overriding_require_verification_is_reflected_in_deps(tmp_path: Path) -> None:
    """Given `require_verification=True`, when a task's deps are built,
    then `deps.require_verification` carries that override."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        require_verification=True,
    )

    assert setup.deps.require_verification is True


def test_overriding_budget_limits_is_reflected_in_deps_and_setup(
    tmp_path: Path,
) -> None:
    """Given an explicit `BudgetLimits`, when a task's deps are built,
    then both `TaskSetup.budget_limits` and `deps.budget`'s own
    classification reflect it, rather than `config.managers.budget`'s
    own (uncapped) defaults."""
    limits = BudgetLimits(session_usd=Decimal("5"))
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        budget_limits=limits,
    )

    assert setup.budget_limits == limits
    event = setup.deps.budget.check(  # type: ignore[union-attr]
        spent_session=Decimal("5"), spent_day=Decimal("5"), spent_month=Decimal("5")
    )
    assert event.status is BudgetStatus.HARD
    assert event.tripped_cap == "session"


def test_budget_limits_left_unset_falls_back_to_config(tmp_path: Path) -> None:
    """Given `budget_limits=None` (the default) and a config that itself
    configures a session cap, when a task's deps are built, then the
    resolved cap comes from `config.managers.budget`, not an uncapped
    fallback."""
    config = KestrelConfig(
        managers=ManagersConfig(budget=BudgetConfig(session_usd=Decimal("2")))
    )
    setup = build_task_deps(
        config=config,
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert setup.budget_limits.session_usd == Decimal("2")


def test_overriding_decide_fn_is_wired_into_approval(tmp_path: Path) -> None:
    """Given a custom `decide_fn`, when a task's deps are built and a
    non-allowlisted destructive action is checked, then the custom
    function -- not the real stdin prompt -- decides it."""
    calls: list[ApprovalRequest] = []

    def _spy_decide(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return "once"

    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        decide_fn=_spy_decide,
    )

    request = ApprovalRequest(kind="delete", summary="rm -rf x", detail="rm -rf x")
    setup.deps.approval.check(request)

    assert calls == [request]


def test_allowlisted_kind_never_reaches_decide_fn(tmp_path: Path) -> None:
    """Given `config.managers.approval.allowlist` names a kind, when a
    task's deps are built and a request of that kind is checked, then
    `decide_fn` is never called -- the allowlist short-circuits it."""
    calls: list[ApprovalRequest] = []

    def _spy_decide(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return "deny"

    config = KestrelConfig(
        managers=ManagersConfig(approval=ApprovalConfig(allowlist=("delete",)))
    )
    setup = build_task_deps(
        config=config,
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        decide_fn=_spy_decide,
    )

    setup.deps.approval.check(
        ApprovalRequest(kind="delete", summary="rm -rf x", detail="rm -rf x")
    )

    assert calls == []


def test_denied_decision_raises_approval_denied(tmp_path: Path) -> None:
    """Given a `decide_fn` that denies, when a non-allowlisted request
    is checked, then `ApprovalDenied` propagates -- proving the override
    genuinely governs the manager's decision, not just its input."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        decide_fn=lambda request: "deny",
    )

    with pytest.raises(ApprovalDenied):
        setup.deps.approval.check(
            ApprovalRequest(kind="delete", summary="rm -rf x", detail="rm -rf x")
        )


def test_overriding_observer_is_reflected_in_deps(tmp_path: Path) -> None:
    """Given a custom observer object, when a task's deps are built,
    then `deps.observer` is that exact instance, not `NULL_OBSERVER`."""

    class _MarkerObserver:
        """A stand-in observer distinguishable by identity alone."""

    marker = _MarkerObserver()
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
        observer=marker,  # type: ignore[arg-type]
    )

    assert setup.deps.observer is marker


def test_overriding_kestrel_md_is_reflected_in_deps(tmp_path: Path) -> None:
    """Given a real `KestrelMd`, when a task's deps are built, then
    `deps.kestrel_md` is that exact instance."""
    kestrel_md = KestrelMd(raw_text="# hi", verify_commands=VerifyCommands())
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=kestrel_md,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert setup.deps.kestrel_md is kestrel_md


def test_session_is_scoped_to_task_id(tmp_path: Path) -> None:
    """Given two different task ids, when each builds its own deps
    against the same repo, then each gets a `SessionManager` journaling
    to a distinct, task-id-named journal file."""
    setup_a = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-a",
    )
    setup_b = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-b",
    )

    assert setup_a.deps.session is not None
    assert setup_b.deps.session is not None
    assert setup_a.deps.session.journal_path != setup_b.deps.session.journal_path
    assert setup_a.deps.session.journal_path.stem == "task-a"
    assert setup_b.deps.session.journal_path.stem == "task-b"
