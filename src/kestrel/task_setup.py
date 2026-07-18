"""Build the collaborator bundle one task needs, independent of how that
task was requested.

`cli.py`'s own `kestrel run` and the TUI's own task-submission handler both
need the exact same `LoopDeps` bundle -- a provider client, an approval
gate, an undo journal, a cost meter, a session journal, and a budget
manager, all wired to the same repo and the same model -- built from the
same rules. Before this module existed, only `cli.py` could build one,
and it did so straight out of an `argparse.Namespace`, which nothing
outside the CLI could construct without faking one up. `build_task_deps`
is that construction logic pulled out from under the `Namespace`
coupling, so any caller -- the CLI included -- builds an identical bundle
from plain keyword arguments instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from kestrel.agent.critique import make_self_critique_fn
from kestrel.agent.loop import LoopDeps, LoopLimits, _default_self_critique
from kestrel.agent.observer import NULL_OBSERVER, LoopObserver
from kestrel.config import KestrelConfig
from kestrel.cost.meter import CostMeter
from kestrel.kestrel_md import KestrelMd
from kestrel.managers.approval import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalRequest,
    _prompt_stdin,
)
from kestrel.managers.budget import BudgetLimits, BudgetManager
from kestrel.managers.mode import ModeManager
from kestrel.managers.session import SessionManager, aggregate_historical_spend
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import Registry
from kestrel.router.policy import resolve_model_id

# Time windows `aggregate_historical_spend` sums a task's day/month
# spend baseline over. The month window is a fixed 30-day approximation
# rather than a real calendar month (leap years, 28-31 day months) --
# close enough for a budget baseline that only needs to roughly bound
# "this month's spend so far," not reproduce a billing statement.
_DAY_WINDOW_S = 24.0 * 60.0 * 60.0
_MONTH_WINDOW_S = 30.0 * _DAY_WINDOW_S


@dataclass(frozen=True, slots=True)
class TaskSetup:
    """The collaborator bundle one task's own caller needs, both to
    drive it and to report on it afterward.

    Attributes:
        deps: The `LoopDeps` bundle to drive the task with.
        undo: The task's own `UndoManager`, readable again after the
            run (e.g. to list touched paths in a summary).
        meter: The `CostMeter` `deps` was built with. For a resumed
            task, `resume_task` replaces `deps.meter` with a freshly
            seeded one, so a caller printing a post-run summary should
            read `deps.meter` at that point, not this field -- this
            field is only guaranteed current for a fresh `run_task`
            call.
        budget_limits: The resolved caps, needed again by a caller that
            wants to reclassify the run's final spend (e.g. to name
            which cap a `BUDGET_HALT` tripped).
        spent_day_usd: The day baseline `deps.spent_day_usd` was built
            from.
        spent_month_usd: The month baseline `deps.spent_month_usd` was
            built from.
    """

    deps: LoopDeps
    undo: UndoManager
    meter: CostMeter
    budget_limits: BudgetLimits
    spent_day_usd: Decimal
    spent_month_usd: Decimal


def build_task_deps(
    *,
    config: KestrelConfig,
    registry: Registry,
    model_id: str,
    kestrel_md: KestrelMd | None,
    repo_root: Path,
    task_id: str,
    limits: LoopLimits = LoopLimits(),
    require_verification: bool = False,
    budget_limits: BudgetLimits | None = None,
    decide_fn: Callable[[ApprovalRequest], ApprovalDecision] = _prompt_stdin,
    observer: LoopObserver = NULL_OBSERVER,
    mode_manager: ModeManager | None = None,
) -> TaskSetup:
    """Build the `LoopDeps` bundle -- and the collaborators a caller
    needs again after the run -- for one task.

    Builds a fresh `ApprovalManager` (pre-approving whatever
    `config.managers.approval.allowlist` names, deciding anything else
    via `decide_fn`), `UndoManager`, and `CostMeter`; a `SessionManager`
    scoped to `task_id` (loading an existing journal when one is already
    there, so a resume picks up where a halted run left off rather than
    starting empty); and a `BudgetManager` from `budget_limits`, which
    resolves to `config.managers.budget`'s own defaults (uncapped) when
    left `None`. `spent_day_usd`/`spent_month_usd` are computed once via
    `aggregate_historical_spend` over every *other* task's own journaled
    spend, always excluding `task_id` itself.

    `decide_fn` defaults to the real stdin prompt
    (`kestrel.managers.approval._prompt_stdin`) -- every existing caller
    that never overrides it sees identical behavior to before this
    function existed. `observer` defaults to `NULL_OBSERVER`, an
    identical no-op contract.

    `mode_manager`, when given, drives `effort`/`available_tools` from
    its own `mode`/`effort()` and forces `require_verification=False` in
    PLAN mode regardless of what the caller passed -- a PLAN-mode task
    is never offered `verify`, so requiring it would nudge the task
    indefinitely. `None` (the default) preserves every existing caller's
    exact behavior: `effort='high'`, `available_tools=None`,
    `require_verification` exactly as passed.

    When `config.managers.self_critique.enabled` (the default), also
    resolves the `"critique"` task class via
    `kestrel.router.policy.resolve_model_id` against `config.router.policy`
    and wires `LoopDeps.self_critique_fn` to
    `kestrel.agent.critique.make_self_critique_fn`'s real, routed check;
    disabling the flag leaves it at `agent.loop`'s own always-approve
    default instead.
    """
    if budget_limits is not None:
        resolved_budget_limits = budget_limits
    else:
        budget_config = config.managers.budget
        resolved_budget_limits = BudgetLimits(
            session_usd=budget_config.session_usd,
            day_usd=budget_config.day_usd,
            month_usd=budget_config.month_usd,
            soft_threshold=budget_config.soft_threshold,
        )
    undo = UndoManager(repo_root=repo_root)
    session = SessionManager(repo_root=repo_root, task_id=task_id)
    now = time.time()
    spent_day_usd = aggregate_historical_spend(
        repo_root, now=now, window_s=_DAY_WINDOW_S, exclude_task_id=task_id
    )
    spent_month_usd = aggregate_historical_spend(
        repo_root, now=now, window_s=_MONTH_WINDOW_S, exclude_task_id=task_id
    )
    meter = CostMeter()
    client = LiteLLMClient(registry)
    if config.managers.self_critique.enabled:
        critique_model_id = resolve_model_id(
            "critique",
            registry=registry,
            policy=config.router.policy.as_mapping(),
            fallback_model_id=model_id,
        )
        self_critique_fn = make_self_critique_fn(
            client=client, model_id=critique_model_id
        )
    else:
        self_critique_fn = _default_self_critique
    effort: Effort = mode_manager.effort() if mode_manager is not None else "high"
    available_tools = (
        frozenset({"read_file", "search"})
        if mode_manager is not None and mode_manager.mode == "plan"
        else None
    )
    effective_require_verification = (
        False
        if mode_manager is not None and mode_manager.mode == "plan"
        else require_verification
    )
    deps = LoopDeps(
        client=client,
        registry=registry,
        model_id=model_id,
        repo_root=repo_root,
        approval=ApprovalManager(
            allowlist=frozenset(config.managers.approval.allowlist),
            decide_fn=decide_fn,
        ),
        undo=undo,
        meter=meter,
        limits=limits,
        self_critique_fn=self_critique_fn,
        effort=effort,
        available_tools=available_tools,
        require_verification=effective_require_verification,
        kestrel_md=kestrel_md,
        session=session,
        budget=BudgetManager(limits=resolved_budget_limits),
        spent_day_usd=spent_day_usd,
        spent_month_usd=spent_month_usd,
        observer=observer,
    )
    return TaskSetup(
        deps=deps,
        undo=undo,
        meter=meter,
        budget_limits=resolved_budget_limits,
        spent_day_usd=spent_day_usd,
        spent_month_usd=spent_month_usd,
    )
