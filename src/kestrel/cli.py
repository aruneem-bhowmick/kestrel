"""Command-line entry point for the ``kestrel`` console script."""

from __future__ import annotations

import asyncio
import shlex
import sys
import time
import uuid
from argparse import SUPPRESS, ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from kestrel import __version__
from kestrel.agent.loop import (
    LoopDeps,
    LoopLimits,
    LoopResult,
    TerminationReason,
    resume_task,
    run_task,
)
from kestrel.config import ConfigError, KestrelConfig, load_config
from kestrel.cost.meter import CostMeter
from kestrel.doctor import all_checks_passed, render_report, run_doctor
from kestrel.kestrel_md import KestrelMd, KestrelMdError, load_kestrel_md
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.budget import BudgetLimits, BudgetManager
from kestrel.managers.session import SessionManager, aggregate_historical_spend
from kestrel.managers.undo import UndoConflictError, UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry
from kestrel.registry.model import ModelEntry, Registry, RegistryError
from kestrel.repl import run_repl

_DEFAULT_LIMITS = LoopLimits()

# Time windows for `aggregate_historical_spend`'s day/month baselines. The
# month window is a fixed 30-day approximation rather than a real calendar
# month (leap years, 28-31 day months) -- close enough for a budget baseline
# that only needs to roughly bound "this month's spend so far," not
# reproduce a billing statement.
_DAY_WINDOW_S = 24.0 * 60.0 * 60.0
_MONTH_WINDOW_S = 30.0 * _DAY_WINDOW_S


def _build_parser() -> ArgumentParser:
    """Build the top-level argument parser.

    The parser accepts the full flag and subcommand surface up front, even
    though most paths do not yet have real behavior wired to them. Building
    the complete surface now means later functionality slots into an
    existing parser instead of requiring one written from scratch.
    """
    parser = ArgumentParser(
        prog="kestrel",
        description="Kestrel: a terminal coding assistant.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed kestrel version and exit.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a kestrel.toml configuration file.",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default=None,
        help="Model registry id to use as the active model.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Run environment and configuration diagnostics."
    )
    doctor_parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Also probe the default model's endpoint with a real, "
            "budget-capped completion."
        ),
    )
    doctor_parser.add_argument(
        "--config",
        metavar="PATH",
        default=SUPPRESS,
        help="Path to a kestrel.toml configuration file.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one task through the tool-calling agent loop, non-interactively.",
    )
    task_group = run_parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "task",
        metavar="TASK",
        nargs="?",
        help="Natural-language description of the task.",
    )
    task_group.add_argument(
        "--resume",
        metavar="TASK_ID",
        default=None,
        help="Resume a previously halted task by the id `kestrel run` printed.",
    )
    run_parser.add_argument(
        "--repo",
        metavar="PATH",
        required=True,
        help="Repository root the task's tools read, edit, and execute against.",
    )
    run_parser.add_argument(
        "--config",
        metavar="PATH",
        default=SUPPRESS,
        help="Path to a kestrel.toml configuration file.",
    )
    run_parser.add_argument(
        "--model",
        metavar="ID",
        default=SUPPRESS,
        help="Model registry id to use as the active model.",
    )
    run_parser.add_argument(
        "--max-turns",
        type=int,
        default=_DEFAULT_LIMITS.max_turns,
        help="Most model calls the task may make before it is stopped.",
    )
    run_parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=_DEFAULT_LIMITS.max_total_tokens,
        help="Most cumulative input-plus-output tokens the task may spend.",
    )
    run_parser.add_argument(
        "--max-wall-clock-s",
        type=float,
        default=_DEFAULT_LIMITS.max_wall_clock_s,
        help="Longest the task may run, in seconds, before it is stopped.",
    )
    run_parser.add_argument(
        "--require-verification",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Withhold task completion from a no-tool-calls turn until the "
            "most recent `verify` call passed (default: enabled)."
        ),
    )
    run_parser.add_argument(
        "--session-budget-usd",
        metavar="USD",
        type=float,
        default=None,
        help=(
            "Cap on this run's own spend. Defaults to kestrel.toml's "
            "[managers.budget].session_usd, itself uncapped by default."
        ),
    )
    run_parser.add_argument(
        "--day-budget-usd",
        metavar="USD",
        type=float,
        default=None,
        help=(
            "Cap on spend across the current day, this run's own spend "
            "included. Defaults to kestrel.toml's "
            "[managers.budget].day_usd, itself uncapped by default."
        ),
    )
    run_parser.add_argument(
        "--month-budget-usd",
        metavar="USD",
        type=float,
        default=None,
        help=(
            "Cap on spend across the current month, this run's own spend "
            "included. Defaults to kestrel.toml's "
            "[managers.budget].month_usd, itself uncapped by default."
        ),
    )
    run_parser.add_argument(
        "--budget-soft-threshold",
        metavar="FRACTION",
        type=float,
        default=None,
        help=(
            "Fraction of a budget cap counted as the soft (warn/degrade) "
            "boundary. Defaults to kestrel.toml's "
            "[managers.budget].soft_threshold, itself 0.8 by default."
        ),
    )

    undo_parser = subparsers.add_parser(
        "undo", help="Revert every file mutation recorded under one task id."
    )
    undo_parser.add_argument(
        "--task-id",
        metavar="ID",
        required=True,
        help="Task id to revert, as printed by `kestrel run`.",
    )
    undo_parser.add_argument(
        "--repo",
        metavar="PATH",
        required=True,
        help="Repository whose undo journal to revert against.",
    )

    return parser


def _resolve_startup(
    args: Namespace,
) -> tuple[KestrelConfig, Registry, str, KestrelMd | None]:
    """Resolve the config, registry, starting model id, and target repo's
    project memory shared by every entry point that talks to a live model
    -- the REPL (no subcommand) and `run` alike.

    Precedence and validation are exactly `load_config`/`load_registry`'s
    own: `args.config` (falling back through `$KESTREL_CONFIG`, then
    `./kestrel.toml`, then the user config directory, then built-in
    defaults) selects the configuration file, its `paths.models_file`
    selects the registry, and `args.model` (falling back to the config's
    own `general.default_model`) selects the starting model id, which
    must already exist in the resolved registry. `KESTREL.md` is loaded
    from `args.repo` for the `run` path, or the current working directory
    for the REPL path -- the REPL has no `--repo` concept of its own, so
    its cwd stands in as the implicit target repo.

    Raises:
        ConfigError: propagated from `load_config` unchanged.
        RegistryError: propagated from `load_registry` unchanged --
            `UnknownModelError` (a `RegistryError` subclass) included,
            raised by the final registry lookup below.
        KestrelMdError: propagated from `load_kestrel_md` unchanged -- a
            KESTREL.md exists but is not valid UTF-8 or its
            `kestrel-verify` block fails to parse.
    """
    explicit_config = Path(args.config) if args.config else None
    config, _config_source = load_config(explicit_config)
    registry = load_registry(config.paths.models_file)
    model_id = args.model or config.general.default_model
    registry.get(model_id)
    repo_root = Path(args.repo) if args.command == "run" else Path.cwd()
    kestrel_md = load_kestrel_md(repo_root)
    return config, registry, model_id, kestrel_md


def _resolve_budget_limits(args: Namespace, config: KestrelConfig) -> BudgetLimits:
    """Build the `BudgetLimits` one `run`/`--resume` invocation checks
    spend against.

    Each of `--session-budget-usd`/`--day-budget-usd`/`--month-budget-usd`/
    `--budget-soft-threshold` overrides its own field when given; any left
    at their argparse default of `None` fall back to
    `config.managers.budget`'s own same-named field, itself uncapped
    (`None`) by default for the three USD caps and `0.8` for the soft
    threshold. A flag's `float` value is routed through `str` before
    `Decimal` so the exact decimal digits a caller typed (e.g. `5.00`) are
    preserved rather than picking up binary-float rounding artifacts.
    """
    budget_config = config.managers.budget
    return BudgetLimits(
        session_usd=(
            Decimal(str(args.session_budget_usd))
            if args.session_budget_usd is not None
            else budget_config.session_usd
        ),
        day_usd=(
            Decimal(str(args.day_budget_usd))
            if args.day_budget_usd is not None
            else budget_config.day_usd
        ),
        month_usd=(
            Decimal(str(args.month_budget_usd))
            if args.month_budget_usd is not None
            else budget_config.month_usd
        ),
        soft_threshold=(
            Decimal(str(args.budget_soft_threshold))
            if args.budget_soft_threshold is not None
            else budget_config.soft_threshold
        ),
    )


def _print_budget_halt(
    task_id: str,
    *,
    budget_limits: BudgetLimits,
    spent_session_usd: Decimal,
    spent_day_usd: Decimal,
    spent_month_usd: Decimal,
    repo_root: Path,
) -> None:
    """Print the abbreviated summary a `BUDGET_HALT` termination gets in
    place of the ordinary one: the task id and reason line (unchanged,
    so a script parsing either summary shape can rely on both always
    being present), followed by a dedicated message naming which cap
    (session, day, or month) actually tripped and the exact
    `kestrel run --resume` invocation that picks the task back up once an
    operator raises it.

    Re-runs the same `BudgetManager.check` classification the loop's own
    last check already made, against the same final spend figures --
    a pure classifier reproduces an identical verdict from identical
    inputs, so this never needs the loop to hand back its own internal
    `BudgetEvent` to name the tripped cap here.
    """
    event = BudgetManager(limits=budget_limits).check(
        spent_session=spent_session_usd,
        spent_day=spent_day_usd + spent_session_usd,
        spent_month=spent_month_usd + spent_session_usd,
    )
    print(f"task_id: {task_id}")
    print(f"reason: {TerminationReason.BUDGET_HALT.name}")
    print(
        f"budget halt: {event.tripped_cap} cap reached; resume with: "
        f"kestrel run --resume {shlex.quote(task_id)} --repo {shlex.quote(str(repo_root))}"
    )


def _print_task_summary(
    task_id: str,
    result: LoopResult,
    undo: UndoManager,
    meter: CostMeter,
    active_entry: ModelEntry,
) -> None:
    """Print the terse summary `run` and `--resume` alike end on once a
    task reaches any termination reason other than `BUDGET_HALT` (that
    reason's own dedicated message is `_print_budget_halt`'s job): the
    task id (needed to `kestrel undo` this run later), the termination
    reason (by its enum member name, e.g. `TASK_COMPLETE`), the turn
    count, the total priced cost, a cache-hit line once at least one
    turn has recorded real usage, and every distinct path the task's own
    `UndoManager` journal recorded a mutation for.
    """
    print(f"task_id: {task_id}")
    print(f"reason: {result.reason.name}")
    print(f"turns: {result.turns_used}")
    print(f"total_usd: ${result.total_usd:.4f}")

    ratio = meter.cache_hit_ratio()
    if ratio is not None:
        alert = meter.cache_alert(active_entry)
        alert_suffix = f" {alert}" if alert is not None else ""
        print(f"cache_hit: {ratio * 100:.0f}%{alert_suffix}")

    touched_paths = sorted(
        {entry.path for entry in undo.entries if entry.task_id == task_id}
    )
    if touched_paths:
        print("files changed:")
        for path in touched_paths:
            print(f"  {path}")


@dataclass(frozen=True, slots=True)
class _RunSetup:
    """The collaborator bundle `_run_task_command` and
    `_resume_task_command` both build, via `_build_run_deps`, before
    driving their own task.

    Attributes:
        deps: The `LoopDeps` bundle to drive the task with.
        undo: The task's own `UndoManager`, read again after the run to
            list touched paths in the summary.
        meter: The `CostMeter` `deps` was built with. For a resumed
            task, `resume_task` replaces `deps.meter` with a freshly
            seeded one, so a caller printing a post-run summary should
            read `deps.meter` at that point, not this field -- this
            field is only guaranteed current for a fresh `run`.
        budget_limits: The resolved caps, needed again by
            `_print_budget_halt` to reclassify the run's final spend.
        spent_day_usd: The day baseline `_print_budget_halt` also needs.
        spent_month_usd: The month baseline `_print_budget_halt` also
            needs.
    """

    deps: LoopDeps
    undo: UndoManager
    meter: CostMeter
    budget_limits: BudgetLimits
    spent_day_usd: Decimal
    spent_month_usd: Decimal


def _build_run_deps(
    *,
    args: Namespace,
    config: KestrelConfig,
    registry: Registry,
    model_id: str,
    kestrel_md: KestrelMd | None,
    repo_root: Path,
    task_id: str,
) -> _RunSetup:
    """Build the `LoopDeps` bundle -- and the collaborators a caller
    needs again after the run -- shared by `_run_task_command` and
    `_resume_task_command`.

    Builds a fresh `ApprovalManager` (pre-approving whatever
    `config.managers.approval.allowlist` names), `UndoManager`, and
    `CostMeter`; a `SessionManager` scoped to `task_id` (loading an
    existing journal when one is already there, so a resume picks up
    where a halted run left off rather than starting empty); and
    `BudgetManager` from `_resolve_budget_limits`. `spent_day_usd`/
    `spent_month_usd` are computed once via `aggregate_historical_spend`
    over every *other* task's own journaled spend, always excluding
    `task_id` itself -- for a fresh run that id has no history yet to
    exclude; for a resume, excluding it is what keeps that task's own
    prior spend (already re-added once its own `CostMeter` is resumed)
    from being double-counted.
    """
    undo = UndoManager(repo_root=repo_root)
    session = SessionManager(repo_root=repo_root, task_id=task_id)
    budget_limits = _resolve_budget_limits(args, config)
    now = time.time()
    spent_day_usd = aggregate_historical_spend(
        repo_root, now=now, window_s=_DAY_WINDOW_S, exclude_task_id=task_id
    )
    spent_month_usd = aggregate_historical_spend(
        repo_root, now=now, window_s=_MONTH_WINDOW_S, exclude_task_id=task_id
    )
    meter = CostMeter()
    deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id=model_id,
        repo_root=repo_root,
        approval=ApprovalManager(
            allowlist=frozenset(config.managers.approval.allowlist)
        ),
        undo=undo,
        meter=meter,
        limits=LoopLimits(
            max_turns=args.max_turns,
            max_total_tokens=args.max_total_tokens,
            max_wall_clock_s=args.max_wall_clock_s,
        ),
        require_verification=args.require_verification,
        kestrel_md=kestrel_md,
        session=session,
        budget=BudgetManager(limits=budget_limits),
        spent_day_usd=spent_day_usd,
        spent_month_usd=spent_month_usd,
    )
    return _RunSetup(
        deps=deps,
        undo=undo,
        meter=meter,
        budget_limits=budget_limits,
        spent_day_usd=spent_day_usd,
        spent_month_usd=spent_month_usd,
    )


def _run_task_command(
    args: Namespace,
    config: KestrelConfig,
    registry: Registry,
    model_id: str,
    kestrel_md: KestrelMd | None,
) -> int:
    """Run `args.task` to completion against `args.repo` and print a terse
    summary.

    Builds its collaborators via `_build_run_deps` -- nothing built
    there is reused across separate `run` invocations. On
    `TerminationReason.BUDGET_HALT`, prints `_print_budget_halt`'s
    dedicated message instead of the ordinary summary; every other
    reason gets `_print_task_summary`. Exits `0` only on
    `TerminationReason.TASK_COMPLETE`.
    """
    repo_root = Path(args.repo)
    task_id = str(uuid.uuid4())
    setup = _build_run_deps(
        args=args,
        config=config,
        registry=registry,
        model_id=model_id,
        kestrel_md=kestrel_md,
        repo_root=repo_root,
        task_id=task_id,
    )

    assert args.task is not None  # enforced by the `task`/`--resume` mutex group
    result = asyncio.run(run_task(args.task, setup.deps, task_id))

    if result.reason is TerminationReason.BUDGET_HALT:
        _print_budget_halt(
            task_id,
            budget_limits=setup.budget_limits,
            spent_session_usd=result.total_usd,
            spent_day_usd=setup.spent_day_usd,
            spent_month_usd=setup.spent_month_usd,
            repo_root=repo_root,
        )
        return 1

    _print_task_summary(
        task_id, result, setup.undo, setup.meter, registry.get(model_id)
    )
    return 0 if result.reason is TerminationReason.TASK_COMPLETE else 1


def _resume_task_command(
    args: Namespace,
    config: KestrelConfig,
    registry: Registry,
    model_id: str,
    kestrel_md: KestrelMd | None,
) -> int:
    """Continue `args.resume` from its own journal under `args.repo` and
    print the identical summary shape `_run_task_command` does.

    Builds its collaborators via `_build_run_deps`, exactly like
    `_run_task_command` does, except `task_id` is `args.resume` rather
    than a freshly generated id -- `_build_run_deps`'s own
    `SessionManager`/`UndoManager` construction already loads that id's
    existing journal when one is there, so they pick up right where the
    halted run left off rather than starting empty. Neither `CostMeter`
    nor `verification_reports` need seeding here -- `resume_task` itself
    reconstructs both from the loaded session state (see
    `kestrel.agent.loop.resume_task`), overwriting whatever placeholder
    `LoopDeps.meter` `_build_run_deps` built.

    Raises nothing on a missing journal -- `FileNotFoundError` is caught
    and reported the same way a `ConfigError` is, exiting `1` instead of
    a raw traceback.
    """
    repo_root = Path(args.repo)
    task_id = args.resume
    assert task_id is not None  # enforced by the `task`/`--resume` mutex group
    setup = _build_run_deps(
        args=args,
        config=config,
        registry=registry,
        model_id=model_id,
        kestrel_md=kestrel_md,
        repo_root=repo_root,
        task_id=task_id,
    )

    try:
        result = asyncio.run(resume_task(task_id, setup.deps))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if result.reason is TerminationReason.BUDGET_HALT:
        _print_budget_halt(
            task_id,
            budget_limits=setup.budget_limits,
            spent_session_usd=result.total_usd,
            spent_day_usd=setup.spent_day_usd,
            spent_month_usd=setup.spent_month_usd,
            repo_root=repo_root,
        )
        return 1

    _print_task_summary(
        task_id, result, setup.undo, setup.deps.meter, registry.get(model_id)
    )
    return 0 if result.reason is TerminationReason.TASK_COMPLETE else 1


def _run_undo_command(args: Namespace) -> int:
    """Revert every mutation `args.task_id` recorded in `args.repo`'s undo
    journal and print each reverted path.

    Returns 1 (rather than raising) when `UndoConflictError` stops a
    partial revert partway through -- the manager's own error message
    already names the conflicting path.
    """
    undo = UndoManager(repo_root=Path(args.repo))
    try:
        reverted = undo.revert_task(args.task_id)
    except UndoConflictError as exc:
        if exc.reverted:
            print(
                f"reverted {len(exc.reverted)} mutation(s) for task '{args.task_id}':"
            )
            for entry in exc.reverted:
                print(f"  {entry.path}")
        print(exc, file=sys.stderr)
        return 1

    if not reverted:
        print(f"no journal entries found for task '{args.task_id}'")
        return 0

    print(f"reverted {len(reverted)} mutation(s) for task '{args.task_id}':")
    for entry in reverted:
        print(f"  {entry.path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Subcommands: (none)=repl, doctor [--live], run
    (TASK | --resume TASK_ID) --repo PATH [...], undo --task-id ID --repo
    PATH. Flags: --version, --config PATH, --model ID.

    ``doctor`` prints one aligned line per flight check and exits 0
    unless any check FAILed. ``run`` resolves config/registry/starting
    model/KESTREL.md exactly like the REPL path, drives `run_task` (or,
    with `--resume`, `resume_task`) to completion, prints its summary or
    (on a budget halt) a dedicated resume hint, and exits 0 on
    `TerminationReason.TASK_COMPLETE` or 1 on any other reason. ``undo``
    reverts a prior run's file mutations and exits 1 only if a conflict
    stops it partway through. Every other path either prints the version
    or starts the REPL against the resolved configuration, registry,
    starting model, and the working directory's own KESTREL.md.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "doctor":
        config_path = Path(args.config) if args.config else None
        results = run_doctor(config_path, live=args.live)
        print(render_report(results), end="")
        return 0 if all_checks_passed(results) else 1

    if args.command == "undo":
        return _run_undo_command(args)

    try:
        config, registry, model_id, kestrel_md = _resolve_startup(args)
    except (ConfigError, RegistryError, KestrelMdError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.command == "run":
        if args.resume is not None:
            return _resume_task_command(args, config, registry, model_id, kestrel_md)
        return _run_task_command(args, config, registry, model_id, kestrel_md)

    client = LiteLLMClient(registry)
    return run_repl(config, registry, client, model_id, kestrel_md=kestrel_md)


if __name__ == "__main__":
    sys.exit(main())
