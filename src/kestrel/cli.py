"""Command-line entry point for the ``kestrel`` console script."""

from __future__ import annotations

import asyncio
import sys
import uuid
from argparse import SUPPRESS, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from kestrel import __version__
from kestrel.agent.loop import LoopDeps, LoopLimits, TerminationReason, run_task
from kestrel.config import ConfigError, KestrelConfig, load_config
from kestrel.cost.meter import CostMeter
from kestrel.doctor import all_checks_passed, render_report, run_doctor
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoConflictError, UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry
from kestrel.registry.model import Registry, RegistryError
from kestrel.repl import run_repl

_DEFAULT_LIMITS = LoopLimits()


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
    run_parser.add_argument(
        "task", metavar="TASK", help="Natural-language description of the task."
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


def _resolve_startup(args: Namespace) -> tuple[KestrelConfig, Registry, str]:
    """Resolve the config, registry, and starting model id shared by every
    entry point that talks to a live model -- the REPL (no subcommand) and
    `run` alike.

    Precedence and validation are exactly `load_config`/`load_registry`'s
    own: `args.config` (falling back through `$KESTREL_CONFIG`, then
    `./kestrel.toml`, then the user config directory, then built-in
    defaults) selects the configuration file, its `paths.models_file`
    selects the registry, and `args.model` (falling back to the config's
    own `general.default_model`) selects the starting model id, which
    must already exist in the resolved registry.

    Raises:
        ConfigError: propagated from `load_config` unchanged.
        RegistryError: propagated from `load_registry` unchanged --
            `UnknownModelError` (a `RegistryError` subclass) included,
            raised by the final registry lookup below.
    """
    explicit_config = Path(args.config) if args.config else None
    config, _config_source = load_config(explicit_config)
    registry = load_registry(config.paths.models_file)
    model_id = args.model or config.general.default_model
    registry.get(model_id)
    return config, registry, model_id


def _run_task_command(
    args: Namespace, config: KestrelConfig, registry: Registry, model_id: str
) -> int:
    """Run `args.task` to completion against `args.repo` and print a terse
    summary.

    Builds a fresh `ApprovalManager` (pre-approving whatever
    `config.managers.approval.allowlist` names), `UndoManager`, and
    `CostMeter` for this run alone -- nothing here is reused across
    separate `run` invocations. The printed summary names the generated
    task id (needed to `kestrel undo` this run later), the termination
    reason (by its enum member name, e.g. `TASK_COMPLETE`), the turn
    count, the total priced cost, and every distinct path the run's own
    `UndoManager` journal recorded a mutation for.
    """
    repo_root = Path(args.repo)
    task_id = str(uuid.uuid4())
    undo = UndoManager(repo_root=repo_root)
    deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id=model_id,
        repo_root=repo_root,
        approval=ApprovalManager(
            allowlist=frozenset(config.managers.approval.allowlist)
        ),
        undo=undo,
        meter=CostMeter(),
        limits=LoopLimits(
            max_turns=args.max_turns,
            max_total_tokens=args.max_total_tokens,
            max_wall_clock_s=args.max_wall_clock_s,
        ),
    )

    result = asyncio.run(run_task(args.task, deps, task_id))

    touched_paths = sorted(
        {entry.path for entry in undo.entries if entry.task_id == task_id}
    )
    print(f"task_id: {task_id}")
    print(f"reason: {result.reason.name}")
    print(f"turns: {result.turns_used}")
    print(f"total_usd: ${result.total_usd:.4f}")
    if touched_paths:
        print("files changed:")
        for path in touched_paths:
            print(f"  {path}")

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
    """Entry point. Subcommands: (none)=repl, doctor [--live], run TASK
    --repo PATH [...], undo --task-id ID --repo PATH. Flags: --version,
    --config PATH, --model ID.

    ``doctor`` prints one aligned line per flight check and exits 0
    unless any check FAILed. ``run`` resolves config/registry/starting
    model exactly like the REPL path, drives `run_task` to completion,
    prints its summary, and exits 0 on `TerminationReason.TASK_COMPLETE`
    or 1 on any other reason. ``undo`` reverts a prior run's file
    mutations and exits 1 only if a conflict stops it partway through.
    Every other path either prints the version or starts the REPL
    against the resolved configuration, registry, and starting model.
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
        config, registry, model_id = _resolve_startup(args)
    except (ConfigError, RegistryError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.command == "run":
        return _run_task_command(args, config, registry, model_id)

    client = LiteLLMClient(registry)
    return run_repl(config, registry, client, model_id)


if __name__ == "__main__":
    sys.exit(main())
