"""Unit tests for `kestrel run`'s verification/resume/budget CLI wiring:
argparse parsing (the `task`/`--resume` mutual exclusivity, the
`--require-verification` default, the budget flags' fallback to
`kestrel.toml`), `_resolve_startup`'s now KESTREL.md-aware return value,
and the two summary-printing helpers (`_print_task_summary`,
`_print_budget_halt`) -- all hermetic and network-free, since a real
`run_task`/`resume_task` call is out of scope here (see
`tests/acceptance/test_p033_dod_phase_2.py` and
`tests/e2e/test_p033_dod_live.py` for that), mirroring
`tests/unit/test_p023_cli.py`'s own split between wiring coverage and
end-to-end behavior.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.cli import (
    _build_parser,
    _print_budget_halt,
    _print_task_summary,
    _resolve_budget_limits,
    _resolve_startup,
    main,
)
from kestrel.config import BudgetConfig, KestrelConfig, ManagersConfig
from kestrel.cost.meter import CostMeter
from kestrel.managers.budget import BudgetLimits
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p033, pytest.mark.unit, pytest.mark.sanity]

_ONE_MODEL_TOML = """\
[[models]]
id = "glm-5.2"
backend = "openrouter"
provider_model = "z-ai/glm-5.2"
api_key_env = "OPENROUTER_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true
"""


def _entry(*, supports_cache: bool = True) -> ModelEntry:
    """A single registry entry with round rates, for cache-alert math a
    test can hand-verify."""
    return ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("1.00"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=True,
        supports_cache=supports_cache,
    )


def _result(reason: TerminationReason, *, turns_used: int = 1) -> LoopResult:
    """A minimal `LoopResult` for exercising the summary printers."""
    return LoopResult(
        reason=reason, turns_used=turns_used, total_usd=Decimal("0.0071"), history=()
    )


# --- argparse wiring -------------------------------------------------------


def test_run_require_verification_defaults_to_true() -> None:
    """Given `run` with neither `--require-verification` nor
    `--no-require-verification`, when parsed, then the flag defaults on
    (D2's own default)."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo"])
    assert args.require_verification is True


def test_run_no_require_verification_flag_disables_it() -> None:
    """Given `--no-require-verification`, when parsed, then the flag is
    `False`."""
    parser = _build_parser()
    args = parser.parse_args(
        ["run", "task", "--repo", "/tmp/repo", "--no-require-verification"]
    )
    assert args.require_verification is False


def test_run_task_and_resume_are_mutually_exclusive() -> None:
    """Given both a positional task and `--resume`, when parsed, then
    argparse rejects the combination before any argument-dependent code
    runs."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "task", "--resume", "abc-123", "--repo", "/tmp/repo"])


def test_run_requires_either_task_or_resume() -> None:
    """Given neither a positional task nor `--resume`, when parsed, then
    argparse rejects it -- the mutually exclusive group is required."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--repo", "/tmp/repo"])


def test_run_resume_flag_parses_without_a_positional_task() -> None:
    """Given `--resume TASK_ID` with no positional task, when parsed,
    then `args.resume` carries the id and `args.task` is `None`."""
    parser = _build_parser()
    args = parser.parse_args(["run", "--resume", "abc-123", "--repo", "/tmp/repo"])
    assert args.resume == "abc-123"
    assert args.task is None


def test_run_budget_flags_default_to_none() -> None:
    """Given `run` with none of the budget-override flags, when parsed,
    then every one defaults to `None` -- falling back to `kestrel.toml`
    is `_resolve_budget_limits`'s job, not argparse's."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo"])
    assert args.session_budget_usd is None
    assert args.day_budget_usd is None
    assert args.month_budget_usd is None
    assert args.budget_soft_threshold is None


def test_run_budget_flags_accept_overrides() -> None:
    """Given every budget flag set explicitly, when parsed, then each
    lands as its own `float`."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "task",
            "--repo",
            "/tmp/repo",
            "--session-budget-usd",
            "5",
            "--day-budget-usd",
            "20",
            "--month-budget-usd",
            "100",
            "--budget-soft-threshold",
            "0.9",
        ]
    )
    assert args.session_budget_usd == 5.0
    assert args.day_budget_usd == 20.0
    assert args.month_budget_usd == 100.0
    assert args.budget_soft_threshold == 0.9


# --- _resolve_budget_limits --------------------------------------------


def _args(
    *,
    session_budget_usd: float | None = None,
    day_budget_usd: float | None = None,
    month_budget_usd: float | None = None,
    budget_soft_threshold: float | None = None,
) -> Namespace:
    """A bare `Namespace` carrying only the four fields
    `_resolve_budget_limits` reads -- narrower than a real parsed `run`
    invocation, since that is all this helper needs."""
    return Namespace(
        session_budget_usd=session_budget_usd,
        day_budget_usd=day_budget_usd,
        month_budget_usd=month_budget_usd,
        budget_soft_threshold=budget_soft_threshold,
    )


def test_resolve_budget_limits_falls_back_to_config_for_every_field() -> None:
    """Given every budget flag left at `None`, when resolved, then every
    `BudgetLimits` field is read from `config.managers.budget` instead."""
    config = KestrelConfig(
        managers=ManagersConfig(
            budget=BudgetConfig(
                session_usd=Decimal("5.00"),
                day_usd=Decimal("20.00"),
                month_usd=Decimal("100.00"),
                soft_threshold=Decimal("0.75"),
            )
        )
    )

    limits = _resolve_budget_limits(_args(), config)

    assert limits == BudgetLimits(
        session_usd=Decimal("5.00"),
        day_usd=Decimal("20.00"),
        month_usd=Decimal("100.00"),
        soft_threshold=Decimal("0.75"),
    )


def test_resolve_budget_limits_flags_override_config_independently() -> None:
    """Given only `--session-budget-usd` set, when resolved, then it
    overrides the config's own session cap while every other field still
    falls back to the config."""
    config = KestrelConfig(
        managers=ManagersConfig(
            budget=BudgetConfig(day_usd=Decimal("20.00"), month_usd=Decimal("100.00"))
        )
    )

    limits = _resolve_budget_limits(_args(session_budget_usd=5.0), config)

    assert limits.session_usd == Decimal("5.0")
    assert limits.day_usd == Decimal("20.00")
    assert limits.month_usd == Decimal("100.00")
    assert limits.soft_threshold == Decimal("0.8")


def test_resolve_budget_limits_preserves_the_typed_decimal_digits() -> None:
    """Given a soft threshold typed as `0.85`, when resolved, then the
    stored `Decimal` reads exactly `0.85` -- not a binary-float artifact
    like `0.8500000000000000711...`."""
    config = KestrelConfig()

    limits = _resolve_budget_limits(_args(budget_soft_threshold=0.85), config)

    assert limits.soft_threshold == Decimal("0.85")


# --- _resolve_startup's KESTREL.md loading -------------------------------


def test_resolve_startup_loads_kestrel_md_for_the_run_path(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given a repo whose root carries a KESTREL.md, when `_resolve_startup`
    runs against `run`-shaped args, then the fourth return value carries
    that file's own raw text."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "KESTREL.md").write_bytes(b"# project notes\n")
    parser = _build_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "run", "task", "--repo", str(repo_dir)]
    )

    _config, _registry, _model_id, kestrel_md = _resolve_startup(args)

    assert kestrel_md is not None
    assert kestrel_md.raw_text == "# project notes\n"


def test_resolve_startup_returns_none_kestrel_md_when_repo_has_none(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given a repo with no KESTREL.md at all, when `_resolve_startup`
    runs, then the fourth return value is `None` rather than an error --
    a missing KESTREL.md is a normal outcome."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    parser = _build_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "run", "task", "--repo", str(repo_dir)]
    )

    _config, _registry, _model_id, kestrel_md = _resolve_startup(args)

    assert kestrel_md is None


def test_resolve_startup_loads_cwd_kestrel_md_for_the_repl_path(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given no subcommand (the REPL path) and a KESTREL.md in the
    current working directory, when `_resolve_startup` runs, then it is
    loaded from `Path.cwd()` rather than left unloaded -- the REPL has no
    `--repo` flag of its own, so its cwd stands in as the implicit target
    repo."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    (tmp_path / "KESTREL.md").write_bytes(b"# repl-side notes\n")
    parser = _build_parser()
    args = parser.parse_args(["--config", str(config_path)])
    assert args.command is None

    _config, _registry, _model_id, kestrel_md = _resolve_startup(args)

    assert kestrel_md is not None
    assert kestrel_md.raw_text == "# repl-side notes\n"


def test_run_with_a_malformed_kestrel_md_exits_one_not_a_traceback(
    tmp_path: Path,
    write_config: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a KESTREL.md whose `kestrel-verify` block is not valid TOML,
    when `run` executes through `main`, then it exits 1 with a readable
    message instead of a raw `KestrelMdError` traceback."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "KESTREL.md").write_text(
        "```kestrel-verify\nnot valid toml =====\n```\n", encoding="utf-8"
    )

    exit_code = main(
        ["run", "task", "--repo", str(repo_dir), "--config", str(config_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "kestrel-verify" in captured.err


# --- resuming a task with no matching journal ----------------------------


def test_resume_with_no_matching_journal_exits_one_not_a_traceback(
    tmp_path: Path,
    write_config: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given `--resume` names a task id with no journal on disk under
    `--repo`, when `run` executes through `main`, then it exits 1 with a
    readable message instead of a real `resume_task` call ever streaming
    anything -- `load_session`'s own `FileNotFoundError` fires before any
    network-facing collaborator is touched."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    exit_code = main(
        [
            "run",
            "--resume",
            "no-such-task",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "no-such-task" in captured.err


# --- _print_task_summary / _print_budget_halt ----------------------------


def test_print_task_summary_includes_cache_hit_line_once_usage_is_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a meter with at least one recorded turn, when the summary is
    printed, then a `cache_hit:` line reports the measured ratio."""
    meter = CostMeter()
    entry = _entry()
    meter.record(
        UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=60), entry
    )
    undo = UndoManager(repo_root=tmp_path)

    _print_task_summary(
        "t-1", _result(TerminationReason.TASK_COMPLETE), undo, meter, entry
    )

    out = capsys.readouterr().out
    assert "cache_hit: 60%" in out


def test_print_task_summary_omits_cache_hit_line_with_no_recorded_turns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a fresh meter with no recorded turns, when the summary is
    printed, then no `cache_hit:` line appears -- there is no ratio yet
    to report."""
    meter = CostMeter()
    entry = _entry()
    undo = UndoManager(repo_root=tmp_path)

    _print_task_summary(
        "t-1", _result(TerminationReason.TASK_COMPLETE), undo, meter, entry
    )

    out = capsys.readouterr().out
    assert "cache_hit:" not in out


def test_print_task_summary_appends_the_cache_alert_text_below_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a three-plus-turn session whose ratio sits below the 50%
    alert threshold, when the summary is printed, then the `cache_hit:`
    line carries the meter's own alert text appended after the
    percentage."""
    meter = CostMeter()
    entry = _entry()
    for _ in range(3):
        meter.record(
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0), entry
        )
    undo = UndoManager(repo_root=tmp_path)

    _print_task_summary(
        "t-1", _result(TerminationReason.TASK_COMPLETE), undo, meter, entry
    )

    out = capsys.readouterr().out
    assert "cache_hit: 0%" in out
    assert "below the 50% alert threshold" in out


def test_print_budget_halt_names_the_tripped_cap_and_the_resume_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a session cap already reached, when the halt message is
    printed, then it carries the same `task_id:`/`reason:` lines the
    ordinary summary does, names the `session` cap specifically, and
    gives the exact `kestrel run --resume` invocation that picks the
    task back up."""
    repo_root = Path("repo")
    _print_budget_halt(
        "t-1",
        budget_limits=BudgetLimits(session_usd=Decimal("1.00")),
        spent_session_usd=Decimal("1.00"),
        spent_day_usd=Decimal("0"),
        spent_month_usd=Decimal("0"),
        repo_root=repo_root,
    )

    out = capsys.readouterr().out
    assert "task_id: t-1" in out
    assert "reason: BUDGET_HALT" in out
    assert "budget halt: session cap reached" in out
    assert f"kestrel run --resume t-1 --repo {repo_root}" in out
