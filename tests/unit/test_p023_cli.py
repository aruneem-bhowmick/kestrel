"""Unit tests for the CLI's `run` and `undo` subcommand wiring: argparse
parsing (required arguments, defaults, both orderings of `--config` and
`--model`), the shared `_resolve_startup` helper the REPL and `run`
paths both call, and `undo`'s own dispatch -- all hermetic and
network-free, since a real `run_task` call is out of scope here (see
`tests/acceptance/test_p023_dod_phase_1.py` and
`tests/e2e/test_p023_dod_live.py` for that).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopLimits
from kestrel.cli import _build_parser, _resolve_startup, main
from kestrel.managers.undo import UndoEntry, UndoManager
from kestrel.registry.model import UnknownModelError

pytestmark = [pytest.mark.p023, pytest.mark.unit, pytest.mark.sanity]

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

_TWO_MODEL_TOML = (
    _ONE_MODEL_TOML
    + """
[[models]]
id = "glm-5.2-zai"
backend = "zai"
provider_model = "glm-5.2"
endpoint = "https://example.invalid/v4"
api_key_env = "ZAI_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true
"""
)


def test_run_requires_a_repo_flag() -> None:
    """Given `run` with no `--repo`, when parsed, then argparse rejects
    it before any argument-dependent code runs."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "do something"])


def test_run_requires_a_task_argument() -> None:
    """Given `run` with `--repo` but no task description, when parsed,
    then argparse rejects it."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--repo", "/tmp/repo"])


def test_undo_requires_both_task_id_and_repo_flags() -> None:
    """Given `undo` missing either `--task-id` or `--repo`, when parsed,
    then argparse rejects both incomplete forms."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["undo", "--repo", "/tmp/repo"])
    with pytest.raises(SystemExit):
        parser.parse_args(["undo", "--task-id", "abc-123"])


def test_run_parses_task_and_repo() -> None:
    """Given a minimal well-formed `run` invocation, when parsed, then
    the task description and repo path land exactly where `main` reads
    them from."""
    parser = _build_parser()
    args = parser.parse_args(["run", "add a function", "--repo", "/tmp/repo"])
    assert args.command == "run"
    assert args.task == "add a function"
    assert args.repo == "/tmp/repo"


def test_run_limit_flags_default_to_loop_limits() -> None:
    """Given `run` with none of the limit-override flags, when parsed,
    then every limit matches `LoopLimits`'s own defaults -- so a `run`
    invocation with no overrides behaves identically to the loop's own
    out-of-the-box caps."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo"])
    defaults = LoopLimits()
    assert args.max_turns == defaults.max_turns
    assert args.max_total_tokens == defaults.max_total_tokens
    assert args.max_wall_clock_s == defaults.max_wall_clock_s


def test_run_limit_flags_accept_overrides() -> None:
    """Given `run` with every limit-override flag set explicitly, when
    parsed, then each one overrides its own default independently."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "task",
            "--repo",
            "/tmp/repo",
            "--max-turns",
            "5",
            "--max-total-tokens",
            "1000",
            "--max-wall-clock-s",
            "30",
        ]
    )
    assert args.max_turns == 5
    assert args.max_total_tokens == 1000
    assert args.max_wall_clock_s == 30.0


def test_run_config_flag_works_before_the_subcommand() -> None:
    """Given `--config` precedes `run` (the top-level flag position),
    when parsed, then it still resolves to the named path."""
    parser = _build_parser()
    args = parser.parse_args(
        ["--config", "/tmp/x.toml", "run", "task", "--repo", "/tmp/repo"]
    )
    assert args.config == "/tmp/x.toml"


def test_run_config_flag_works_after_the_subcommand() -> None:
    """Given `--config` follows `run` (the subcommand-scoped position),
    when parsed, then it resolves identically to the flag preceding it."""
    parser = _build_parser()
    args = parser.parse_args(
        ["run", "task", "--repo", "/tmp/repo", "--config", "/tmp/x.toml"]
    )
    assert args.config == "/tmp/x.toml"


def test_run_model_flag_works_before_and_after_the_subcommand() -> None:
    """Given `--model` in either position around `run`, when parsed,
    then both resolve to the same value."""
    parser = _build_parser()
    before = parser.parse_args(
        ["--model", "glm-5.2", "run", "task", "--repo", "/tmp/repo"]
    )
    after = parser.parse_args(
        ["run", "task", "--repo", "/tmp/repo", "--model", "glm-5.2"]
    )
    assert before.model == "glm-5.2"
    assert after.model == "glm-5.2"


def test_undo_parses_task_id_and_repo() -> None:
    """Given a well-formed `undo` invocation, when parsed, then the task
    id and repo path land exactly where `main` reads them from."""
    parser = _build_parser()
    args = parser.parse_args(["undo", "--task-id", "abc-123", "--repo", "/tmp/repo"])
    assert args.command == "undo"
    assert args.task_id == "abc-123"
    assert args.repo == "/tmp/repo"


def test_resolve_startup_returns_the_configured_default_model(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given a config naming a default model that exists in the
    registry, when `_resolve_startup` runs against `run`-shaped args with
    no `--model` override, then it returns that config, a registry
    containing the entry, and that entry's id."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    parser = _build_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "run", "task", "--repo", str(tmp_path)]
    )

    config, registry, model_id, kestrel_md = _resolve_startup(args)

    assert model_id == "glm-5.2"
    assert config.general.default_model == "glm-5.2"
    assert registry.get("glm-5.2").backend == "openrouter"
    assert kestrel_md is None


def test_resolve_startup_model_flag_overrides_the_configured_default(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given `--model` names a different, still-valid registry entry,
    when `_resolve_startup` runs, then the explicit override wins over
    the config's own default."""
    config_path = write_config(tmp_path, _TWO_MODEL_TOML, default_model="glm-5.2")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--model",
            "glm-5.2-zai",
            "run",
            "task",
            "--repo",
            str(tmp_path),
        ]
    )

    _config, registry, model_id, _kestrel_md = _resolve_startup(args)

    assert model_id == "glm-5.2-zai"
    assert registry.get("glm-5.2-zai").backend == "zai"


def test_resolve_startup_raises_for_an_unknown_model(
    tmp_path: Path, write_config: Callable[..., Path]
) -> None:
    """Given `--model` names an id absent from the registry, when
    `_resolve_startup` runs, then `UnknownModelError` propagates rather
    than resolving to a silently wrong entry."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--model",
            "does-not-exist",
            "run",
            "task",
            "--repo",
            str(tmp_path),
        ]
    )

    with pytest.raises(UnknownModelError):
        _resolve_startup(args)


def test_run_missing_config_file_exits_one_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given `--config` names a file that does not exist, when `run`
    executes through `main`, then it exits 1 with a readable message
    instead of a real `run_task` call ever being attempted."""
    missing = tmp_path / "missing.toml"

    exit_code = main(["run", "task", "--repo", str(tmp_path), "--config", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not found" in captured.err


def test_run_unknown_model_exits_one_not_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_config: Callable[..., Path],
) -> None:
    """Given `--model` names an id the registry does not have, when
    `run` executes through `main`, then it exits 1 naming the unknown id
    instead of a real `run_task` call ever being attempted."""
    config_path = write_config(tmp_path, _ONE_MODEL_TOML, default_model="glm-5.2")

    exit_code = main(
        [
            "run",
            "task",
            "--repo",
            str(tmp_path),
            "--config",
            str(config_path),
            "--model",
            "does-not-exist",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unknown model id" in captured.err


def test_undo_with_no_matching_journal_entries_reports_none_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a repo whose undo journal has no entries for the given task
    id, when `undo` executes, then it exits 0 and reports that plainly
    rather than treating an empty result as an error."""
    exit_code = main(["undo", "--task-id", "no-such-task", "--repo", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "no journal entries found for task 'no-such-task'" in captured.out


def test_undo_reverts_a_real_journal_entry_and_reports_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a repo whose undo journal has one real mutation recorded
    for a task id, when `undo` executes, then the file is restored to
    its pre-mutation content and the reverted path is printed."""
    target = tmp_path / "f.py"
    target.write_text("after content", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    undo.record(
        UndoEntry(
            turn_id=1,
            task_id="t-1",
            path="f.py",
            before="before content",
            after="after content",
        )
    )

    exit_code = main(["undo", "--task-id", "t-1", "--repo", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert target.read_text(encoding="utf-8") == "before content"
    assert "reverted 1 mutation(s) for task 't-1':" in captured.out
    assert "f.py" in captured.out


def test_undo_conflict_exits_one_naming_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a journal entry whose recorded `after` content no longer
    matches what is actually on disk, when `undo` executes, then it
    exits 1 with the manager's own conflict message rather than
    silently overwriting the out-of-band change."""
    target = tmp_path / "f.py"
    target.write_text("something else entirely", encoding="utf-8")
    undo = UndoManager(repo_root=tmp_path)
    undo.record(
        UndoEntry(turn_id=1, task_id="t-1", path="f.py", before="before", after="after")
    )

    exit_code = main(["undo", "--task-id", "t-1", "--repo", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "current content does not match" in captured.err
