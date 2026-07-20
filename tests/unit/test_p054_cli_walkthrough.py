"""Unit tests for wiring the `Walkthrough` artifact into `kestrel run`'s
own summary: `_print_task_summary`'s new `walkthrough_path` line,
`_build_and_persist_walkthrough`'s success and `WalkthroughError` paths,
and both `_run_task_command`/`_resume_task_command` driving a real,
scripted task through to a persisted walkthrough naming an embedded
verification report.

The end-to-end scenarios below reuse `tests/system/test_p049_plan_task_
scripted.py`'s own `main()`-driven, mock-server-plus-fixture-repo
pattern. `verify`'s own tool call is stubbed via a monkeypatched
`kestrel.agent.loop.dispatch` -- exactly like `tests/unit/test_p026_
verification_gate.py` already does -- so this suite never depends on
the real `bwrap` sandbox `kestrel.tools.verify.run_verification` would
otherwise need.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
from kestrel.agent.loop import LoopResult, TerminationReason
from kestrel.agent.walkthrough import Walkthrough, render_walkthrough_markdown
from kestrel.cli import _build_and_persist_walkthrough, _print_task_summary, main
from kestrel.cost.meter import CostMeter
from kestrel.managers.undo import UndoManager
from kestrel.provider.events import ToolCallEvent
from kestrel.registry.model import ModelEntry
from kestrel.tools.registry import ToolResult
from kestrel.tools.registry import dispatch as _real_dispatch
from kestrel.tools.verify import VerificationCommandResult, VerificationReport

pytestmark = [pytest.mark.p054, pytest.mark.unit]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_TOOLCALL_VERIFY = _CASSETTES / "toolcall_verify.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_TASK_ID_RE = re.compile(r"^task_id: (?P<task_id>\S+)$", re.MULTILINE)
_TURNS_RE = re.compile(r"^turns: (?P<turns>\d+)$", re.MULTILINE)
_TOTAL_USD_RE = re.compile(r"^total_usd: \$(?P<total_usd>[\d.]+)$", re.MULTILINE)
_WALKTHROUGH_LINE_RE = re.compile(r"^walkthrough: (?P<path>.+)$", re.MULTILINE)

_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "implement greet in greet.py, then verify"


def _entry() -> ModelEntry:
    """A single, cheap OpenRouter-routed registry entry matching every
    cassette's own `model` field."""
    return ModelEntry(
        id="glm-5.2",
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


def _result(reason: TerminationReason, *, turns_used: int = 1) -> LoopResult:
    """A minimal `LoopResult` for exercising the summary printers."""
    return LoopResult(
        reason=reason, turns_used=turns_used, total_usd=Decimal("0.0071"), history=()
    )


def _passing_report() -> VerificationReport:
    """A single-command, trivially-passing `VerificationReport` -- its
    exact `task_id`/`turn_id` are irrelevant, since neither is rendered
    by `render_verification_markdown`."""
    return VerificationReport(
        task_id="stub-task",
        turn_id=1,
        commands=(
            VerificationCommandResult(
                name="test",
                command="pytest -q",
                exit_code=0,
                timed_out=False,
                stdout="3 passed in 0.11s",
                stderr="",
            ),
        ),
        passed=True,
    )


def _stub_verify_dispatch(
    monkeypatch: pytest.MonkeyPatch, report: VerificationReport
) -> None:
    """Monkeypatch `kestrel.agent.loop.dispatch` so a `verify` call
    appends `report` to its own `report_sink` without running a real
    sandboxed command; every other tool name still runs through the
    real dispatcher unchanged, since what is under test here is the
    walkthrough wiring around a verification result, not `verify`'s own
    sandboxed behavior (see `tests/system/test_p040_artifact_pane_live.py`
    for that)."""

    def _dispatch(
        event: ToolCallEvent, *, repo_root: Path, **context: object
    ) -> ToolResult:
        """Stand in for the real dispatcher: fake `verify`, forward
        every other tool name unchanged."""
        if event.name != "verify":
            return _real_dispatch(event, repo_root=repo_root, **context)
        report_sink = context.get("report_sink")
        if isinstance(report_sink, list):
            report_sink.append(report)
        return ToolResult(tool_call_id=event.id, content="verify: stub passed")

    monkeypatch.setattr(loop_module, "dispatch", _dispatch)


def _write_run_config(config_dir: Path) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    OpenRouter-routed entry, redirected to the hermetic mock server
    entirely through the `KESTREL_OPENROUTER_BASE_URL` environment seam
    -- the same shape `test_p049_plan_task_scripted.py` already writes."""
    models_toml = config_dir / "models.toml"
    models_toml.write_text(
        """\
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
""",
        encoding="utf-8",
    )
    kestrel_toml = config_dir / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "glm-5.2"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def _run_env(monkeypatch: pytest.MonkeyPatch, *, openrouter_base: str) -> None:
    """Set the environment variables `main`'s own config/registry
    resolution and the provider layer need against the mock backend."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", openrouter_base)
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)


def _write_done_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls."""
    chunks = [
        {
            "id": "chatcmpl-done",
            "object": "chat.completion.chunk",
            "created": 1700000020,
            "model": "z-ai/glm-5.2",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-done",
            "object": "chat.completion.chunk",
            "created": 1700000020,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-done",
            "object": "chat.completion.chunk",
            "created": 1700000020,
            "model": "z-ai/glm-5.2",
            "choices": [],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 6,
                "total_tokens": 46,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    lines: list[str] = []
    for chunk in chunks:
        lines.append("data: " + json.dumps(chunk))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- _print_task_summary ----------------------------------------------------


def test_print_task_summary_appends_the_walkthrough_line_when_given_a_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a `walkthrough_path`, when the summary is printed, then a
    final `walkthrough: {path}` line names it."""
    undo = UndoManager(repo_root=tmp_path)
    walkthrough_path = tmp_path / ".kestrel" / "artifacts" / "walkthrough-t-1.md"

    _print_task_summary(
        "t-1",
        _result(TerminationReason.TASK_COMPLETE),
        undo,
        CostMeter(),
        _entry(),
        walkthrough_path=walkthrough_path,
    )

    out = capsys.readouterr().out
    assert out.rstrip("\n").splitlines()[-1] == f"walkthrough: {walkthrough_path}"


def test_print_task_summary_omits_the_walkthrough_line_when_path_is_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given no `walkthrough_path` (the default), when the summary is
    printed, then no `walkthrough:` line appears -- every pre-existing
    caller that never passes it sees output identical to before this
    parameter existed."""
    undo = UndoManager(repo_root=tmp_path)

    _print_task_summary(
        "t-1", _result(TerminationReason.TASK_COMPLETE), undo, CostMeter(), _entry()
    )

    out = capsys.readouterr().out
    assert "walkthrough:" not in out


# --- _build_and_persist_walkthrough -----------------------------------------


def test_build_and_persist_walkthrough_returns_the_persisted_path(
    tmp_path: Path,
) -> None:
    """Given a task with no touched paths and no verification, when
    built and persisted, then the returned path exists and carries the
    exact rendered markdown `render_walkthrough_markdown` produces."""
    undo = UndoManager(repo_root=tmp_path)

    path = _build_and_persist_walkthrough(
        _result(TerminationReason.TASK_COMPLETE),
        task_id="t-2",
        undo=undo,
        verification_reports=(),
        repo_root=tmp_path,
    )

    assert path is not None
    assert path.is_file()
    expected = Walkthrough(
        task_id="t-2",
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=1,
        total_usd=Decimal("0.0071"),
        touched_paths=(),
        verification=None,
    )
    assert path.read_text(encoding="utf-8") == render_walkthrough_markdown(expected)


def test_build_and_persist_walkthrough_reports_a_walkthrougherror_to_stderr(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Given `.kestrel` replaced with a symlink escaping the repo root,
    when a walkthrough is built and persisted, then `None` is returned
    and the remedy is printed to stderr rather than raising -- a
    walkthrough that fails to persist must never crash the caller."""
    outside = tmp_path_factory.mktemp("cli-walkthrough-outside")
    (tmp_path / ".kestrel").symlink_to(outside, target_is_directory=True)
    undo = UndoManager(repo_root=tmp_path)

    path = _build_and_persist_walkthrough(
        _result(TerminationReason.TASK_COMPLETE),
        task_id="t-3",
        undo=undo,
        verification_reports=(),
        repo_root=tmp_path,
    )

    assert path is None
    assert list(outside.iterdir()) == []


# --- end-to-end: kestrel run / --resume -------------------------------------


def test_run_task_command_prints_and_persists_the_walkthrough_with_verification(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a fixture repo and a mock server scripted to reply with an
    `edit_file` call, then a `verify` call (its own dispatch stubbed to
    a passing report), then a plain no-more-tools reply, when `kestrel
    run` executes through `main`, then it exits 0, prints a
    `walkthrough:` line naming a real file, and that file's own content
    is exactly `render_walkthrough_markdown` of the expected
    `Walkthrough` -- `reason`/`turns`/`total_usd` read back from the
    summary's own printed lines, `touched_paths` naming the edited
    file, and an embedded, non-empty `## Verification` section."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")

    report = _passing_report()
    _stub_verify_dispatch(monkeypatch, report)

    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _TOOLCALL_VERIFY,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ]
    )
    _run_env(monkeypatch, openrouter_base=base_url)
    config_path = _write_run_config(tmp_path)

    exit_code = main(
        ["run", _TASK_TEXT, "--repo", str(repo_dir), "--config", str(config_path)]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "reason: TASK_COMPLETE" in out

    task_id_match = _TASK_ID_RE.search(out)
    assert task_id_match is not None, out
    task_id = task_id_match["task_id"]

    turns_match = _TURNS_RE.search(out)
    assert turns_match is not None, out
    total_usd_match = _TOTAL_USD_RE.search(out)
    assert total_usd_match is not None, out

    walkthrough_match = _WALKTHROUGH_LINE_RE.search(out)
    assert walkthrough_match is not None, out
    walkthrough_path = Path(walkthrough_match["path"])
    assert walkthrough_path.exists()
    assert walkthrough_path.name == f"walkthrough-{task_id}.md"

    expected = Walkthrough(
        task_id=task_id,
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=int(turns_match["turns"]),
        total_usd=Decimal(total_usd_match["total_usd"]),
        touched_paths=("greet.py",),
        verification=report,
    )
    persisted_text = walkthrough_path.read_text(encoding="utf-8")
    assert persisted_text == render_walkthrough_markdown(expected)
    assert "## Verification" in persisted_text
    assert "_no verification ran_" not in persisted_text
    assert "# Verification: PASSED" in persisted_text


def test_resume_task_command_also_prints_and_persists_a_walkthrough(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a task already run to `TASK_COMPLETE` once, when it is
    resumed via `kestrel run --resume` and reaches a second, ordinary
    completion, then `_resume_task_command` prints and persists its own
    walkthrough exactly like a fresh `run` does -- `touched_paths` still
    names the file the *original* run's own `edit_file` call touched,
    since both runs share the same task id's undo journal, and no
    `verify` call means `## Verification` reads `_no verification ran_`."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")
    config_path = _write_run_config(tmp_path)

    first_base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ]
    )
    _run_env(monkeypatch, openrouter_base=first_base_url)

    first_exit_code = main(
        [
            "run",
            _TASK_TEXT,
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ]
    )
    first_out = capsys.readouterr().out
    assert first_exit_code == 0

    task_id_match = _TASK_ID_RE.search(first_out)
    assert task_id_match is not None, first_out
    task_id = task_id_match["task_id"]

    followup_cassette = _write_done_cassette(
        tmp_path / "followup-done.sse", text="Nothing further needed."
    )
    second_base_url = mock_openai_server(
        cassette_sequence=[followup_cassette, _CRITIQUE_APPROVE]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", second_base_url)

    second_exit_code = main(
        [
            "run",
            "--resume",
            task_id,
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ]
    )
    second_out = capsys.readouterr().out

    assert second_exit_code == 0
    assert f"task_id: {task_id}" in second_out
    assert "reason: TASK_COMPLETE" in second_out

    walkthrough_match = _WALKTHROUGH_LINE_RE.search(second_out)
    assert walkthrough_match is not None, second_out
    walkthrough_path = Path(walkthrough_match["path"])
    assert walkthrough_path.exists()

    persisted_text = walkthrough_path.read_text(encoding="utf-8")
    assert "greet.py" in persisted_text
    assert "_no verification ran_" in persisted_text
