"""System test: `kestrel run --mode plan` driven end to end through
`kestrel.cli.main`, against a real `LiteLLMClient` and a real mock
chat-completions server -- proving `build_task_deps`'s `mode_manager`
wiring and `_run_task_command`'s PLAN-mode branch cooperate correctly
against a genuine `run_task` call, not merely asserted against a
hand-built `LoopResult` the way a unit test would.

Every scenario below scripts one extra request per real turn for the
routed self-critique check (`[managers.self_critique]` is enabled by
default), exactly like `tests/acceptance/test_p023_dod_phase_1.py`
already does for its own CLI-driven scenarios.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.agent.plan import (
    ImplementationPlan,
    parse_plan_lines,
    render_plan_markdown,
)
from kestrel.cli import main

pytestmark = [pytest.mark.p049, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_TASK_ID_RE = re.compile(r"^task_id: (?P<task_id>\S+)$", re.MULTILINE)
_PLAN_LINE_RE = re.compile(r"^plan: (?P<path>.+)$", re.MULTILINE)

_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_GREET_STUB = "# TODO: implement greet\n"


def _write_run_config(config_dir: Path) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    OpenRouter-routed entry, redirected to the hermetic mock server
    entirely through the `KESTREL_OPENROUTER_BASE_URL` environment
    seam -- the same shape `test_p023_dod_phase_1.py` already writes."""
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


def _write_plan_reply_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls -- standing in for the
    PLAN-mode model turn whose reply is the plan itself."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000010,
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
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000010,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000010,
            "model": "z-ai/glm-5.2",
            "choices": [],
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 25,
                "total_tokens": 115,
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


def _expected_plan_markdown(task_id: str) -> str:
    """The exact markdown `render_plan_markdown` produces for
    `_PLAN_TEXT`, parsed under `task_id` -- the yardstick the persisted
    artifact on disk is checked against."""
    plan = ImplementationPlan(
        task_id=task_id, raw_text=_PLAN_TEXT, lines=parse_plan_lines(_PLAN_TEXT)
    )
    return render_plan_markdown(plan)


def test_plan_mode_run_persists_and_prints_the_plan_artifact(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a fixture repo and a mock server scripted to reply with one
    `read_file` exploration turn followed by a plain numbered-plan final
    reply, when `kestrel run --mode plan` executes through `main`, then
    it exits 0, prints a `plan:` line naming a real
    `.kestrel/artifacts/plan-*.md` file, and that file's own content is
    exactly `render_plan_markdown`'s output for the parsed reply."""
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )

    plan_cassette = _write_plan_reply_cassette(
        tmp_path / "plan-reply.sse", text=_PLAN_TEXT
    )
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            plan_cassette,
            _CRITIQUE_APPROVE,
        ]
    )
    _run_env(monkeypatch, openrouter_base=base_url)
    config_path = _write_run_config(tmp_path)

    exit_code = main(
        [
            "run",
            "add auth",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--mode",
            "plan",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "reason: TASK_COMPLETE" in out

    task_id_match = _TASK_ID_RE.search(out)
    assert task_id_match is not None, out
    task_id = task_id_match["task_id"]

    plan_line_match = _PLAN_LINE_RE.search(out)
    assert plan_line_match is not None, out
    plan_path = Path(plan_line_match["path"])
    assert plan_path.exists()
    assert plan_path.name == f"plan-{task_id}.md"
    assert plan_path.read_text(encoding="utf-8") == _expected_plan_markdown(task_id)


@pytest.mark.redteam
def test_plan_mode_run_refuses_an_edit_attempt_and_still_reaches_a_plan(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a scripted first turn that requests `edit_file` -- a
    confused-model case that should never happen from a well-behaved
    PLAN-mode reply -- when `kestrel run --mode plan` executes, then the
    refused-call path from `build_task_deps`'s `available_tools`
    restriction fires (the refusal text folds into the next turn's own
    request body rather than the file changing on disk), and the task
    still reaches a persisted, parseable plan on its second turn."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")
    original_bytes = (repo_dir / "greet.py").read_bytes()

    plan_cassette = _write_plan_reply_cassette(
        tmp_path / "plan-reply.sse", text=_PLAN_TEXT
    )
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            plan_cassette,
            _CRITIQUE_APPROVE,
        ],
        capture=captured,
    )
    _run_env(monkeypatch, openrouter_base=base_url)
    config_path = _write_run_config(tmp_path)

    exit_code = main(
        [
            "run",
            "implement greet in greet.py",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--mode",
            "plan",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "reason: TASK_COMPLETE" in out
    assert (repo_dir / "greet.py").read_bytes() == original_bytes

    task_id_match = _TASK_ID_RE.search(out)
    assert task_id_match is not None, out
    task_id = task_id_match["task_id"]

    plan_line_match = _PLAN_LINE_RE.search(out)
    assert plan_line_match is not None, out
    plan_path = Path(plan_line_match["path"])
    assert plan_path.read_text(encoding="utf-8") == _expected_plan_markdown(task_id)

    assert len(captured) == 4
    second_turn_request = captured[2].decode("utf-8")
    assert "is not available in this mode" in second_turn_request
    assert "edit_file" in second_turn_request
