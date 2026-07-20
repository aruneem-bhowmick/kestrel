"""System test: a FAST-mode task's own termination replaces whatever the
artifact pane showed mid-run with its auto-generated `Walkthrough` --
proving `KestrelApp._show_walkthrough_from_result` and `ArtifactPane.
show_walkthrough` cooperate correctly against a genuine `run_task`/
`resume_task` sequence driven through the real cockpit, not merely
asserted against a hand-built `Walkthrough` the way a unit test would.

Extends `tests/acceptance/test_p043_dod_phase_3.py`'s own read-then-edit
scripted scenario (no `verify` call) through to termination, and reuses
`tests/system/test_p052_tui_execute_plan_live.py`'s own PLAN-then-EXECUTE
scripted flow for the plan-execution case.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.agent.loop import TerminationReason
from kestrel.agent.plan import (
    ImplementationPlan,
    parse_plan_lines,
    render_plan_markdown,
)
from kestrel.agent.walkthrough import Walkthrough, render_walkthrough_markdown
from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import sanitize_terminal
from kestrel.tui.app import ArtifactPane, KestrelApp

pytestmark = [pytest.mark.p054, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "read src/greet.py, then implement greet in greet.py"
_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_PLAN_TASK_TEXT = "read src/greet.py, then plan how to add auth"
_EXECUTE_TEXT = "implement it"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassettes' own `model` field."""
    entry = ModelEntry(
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
    return Registry(models={"glm-5.2": entry}, source=None)


def _write_fixture_repo(repo_root: Path) -> None:
    """Write the same small fixture repo `test_p043_dod_phase_3.py`
    scripts its own task against: a `src/greet.py` module for the
    `read_file` call, and a top-level `greet.py` stub for the
    `edit_file` call to replace."""
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (repo_root / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


def _write_plan_reply_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls -- standing in for a
    PLAN-mode model turn, whose reply is the plan itself."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000030,
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
            "created": 1700000030,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000030,
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


async def test_fast_mode_task_completion_shows_the_walkthrough_in_the_artifact_pane(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo and a mock server scripted to reply with a
    `read_file` call, then an `edit_file` call, then a plain no-more-
    tools reply (no `verify` call anywhere in the scenario), when that
    task is submitted through `#task_input` and runs to completion,
    then the artifact pane's rendered content is exactly
    `render_walkthrough_markdown` of the task's own `Walkthrough` --
    not the placeholder text a fresh app starts with -- naming
    `greet.py` under `## Files changed` and reading `_no verification
    ran_` under `## Verification`.
    """
    _write_fixture_repo(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        placeholder = artifact_pane.source

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        task_id = pilot.app._last_completed_task_id
        assert task_id is not None
        meter = pilot.app._last_meter
        assert meter is not None

        expected = Walkthrough(
            task_id=task_id,
            reason=TerminationReason.TASK_COMPLETE,
            turns_used=3,
            total_usd=meter.session_usd,
            touched_paths=("greet.py",),
            verification=None,
        )
        expected_markdown = sanitize_terminal(render_walkthrough_markdown(expected))

        assert artifact_pane.source != placeholder
        assert artifact_pane.source == expected_markdown
        assert "## Files changed" in artifact_pane.source
        assert "greet.py" in artifact_pane.source
        assert "_no verification ran_" in artifact_pane.source

        persisted = list((tmp_path / ".kestrel" / "artifacts").glob("walkthrough-*.md"))
        assert len(persisted) == 1
        assert persisted[0].read_text(encoding="utf-8") == render_walkthrough_markdown(
            expected
        )


async def test_executing_a_plan_shows_the_walkthrough_naming_the_edited_file(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a completed PLAN-mode task on screen, when the cockpit's
    mode switches to `"fast"` and `#task_input` is resubmitted to
    execute it (`kestrel.tui.app.KestrelApp._execute_plan`), then the
    artifact pane ends showing that execution's own `Walkthrough` --
    unconditionally, since `_execute_plan` only ever runs in FAST mode
    -- naming the file its own `edit_file` call touched, replacing the
    `ImplementationPlan` the pane showed while the plan was still on
    screen.
    """
    _write_fixture_repo(tmp_path)

    plan_cassette = _write_plan_reply_cassette(
        tmp_path / "plan-reply.sse", text=_PLAN_TEXT
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            plan_cassette,
            _CRITIQUE_APPROVE,
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        pilot.app.action_set_mode("plan")
        await pilot.pause()

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _PLAN_TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        plan_task_id = pilot.app._last_completed_task_id
        assert plan_task_id is not None
        expected_plan = ImplementationPlan(
            task_id=plan_task_id,
            raw_text=_PLAN_TEXT,
            lines=parse_plan_lines(_PLAN_TEXT),
        )
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_plan)
        )

        pilot.app.action_set_mode("fast")
        await pilot.pause()

        task_input.focus()
        task_input.value = _EXECUTE_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app._plan_task_id is None
        assert pilot.app._last_plan is None
        meter = pilot.app._last_meter
        assert meter is not None

        expected_walkthrough = Walkthrough(
            task_id=plan_task_id,
            reason=TerminationReason.TASK_COMPLETE,
            turns_used=4,
            total_usd=meter.session_usd,
            touched_paths=("greet.py",),
            verification=None,
        )
        assert artifact_pane.source == sanitize_terminal(
            render_walkthrough_markdown(expected_walkthrough)
        )
        assert "greet.py" in artifact_pane.source
