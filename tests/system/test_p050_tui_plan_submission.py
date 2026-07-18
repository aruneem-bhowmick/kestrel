"""System test: submitting a task while the cockpit's own mode is
`"plan"` drives a real, read-only PLAN-mode task through `KestrelApp`
and renders the resulting `ImplementationPlan` in the artifact pane --
proving `_prepare_task_run`'s new `mode_manager` wiring and
`_show_plan_from_result` cooperate correctly against a genuine
`run_task` call driven through the real cockpit, not merely asserted
against a hand-built `LoopResult` the way a unit test would.

Reuses `tests/acceptance/test_p043_dod_phase_3.py`'s own `run_test()`-
driven, `pilot.press`-first style, and `test_p049_plan_task_scripted.py`'s
own scripted read_file-then-plan-reply cassette shape. Every scripted
sequence below also interleaves the routed self-critique check's own
request per real turn, since `[managers.self_critique]` is enabled by
default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.agent.plan import (
    ImplementationPlan,
    parse_plan_lines,
    render_plan_markdown,
)
from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import sanitize_terminal
from kestrel.tui.app import ArtifactPane, DiffPane, KestrelApp, ToolLogPane

pytestmark = [pytest.mark.p050, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_TASK_TEXT = "read src/greet.py, then plan how to add auth"


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


def _write_plan_reply_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls -- standing in for the
    PLAN-mode model turn whose reply is the plan itself."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000011,
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
            "created": 1700000011,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000011,
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


async def test_plan_mode_submission_renders_the_plan_in_the_artifact_pane(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the cockpit's mode switched to `"plan"` and a mock server
    scripted to reply with one `read_file` exploration turn followed by
    a plain numbered-plan reply, when a task is submitted through
    `#task_input`, then: the artifact pane's rendered content matches
    the expected plan's own rendered markdown; `_plan_task_id` names the
    submitted task; `_last_plan.lines` carries the expected count and
    text; the plan is persisted under `.kestrel/artifacts/`; and the
    tool log and diff pane show only the `read_file` call -- no
    `edit_file`/`execute` line ever appears, since PLAN mode never
    offers either tool.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )

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
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        task_id = pilot.app._last_completed_task_id
        assert task_id is not None
        expected_plan = ImplementationPlan(
            task_id=task_id, raw_text=_PLAN_TEXT, lines=parse_plan_lines(_PLAN_TEXT)
        )

        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_plan)
        )

        assert pilot.app._plan_task_id == task_id
        assert pilot.app._last_plan is not None
        assert [line.text for line in pilot.app._last_plan.lines] == list(
            _PLAN_TEXT.splitlines()
        )

        persisted = list((tmp_path / ".kestrel" / "artifacts").glob("plan-*.md"))
        assert len(persisted) == 1

        tool_log = pilot.app.query_one("#tool_log", ToolLogPane)
        tool_log_content = "\n".join(strip.text for strip in tool_log.lines)
        assert "read_file(" in tool_log_content
        assert "edit_file(" not in tool_log_content
        assert "execute(" not in tool_log_content

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        assert diff_pane.content == "no changes yet"
