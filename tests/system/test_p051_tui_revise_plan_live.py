"""System test: with a plan on screen, pressing `c` opens
`PlanCommentModal`; submitting one comment against it and then
resubmitting the (now-empty) task-input box drives `revise_plan`
instead of starting a brand new task, and the artifact pane refreshes
with the model's own revised plan -- proving `KestrelApp.action_comment
_on_plan`/`_on_plan_comment`/`_revise_plan` cooperate correctly against
a genuine `run_task`-then-`resume_task` sequence driven through the
real cockpit, not merely asserted against hand-built `LoopResult`
values the way a unit test would.

Extends `tests/system/test_p050_tui_plan_submission.py`'s own scripted
scenario -- the same fixture repo, the same initial `read_file`-then-
plan-reply cassette pair -- with a third cassette scripting the revised
plan reply. Every scripted sequence below also interleaves the routed
self-critique check's own request per real turn, since
`[managers.self_critique]` is enabled by default, exactly as
`test_p050_tui_plan_submission.py` already does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from kestrel.agent.plan import (
    ImplementationPlan,
    parse_plan_lines,
    render_plan_markdown,
)
from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import sanitize_terminal
from kestrel.tui.app import ArtifactPane, ConversationPane, KestrelApp
from kestrel.tui.plan_comment_modal import PlanCommentModal

pytestmark = [pytest.mark.p051, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_INITIAL_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_REVISED_PLAN_TEXT = (
    "1. Add an authentication middleware module built on Alembic.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_TASK_TEXT = "read src/greet.py, then plan how to add auth"
_COMMENT_TEXT = "use Alembic instead"


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
    `text` verbatim and requests no tool calls -- standing in for a
    PLAN-mode model turn, whose reply is the plan itself."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000012,
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
            "created": 1700000012,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000012,
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


async def test_commenting_on_a_plan_line_and_resubmitting_revises_the_plan(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the cockpit's mode switched to `"plan"` and a completed
    initial plan on screen, when `c` opens `PlanCommentModal`, one
    comment against line 1 is submitted, and the (now-empty)
    `#task_input` is resubmitted, then: the artifact pane's content
    becomes the *revised* plan's own rendering; `_pending_plan_comments`
    is empty again; and the conversation pane carries a "revising plan
    with 1 comment(s)" line.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )

    initial_plan_cassette = _write_plan_reply_cassette(
        tmp_path / "initial-plan-reply.sse", text=_INITIAL_PLAN_TEXT
    )
    revised_plan_cassette = _write_plan_reply_cassette(
        tmp_path / "revised-plan-reply.sse", text=_REVISED_PLAN_TEXT
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            initial_plan_cassette,
            _CRITIQUE_APPROVE,
            revised_plan_cassette,
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

        initial_task_id = pilot.app._last_completed_task_id
        assert initial_task_id is not None
        expected_initial_plan = ImplementationPlan(
            task_id=initial_task_id,
            raw_text=_INITIAL_PLAN_TEXT,
            lines=parse_plan_lines(_INITIAL_PLAN_TEXT),
        )
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_initial_plan)
        )

        await pilot.press("f4")
        await pilot.press("c")
        await pilot.pause()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, PlanCommentModal)
        modal.query_one("#plan_comment_line_number", Input).value = "1"
        modal.query_one("#plan_comment_text", Input).value = _COMMENT_TEXT

        modal.query_one("#submit", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert len(pilot.app._pending_plan_comments) == 1
        assert pilot.app._pending_plan_comments[0].line_index == 1
        assert pilot.app._pending_plan_comments[0].comment == _COMMENT_TEXT

        task_input.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        revised_task_id = pilot.app._last_completed_task_id
        assert revised_task_id == initial_task_id
        expected_revised_plan = ImplementationPlan(
            task_id=revised_task_id,
            raw_text=_REVISED_PLAN_TEXT,
            lines=parse_plan_lines(_REVISED_PLAN_TEXT),
        )
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_revised_plan)
        )

        assert pilot.app._pending_plan_comments == []

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)
        assert "revising plan with 1 comment(s)" in content
