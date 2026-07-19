"""System test: with a plan on screen, switching the cockpit's own mode
to `"fast"` and resubmitting continues that same task as a real,
tool-enabled execution -- proving `KestrelApp.on_input_submitted`'s
`executing_plan` branch and `_execute_plan` cooperate correctly against
a genuine `run_task`-then-`resume_task` sequence driven through the
real cockpit, not merely asserted against hand-built `LoopResult`
values the way a unit test would.

Extends `tests/system/test_p050_tui_plan_submission.py`'s own scripted
scenario -- the same fixture repo, the same initial `read_file`-then-
plan-reply cassette pair -- with a scripted FAST-mode continuation (an
`edit_file` call, then a plain no-more-tools reply) and a further,
unrelated FAST submission proving the one-shot plan-execution wiring
does not linger past its own single use. Every scripted sequence below
also interleaves the routed self-critique check's own request per real
turn, since `[managers.self_critique]` is enabled by default, exactly
as `test_p050_tui_plan_submission.py` already does.
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
from kestrel.tui import app as app_module
from kestrel.tui.app import ArtifactPane, ConversationPane, DiffPane, KestrelApp

pytestmark = [pytest.mark.p052, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "read src/greet.py, then plan how to add auth"
_EXECUTE_TEXT = "implement it"
_FOLLOWUP_TEXT = "summarize the readme"


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
            "created": 1700000013,
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
            "created": 1700000013,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000013,
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


def _write_fixture_repo(tmp_path: Path) -> None:
    """Write a fixture repo satisfying every scripted tool call: a
    `src/greet.py` module for the plan task's own `read_file` call, and
    a top-level `greet.py` stub -- the exact anchor
    `toolcall_edit_greet.sse`'s own `edit_file` call replaces -- for the
    execution's `edit_file` call."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (tmp_path / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


def _spy_on_turn_started(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap `TuiLoopObserver.on_turn_started` to record every `turn_id`
    it receives, in arrival order, across every task this test drives --
    proving a continued task's own turn numbering picks up where the
    prior one left off, while a genuinely new task's own numbering
    starts back over at `1`."""
    observed: list[int] = []
    original = app_module.TuiLoopObserver.on_turn_started

    def _wrapped(self: object, *, turn_id: int, active_model_id: str) -> None:
        """Record `turn_id`, then forward the call to the real hook so
        the status bar still refreshes exactly as it would unspied."""
        observed.append(turn_id)
        original(self, turn_id=turn_id, active_model_id=active_model_id)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module.TuiLoopObserver, "on_turn_started", _wrapped)
    return observed


def _spy_on_loop_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    """Wrap the module-level `run_task`/`resume_task` names `KestrelApp`
    itself calls, recording `(name, task_id)` for every invocation in
    arrival order -- distinguishing a submission that started a brand
    new task (`run_task`) from one that continued an existing one
    (`resume_task`), which `_plan_task_id`'s own presence or absence
    alone does not directly prove."""
    observed: list[tuple[str, str]] = []
    original_run_task = app_module.run_task
    original_resume_task = app_module.resume_task

    async def _wrapped_run_task(*args: object, **kwargs: object) -> object:
        """Record this call's own `task_id` (the third positional
        argument every real caller passes), then forward to the real
        `run_task`."""
        task_id = args[2]
        observed.append(("run_task", task_id))  # type: ignore[arg-type]
        return await original_run_task(*args, **kwargs)  # type: ignore[arg-type]

    async def _wrapped_resume_task(*args: object, **kwargs: object) -> object:
        """Record this call's own `task_id` (the first positional
        argument every real caller passes), then forward to the real
        `resume_task`."""
        task_id = args[0]
        observed.append(("resume_task", task_id))  # type: ignore[arg-type]
        return await original_resume_task(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module, "run_task", _wrapped_run_task)
    monkeypatch.setattr(app_module, "resume_task", _wrapped_resume_task)
    return observed


@pytest.mark.parametrize(
    ("execute_text", "expected_injected_message"),
    [
        pytest.param(_EXECUTE_TEXT, _EXECUTE_TEXT, id="typed-text"),
        pytest.param(
            "",
            "Proceed to implement the approved plan above.",
            id="blank-input",
        ),
    ],
)
async def test_switching_to_fast_and_resubmitting_executes_the_plan(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    execute_text: str,
    expected_injected_message: str,
) -> None:
    """Given a completed PLAN-mode task on screen, when the cockpit's
    mode switches to `"fast"` and `#task_input` is resubmitted with
    `execute_text` (either freshly typed text or, in the `blank-input`
    case, nothing at all), then: (1) the diff pane shows the real
    `edit_file` mutation the execution turn made; (2) the turn-id
    sequence observed across the plan task and its execution continues
    `1, 2, 3, 4` with no reset back to `1`; (3) `_plan_task_id` and
    `_last_plan` are both `None` afterward; (4) the conversation pane
    names `expected_injected_message` as what was sent -- the box's own
    text when non-empty, else the fixed "proceed as planned" default;
    and (5) a further, unrelated FAST submission starts a genuinely new
    task via `run_task` -- never another `resume_task` call against the
    already-executed plan -- proving the one-shot clearing actually
    took effect rather than leaving the plan re-executable.
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
    turns_observed = _spy_on_turn_started(monkeypatch)
    entry_points_observed = _spy_on_loop_entry_points(monkeypatch)

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
        expected_plan = ImplementationPlan(
            task_id=initial_task_id,
            raw_text=_PLAN_TEXT,
            lines=parse_plan_lines(_PLAN_TEXT),
        )
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_plan)
        )
        assert entry_points_observed == [("run_task", initial_task_id)]

        pilot.app.action_set_mode("fast")
        await pilot.pause()

        task_input.focus()
        task_input.value = execute_text
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app._last_completed_task_id == initial_task_id
        assert pilot.app._plan_task_id is None
        assert pilot.app._last_plan is None
        assert entry_points_observed == [
            ("run_task", initial_task_id),
            ("resume_task", initial_task_id),
        ]
        assert turns_observed == [1, 2, 3, 4]

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        diff_text = diff_pane.content.code
        assert "-# TODO: implement greet" in diff_text
        assert "+def greet(name: str) -> str:" in diff_text

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)
        assert f"executing plan: {expected_injected_message}" in content

        task_input.focus()
        task_input.value = _FOLLOWUP_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        followup_task_id = pilot.app._last_completed_task_id
        assert followup_task_id is not None
        assert followup_task_id != initial_task_id
        assert entry_points_observed[-1] == ("run_task", followup_task_id)
        assert turns_observed[-1] == 1
