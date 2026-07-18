"""Acceptance suite proving every Definition-of-Done clause for the fully
assembled Textual cockpit against one real, driven cockpit rather than
each pane's own isolated suite: the streaming conversation pane, tool
log, and diff view populate from a real multi-tool task; the status bar
reports live spend and context-window usage that matches the same task's
own `CostMeter`/`TurnCost` values exactly; and the command palette
resolves a partial query through real arrow-key navigation, with no
`pilot.click` anywhere in that scenario.

Every scenario here drives a real `KestrelApp` via `run_test()` against
the hermetic mock chat-completions server (see
``tests/fixtures/mock_openai.py``) and a real fixture repo -- none of it
depends on a live model or a live credential. The streaming/tool-log/
diff and status-bar scenarios reuse the exact cassette sequence and
fixture-repo shape `tests/system/test_p039_tool_log_diff_live.py`
already exercises pane by pane; this suite's own contribution is
asserting every clause together, against the same assembled run, the way
someone actually using the cockpit would experience it.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import (
    ConversationPane,
    DiffPane,
    KestrelApp,
    StatusBar,
    ToolLogPane,
)

pytestmark = [
    pytest.mark.p043,
    pytest.mark.acceptance,
    pytest.mark.system,
    pytest.mark.ui,
    pytest.mark.dod_phase_3,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
# `build_task_deps` now routes every turn's self-critique check through
# its own real, routed call by default, so every scripted sequence below
# must reply to it too -- one extra request per real turn, interleaved
# right after that turn's own.
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "read src/greet.py, then implement greet in greet.py"


def _registry(*model_ids: str) -> Registry:
    """A `Registry` carrying one cheap OpenRouter-routed entry per id in
    `model_ids`, all shaped to match the cassettes' own `model` field."""
    entries = {
        model_id: ModelEntry(
            id=model_id,
            backend="openrouter",
            provider_model=f"z-ai/{model_id}",
            api_key_env="OPENROUTER_API_KEY",
            context_window=200_000,
            max_output=16_384,
            usd_per_mtok_input=Decimal("0.60"),
            usd_per_mtok_output=Decimal("2.20"),
            usd_per_mtok_cached=Decimal("0.11"),
            supports_tools=True,
            supports_cache=True,
        )
        for model_id in model_ids
    }
    return Registry(models=entries, source=None)


def _write_fixture_repo(repo_root: Path) -> None:
    """Write the same small fixture repo
    `test_p039_tool_log_diff_live.py` scripts its own task against: a
    `src/greet.py` module for the `read_file` call, and a top-level
    `greet.py` stub for the `edit_file` call to replace."""
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (repo_root / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


async def test_dod_streaming_pane_tool_log_and_diff_view(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo and a mock server scripted to reply with a
    `read_file` call, then an `edit_file` call, then a plain no-more-
    tools reply, when that task is submitted through `#task_input`,
    then: the conversation pane shows the streamed final reply, the
    tool log shows two started/finished pairs in call order, and the
    diff pane's rendered content reflects the real `edit_file`
    mutation.
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
        registry=_registry("glm-5.2"),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)
        assert "Task complete." in content

        tool_log = pilot.app.query_one("#tool_log", ToolLogPane)
        tool_log_content = "\n".join(strip.text for strip in tool_log.lines)
        started_read = tool_log_content.index("-> read_file(")
        finished_read = tool_log_content.index("<- read_file (")
        started_edit = tool_log_content.index("-> edit_file(")
        finished_edit = tool_log_content.index("<- edit_file (")
        assert started_read < finished_read < started_edit < finished_edit

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        diff_text = diff_pane.content.code
        assert "-# TODO: implement greet" in diff_text
        assert "+def greet(name: str) -> str:" in diff_text
        assert '+    return f"Hello, {name}!"' in diff_text


async def test_dod_status_bar_live_dollars_and_context_pct(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the same scripted task as above, when it completes, then
    the status bar's rendered line carries a nonzero session-spend
    figure and a real (non `--`) context-usage percentage, and both
    match the task's own final `CostMeter`/`TurnCost` values exactly --
    not merely "some number changed."
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
        registry=_registry("glm-5.2"),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        status_bar = pilot.app.query_one("#status_bar", StatusBar)
        idle_status = str(status_bar.render())

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        final_status = str(status_bar.render())
        assert final_status != idle_status

        meter = pilot.app._last_meter
        assert meter is not None
        assert meter.session_usd > 0
        assert f"session ${meter.session_usd:.4f}" in final_status

        last_turn = meter.turns[-1]
        assert f"({last_turn.input_tokens}/200000)" in final_status
        assert "ctx --%" not in final_status


async def test_dod_command_palette_keyboard_first(tmp_path: Path) -> None:
    """Given a cockpit registered against two models and idle on the
    first, when the command palette is opened with `ctrl+p`, `/model`
    is typed, the highlight is moved down one entry with the arrow key,
    and `enter` is pressed -- entirely through `pilot.press`, never
    `pilot.click` -- then `active_model_id` switches to the second
    entry and the status bar's rendered line reflects it.
    """
    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry("glm-5.2", "glm-5.2-mini"),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"/model")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app.active_model_id == "glm-5.2-mini"

        status_bar = pilot.app.query_one("#status_bar", StatusBar)
        assert "glm-5.2-mini" in str(status_bar.render())
