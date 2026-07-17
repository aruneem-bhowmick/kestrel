"""System test: submitting a task through the TUI's own task-input box
drives it through the real tool-calling agent loop against a scripted
mock-server cassette sequence and a real fixture repo -- the
conversation pane streams the assistant's own text, the status bar
updates live, and the run ends with a terse termination summary.

Reuses `test_p022_loop_scripted_task.py`'s own mock-server-plus-`bwrap`
fixture-repo pattern. Skipped locally when `bwrap` is not on `PATH`,
exactly like that suite; CI installs `bubblewrap` on every runner, so
this suite always actually runs there.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input
from textual.widgets import RichLog as _RichLog

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.sandbox import bwrap_available
from kestrel.tui.app import ConversationPane, KestrelApp, StatusBar

pytestmark = [
    pytest.mark.p038,
    pytest.mark.system,
    pytest.mark.ui,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EXECUTE_PYTEST = _CASSETTES / "toolcall_execute_pytest.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"
_TASK_TEXT = "read src/greet.py, then run the test suite"


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


def _write_fixture_repo(tmp_path: Path) -> None:
    """Write the same small fixture repo
    `test_p022_loop_scripted_task.py` uses: one Python module the
    scripted `read_file` call reads, and an empty test file the
    scripted `execute` call runs pytest against."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_greet.py").write_text("", encoding="utf-8")


async def test_submitting_a_task_streams_the_conversation_pane_and_status_bar(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo and a mock server scripted to reply with a
    `read_file` tool call, then an `execute` tool call, then a plain
    no-more-tools reply, when a task is submitted through
    `#task_input`, then: (1) the conversation pane's content contains
    the streamed assistant text from the final reply; (2)
    `ConversationPane.clear()` is never called during the run; (3) the
    status bar's rendered line changes at least once and ends with a
    nonzero session spend and a real context-usage figure; (4) the
    final conversation line names `TASK_COMPLETE`; (5) `_current_task_id`
    is cleared once the task ends, so a later submission is accepted
    again.
    """
    _write_fixture_repo(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _TOOLCALL_EXECUTE_PYTEST,
            _DONE_CASSETTE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    clear_calls: list[None] = []
    original_clear = _RichLog.clear

    def _spy_clear(self: _RichLog, *args: object, **kwargs: object) -> _RichLog:
        clear_calls.append(None)
        return original_clear(self, *args, **kwargs)

    monkeypatch.setattr(ConversationPane, "clear", _spy_clear)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
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

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        lines = [strip.text for strip in conversation.lines]
        content = "\n".join(lines)

        assert "Task complete." in content
        assert clear_calls == []

        final_status = str(status_bar.render())
        assert final_status != idle_status
        assert "session $0.0000" not in final_status
        assert "ctx --%" not in final_status

        assert "TASK_COMPLETE" in lines[-1]

        assert isinstance(pilot.app, KestrelApp)
        assert pilot.app._current_task_id is None
