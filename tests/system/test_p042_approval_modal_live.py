"""System test: the TUI's own approval bridge, end to end and for real.

`kestrel.tui.approval_modal.make_tui_decide_fn` is what lets
`_dispatch_tool_call` -- run via `asyncio.to_thread` on a genuine
background OS thread, per `agent/loop.py`'s own `_drive` -- resolve a
destructive action's approval through a real `ApprovalModal` on the
app's own event loop instead of deadlocking or silently doing nothing.
Nothing here is mocked or stubbed on either side of that bridge: a real
background thread genuinely calls `asyncio.run_coroutine_threadsafe`
into a real Textual event loop, and blocks on a real
`concurrent.futures.Future` until a real key press resolves it.

Reuses `test_p038_tui_conversation_stream.py`'s own mock-server-plus-
`bwrap` fixture-repo pattern, scripting a destructive `execute` call via
`toolcall_execute_rm.sse` -- the same cassette
`test_p019_approval_cli.py`'s own real-sandbox approval suite already
established -- followed by `done_no_more_tools.sse`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input, Static

from kestrel.config import KestrelConfig
from kestrel.managers.session import load_session
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.framing import frame_untrusted
from kestrel.tools.execute import classify_destructive_action
from kestrel.tools.sandbox import bwrap_available
from kestrel.tui.app import ConversationPane, KestrelApp
from kestrel.tui.approval_modal import ApprovalModal

pytestmark = [
    pytest.mark.p042,
    pytest.mark.system,
    pytest.mark.ui,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EXECUTE_RM = _CASSETTES / "toolcall_execute_rm.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

# A stuck thread-to-event-loop bridge (the exact failure mode this suite
# exists to rule out) would otherwise hang `workers.wait_for_complete()`
# forever rather than failing the run; this bounds it to a generous but
# finite wait so a genuine deadlock surfaces as a test failure, not a
# hung CI job.
_WORKER_TIMEOUT_S = 30.0

_MODEL_ID = "glm-5.2"
_TARGET_NAME = "somefile"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassette's own `model` field."""
    entry = ModelEntry(
        id=_MODEL_ID,
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
    return Registry(models={_MODEL_ID: entry}, source=None)


def _boot_app(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> KestrelApp:
    """Write `somefile` under `tmp_path`, script a mock server replying
    with the `rm somefile` tool call followed by a plain no-more-tools
    reply, and return a fresh `KestrelApp` rooted at `tmp_path`."""
    (tmp_path / _TARGET_NAME).write_text("delete me\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_EXECUTE_RM, _DONE_CASSETTE],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    return KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
    )


async def _wait_for_modal(pilot: object, *, timeout: float = 5.0) -> ApprovalModal:
    """Poll until a genuine, fully mounted `ApprovalModal` sits atop
    `pilot.app`'s screen stack.

    The modal is scheduled from a real background thread via
    `asyncio.run_coroutine_threadsafe`, so there is a real (if small)
    race between that thread starting and the modal actually landing --
    a single `pilot.pause()` is not guaranteed to win it. Waits for
    `#approval_summary` to resolve too, not just the screen type, since
    a screen can be on the stack for one message loop turn before its
    own `compose()` children are actually mounted.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        screen = pilot.app.screen  # type: ignore[attr-defined]
        if isinstance(screen, ApprovalModal):
            try:
                screen.query_one("#approval_summary")
            except NoMatches:
                pass
            else:
                return screen
        await pilot.pause()  # type: ignore[attr-defined]
    raise AssertionError("ApprovalModal was never pushed within the timeout")


async def test_approval_modal_matches_classification_and_approving_once_runs_the_command(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a scripted destructive `rm` tool call, when the task is
    submitted, then: a real `ApprovalModal` is pushed with `summary`/
    `detail` text exactly matching `classify_destructive_action`'s own
    rendering for that command; pressing `"y"` resolves the pending
    `run_coroutine_threadsafe` future with `"once"`, letting the
    sandboxed `rm` genuinely run to completion through real `bwrap`
    (the target file is gone); and the task proceeds to
    `TASK_COMPLETE`."""
    target = tmp_path / _TARGET_NAME
    app = _boot_app(tmp_path, mock_openai_server, monkeypatch)

    expected_request = classify_destructive_action(["rm", _TARGET_NAME])
    assert expected_request is not None

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "delete somefile"
        await pilot.press("enter")

        modal = await _wait_for_modal(pilot)
        summary_widget = modal.query_one("#approval_summary", Static)
        detail_widget = modal.query_one("#approval_detail", Static)
        assert summary_widget.content == expected_request.summary
        assert detail_widget.content == expected_request.detail

        await pilot.press("y")
        await pilot.pause()
        await asyncio.wait_for(
            pilot.app.workers.wait_for_complete(), timeout=_WORKER_TIMEOUT_S
        )
        await pilot.pause()

        assert not target.exists()

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        lines = [strip.text for strip in conversation.lines]
        assert "TASK_COMPLETE" in lines[-1]

        assert isinstance(pilot.app, KestrelApp)
        assert pilot.app._current_task_id is None


async def test_denying_the_approval_leaves_the_file_and_carries_the_framed_refusal(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the same scripted destructive `rm` tool call, when `"n"` is
    pressed at the modal instead, then the delete is denied, the target
    file survives, and the task's own journaled history carries the
    exact framed `ApprovalDenied` refusal `_dispatch_tool_call`'s
    existing Phase-1 handling already produces for the CLI path -- a
    regression pin proving this new UI path reaches the identical
    refusal shape."""
    target = tmp_path / _TARGET_NAME
    app = _boot_app(tmp_path, mock_openai_server, monkeypatch)

    expected_request = classify_destructive_action(["rm", _TARGET_NAME])
    assert expected_request is not None
    expected_refusal = frame_untrusted(
        expected_request.summary, source="tool_stderr", origin="execute"
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "delete somefile"
        await pilot.press("enter")

        await _wait_for_modal(pilot)
        await pilot.press("n")
        await pilot.pause()
        await asyncio.wait_for(
            pilot.app.workers.wait_for_complete(), timeout=_WORKER_TIMEOUT_S
        )
        await pilot.pause()

        assert target.exists()

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        lines = [strip.text for strip in conversation.lines]
        assert "TASK_COMPLETE" in lines[-1]

        assert isinstance(pilot.app, KestrelApp)
        task_id = pilot.app._last_completed_task_id
        assert task_id is not None
        session = load_session(tmp_path, task_id)
        tool_contents = [m["content"] for m in session.history if m["role"] == "tool"]
        assert expected_refusal in tool_contents
