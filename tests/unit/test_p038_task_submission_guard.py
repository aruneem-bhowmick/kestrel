"""Unit tests for `KestrelApp.on_input_submitted`'s busy guard:
`_current_task_id` -- set for the duration of `_run_task` -- must stop a
second submission from ever starting a concurrent worker, since two
agent loops running at once would interleave writes into the same
conversation pane and status bar and act on the same repo
simultaneously.

Mounts a real `KestrelApp` (via `kestrel_app_factory`, see
`tests/unit/conftest.py`) but drives no real task -- `_current_task_id`
is set directly to simulate a task already in flight, keeping this
suite fast and network-free.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual.widgets import Input

from kestrel.tui.app import ConversationPane, KestrelApp

pytestmark = [pytest.mark.p038, pytest.mark.ui, pytest.mark.sanity]


async def test_busy_submission_is_declined_and_input_is_preserved(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given a task is already running, when another submission comes
    in, then no new worker starts (the conversation pane never sees the
    "> " echo `_run_task` writes for a genuinely accepted submission),
    a busy note is written instead, and the input's own text is left in
    place rather than cleared, so the user can resubmit once the
    running task ends."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        app._current_task_id = "already-running"

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "second task"
        await pilot.press("enter")
        await pilot.pause()

        assert task_input.value == "second task"

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)
        assert "busy" in content.lower()
        assert "> second task" not in content


async def test_empty_submission_still_clears_the_input_while_busy(
    kestrel_app_factory: Callable[[], KestrelApp],
) -> None:
    """Given a task is already running, when an empty submission
    arrives, then it is treated exactly like an empty submission while
    idle -- the input is cleared and nothing is written to the
    conversation pane -- rather than triggering the busy note, since an
    empty submission was never going to start a task in the first
    place."""
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        app._current_task_id = "already-running"

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "   "
        await pilot.press("enter")
        await pilot.pause()

        assert task_input.value == ""

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)
        assert "busy" not in content.lower()


async def test_idle_submission_after_busy_state_clears_is_accepted(
    kestrel_app_factory: Callable[[], KestrelApp],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a prior busy flag has since been cleared (as `_run_task`'s
    own `finally` block always does), when a submission comes in, then
    it is accepted -- the input clears and `_run_task` actually runs --
    same as if no task had ever run.

    `_run_task` itself is stubbed out here so this stays a fast,
    network-free unit test; `tests/system/test_p038_tui_conversation_stream.py`
    already covers a real task end to end.
    """
    async with kestrel_app_factory().run_test() as pilot:
        app = pilot.app
        assert isinstance(app, KestrelApp)
        assert app._current_task_id is None

        calls: list[str] = []

        async def _fake_run_task(text: str) -> None:
            """Record the submitted text instead of driving a real task."""
            calls.append(text)

        monkeypatch.setattr(app, "_run_task", _fake_run_task)

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "a fresh task"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert task_input.value == ""
        assert calls == ["a fresh task"]
