"""Red-team system test: hostile terminal escape sequences in a model's
own streamed reply must never reach the TUI's conversation pane.

Reuses `openrouter_glm52_ansi.sse`, the same cassette
`test_p008_redteam_ansi.py` already established for the plain REPL,
driven this time through the TUI's own `#task_input` -- mirroring that
existing test's assertions at the cockpit layer, where every
model-sourced chunk now also passes through
`kestrel.tui.observer_bridge.TuiLoopObserver.on_text_delta` before
`ConversationPane.append_delta` ever buffers it.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import ConversationPane, KestrelApp

pytestmark = [pytest.mark.p038, pytest.mark.system, pytest.mark.ui, pytest.mark.redteam]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_ANSI_CASSETTE = _CASSETTES / "openrouter_glm52_ansi.sse"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassette's own `model` field."""
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


async def test_hostile_escape_sequences_never_reach_the_conversation_pane(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock backend replays a completion containing a
    screen-clear CSI sequence and an OSC window-title sequence, when a
    task is submitted through `#task_input`, then the raw escape bytes
    never appear in the conversation pane's rendered content while the
    surrounding text survives."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(_ANSI_CASSETTE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "hello"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        content = "\n".join(strip.text for strip in conversation.lines)

        assert "\x1b" not in content
        assert "\x9b" not in content
        assert "\x07" not in content
        assert "before" in content
        assert "after" in content
