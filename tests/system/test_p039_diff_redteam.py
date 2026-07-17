"""Red-team system test: a hostile terminal escape sequence sitting in a
file's own *before* state -- the exact `ansi_escape_laden_payload`
injection-corpus case -- must never survive into the diff pane's own
rendered plain text, even though `DiffPane.show_diff` builds its
unified diff before sanitizing it, not the other way around.

Mirrors `test_p038_tui_conversation_redteam.py`'s own drive-through-the-
real-cockpit approach, this time proving the same guarantee at the diff
pane's own rendering layer rather than the conversation pane's.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tui.app import DiffPane, KestrelApp

pytestmark = [pytest.mark.p039, pytest.mark.system, pytest.mark.ui, pytest.mark.redteam]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EDIT_ANSI_PAYLOAD = _CASSETTES / "toolcall_edit_ansi_payload.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_HOSTILE_CASE_ID = "ansi_escape_laden_payload"
_TARGET_PATH = "payload.txt"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


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


async def test_diff_pane_never_renders_raw_escape_bytes_from_a_hostile_before_state(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture file whose *before* content is the
    `ansi_escape_laden_payload` corpus case's own payload, when a task
    submitted through `#task_input` edits it, then the diff pane's
    rendered `Syntax` object's plain text carries none of the payload's
    raw escape bytes, while the diff's own removed/added line structure
    -- and the surrounding, non-escape text on both sides of the edit --
    survives intact."""
    case = _find_case(_HOSTILE_CASE_ID)
    (tmp_path / _TARGET_PATH).write_text(case.payload, encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_EDIT_ANSI_PAYLOAD, _DONE_CASSETTE],
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
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "edit payload.txt"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        plain_text = diff_pane.content.code

        assert "\x1b" not in plain_text
        assert "\x9b" not in plain_text
        assert "\x07" not in plain_text

        assert "-before" in plain_text
        assert "+BEFORE-EDITED" in plain_text
        assert "after" in plain_text
        assert "@@" in plain_text
