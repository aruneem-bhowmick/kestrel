"""System test: once a FAST-mode task's `Walkthrough` renders, the
cockpit proposes durable learnings from it and, when at least one comes
back, pushes `WritebackModal`; confirming it persists every checked
entry into the repo's own `.kestrel/kb.sqlite3` -- the TUI's own
end-to-end counterpart to `test_p063_cli_run_kb_end_to_end.py`'s CLI
round trip.

Self-critique is disabled, exactly like `test_p066_tui_kb_retrieval_
stream.py`, so the scripted task's own single turn and the writeback
proposal call that follows it are the *only* two chat-completion
requests this suite has to account for.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig, ManagersConfig, SelfCritiqueConfig
from kestrel.kb.service import DEFAULT_EMBEDDING_DIM
from kestrel.kb.store import KnowledgeStore, resolve_kb_path
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp
from kestrel.tui.writeback_modal import WritebackModal

pytestmark = [pytest.mark.p066, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_LEARNINGS_TWO = _CASSETTES / "learnings_two.sse"

_KB_VECTOR: tuple[float, ...] = (1.0,) + (0.0,) * (DEFAULT_EMBEDDING_DIM - 1)
_LEARNING_ONE_TEXT = (
    "This repo's Makefile targets assume tabs, not spaces, for recipe indentation"
)
_LEARNING_TWO_TEXT = "Run `uv run pytest -m sanity` before pushing any change here"


def _registry(*, ollama_base_url: str) -> Registry:
    """One OpenRouter-routed chat entry matching both cassettes' own
    `model` field, plus one `"local"`-tagged Ollama entry pointed at
    the mock embedding server."""
    chat_entry = ModelEntry(
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
    embed_entry = ModelEntry(
        id="nomic-embed-text",
        backend="ollama",
        provider_model="nomic-embed-text",
        endpoint=ollama_base_url,
        context_window=8192,
        max_output=1,
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=False,
        supports_cache=False,
        tags=frozenset({"local"}),
    )
    return Registry(models={"glm-5.2": chat_entry, "nomic-embed-text": embed_entry}, source=None)


async def test_confirming_the_writeback_modal_persists_both_proposed_learnings(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    mock_ollama_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a scripted task that ends `TASK_COMPLETE` followed by a
    two-learning writeback proposal reply, when the task is submitted
    through `#task_input` and, once `WritebackModal` appears, its
    `commit` button is pressed with every checkbox left at its default
    checked state, then both learnings land in the repo's own
    `.kestrel/kb.sqlite3`.
    """
    ollama_base_url = mock_ollama_server(embeddings=[list(_KB_VECTOR)])

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE, _LEARNINGS_TWO]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    config = KestrelConfig(
        managers=ManagersConfig(self_critique=SelfCritiqueConfig(enabled=False))
    )
    app = KestrelApp(
        config=config,
        registry=_registry(ollama_base_url=ollama_base_url),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "say hello"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, WritebackModal)
        assert len(modal._checkboxes) == 2

        await pilot.click("#commit")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    store = KnowledgeStore(
        db_path=resolve_kb_path(tmp_path, global_=False),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )
    try:
        stored = store.search(_KB_VECTOR, top_k=10)
    finally:
        store.close()
    stored_texts = {scored.note.text for scored in stored}
    assert stored_texts == {_LEARNING_ONE_TEXT, _LEARNING_TWO_TEXT}


@pytest.mark.cost_regression
async def test_writeback_proposal_usage_never_lands_in_the_task_meter(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    mock_ollama_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the identical scripted scenario, when the task completes
    and its writeback proposal call is made, then `KestrelApp._last_
    meter` -- the task's own `CostMeter`, exposed by `/cost` -- still
    carries exactly the one real turn the task itself billed, never a
    second entry for the writeback proposal's own usage: `complete_
    short_text` (what `propose_learnings` calls) never threads a
    `CostMeter` through at all, mirroring P-063's own identical
    assertion for the CLI path.
    """
    ollama_base_url = mock_ollama_server(embeddings=[list(_KB_VECTOR)])

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE, _LEARNINGS_TWO]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    config = KestrelConfig(
        managers=ManagersConfig(self_critique=SelfCritiqueConfig(enabled=False))
    )
    app = KestrelApp(
        config=config,
        registry=_registry(ollama_base_url=ollama_base_url),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "say hello"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, WritebackModal)
        for checkbox in modal._checkboxes:
            checkbox.value = False
        await pilot.click("#commit")
        await pilot.pause()

        assert isinstance(pilot.app, KestrelApp)
        meter = pilot.app._last_meter
        assert meter is not None
        assert len(meter.turns) == 1
