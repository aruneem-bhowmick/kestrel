"""System test: the cockpit's own `KestrelApp._run_task` retrieves
knowledge-base context before a brand-new task's first turn, exactly
like the CLI's `_run_task_command` already does -- proven end to end
through a real, mounted `KestrelApp` rather than just against
`build_kb_context` in isolation.

Self-critique is disabled so the only chat-completion request this
task's single turn makes is its own first (and only) one, keeping the
captured request list unambiguous -- the same convention `test_p059_
kb_reaches_dispatch_context.py`/`test_p063_cli_run_kb_end_to_end.py`
already use for a KB-focused scenario.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig, ManagersConfig, SelfCritiqueConfig
from kestrel.kb.service import DEFAULT_EMBEDDING_DIM
from kestrel.kb.store import KnowledgeNote, KnowledgeStore, resolve_kb_path
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p066, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_KB_VECTOR: tuple[float, ...] = (1.0,) + (0.0,) * (DEFAULT_EMBEDDING_DIM - 1)
_SEEDED_NOTE_TEXT = "This repo's release script must run from a clean git tree"


def _registry(*, ollama_base_url: str) -> Registry:
    """One OpenRouter-routed chat entry matching the cassette's own
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


def _seed_note(repo_root: Path) -> None:
    """Insert `_SEEDED_NOTE_TEXT` directly into the fixture repo's own
    per-repo store, at `_KB_VECTOR` -- bypassing `KbService` (and
    therefore any embedding call) entirely, since the mock Ollama
    server below always answers with this exact vector anyway."""
    store = KnowledgeStore(
        db_path=resolve_kb_path(repo_root, global_=False),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )
    try:
        store.add_note(
            KnowledgeNote(
                id=None,
                text=_SEEDED_NOTE_TEXT,
                embedding=_KB_VECTOR,
                repo=str(repo_root.resolve()),
                tags=(),
                source_task="seed",
                timestamp=0.0,
            )
        )
    finally:
        store.close()


async def test_submitting_a_task_seeds_kb_context_into_the_first_outgoing_request(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    mock_ollama_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo whose knowledge base already holds one
    note, and a mock Ollama server that always answers with that exact
    note's own embedding vector, when a task is submitted through
    `#task_input`, then the very first outgoing chat-completion
    request's own message array carries a `<<<UNTRUSTED:kb:...>>>`-
    framed segment naming that note's text -- proving retrieval ran
    before the task's first turn, seeded through `run_task`'s own
    `kb_context` parameter.
    """
    _seed_note(tmp_path)

    ollama_base_url = mock_ollama_server(embeddings=[list(_KB_VECTOR)])

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE], capture=captured
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

    assert len(captured) >= 1
    first_request = captured[0]
    assert b"<<<UNTRUSTED:kb:" in first_request
    assert _SEEDED_NOTE_TEXT.encode("utf-8") in first_request
