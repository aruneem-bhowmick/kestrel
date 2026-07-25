"""System test: `search`'s `semantic` mode driven through a real
`run_task` call -- extends `test_p022_loop_scripted_task.py`'s own mock
chat-completions harness with `deps.kb` set to a real `KbService`,
pre-seeded (via `add_note`) with one note, proving the whole chain
(schema -> `parse_search_args` -> `dispatch` -> `search`'s own
knowledge-base branch) folds a real knowledge-base match back into the
next model call's request body, merged alongside `rg`'s own hit from a
fixture repo file.

A second scenario proves `semantic: true` degrades silently to
`rg`-only output when `deps.kb` is `None` -- no crash, no
knowledge-base section.

Skipped locally when `rg` is not on `PATH`, exactly like
`test_p015_search_rg.py`: the binary genuinely may not be installed.
CI installs `ripgrep` on every runner, so this suite always actually
runs there.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.config import KbConfig
from kestrel.cost.meter import CostMeter
from kestrel.kb.service import KbService
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [
    pytest.mark.p060,
    pytest.mark.system,
    pytest.mark.skipif(shutil.which("rg") is None, reason="rg not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_SEARCH_SEMANTIC = _CASSETTES / "toolcall_search_semantic.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_PATTERN = "widget"
_KB_VECTOR = (1.0, 0.0, 0.0, 0.0)
_KB_NOTE_TEXT = "widgets are documented in docs/widgets.md"


@dataclasses.dataclass
class _FixedEmbeddingClient:
    """An `EmbeddingClient` returning a fixed vector for every text --
    this suite only needs a deterministic query/note vector pair to seed
    and then re-find one note, never a semantically meaningful one."""

    vector: tuple[float, ...] = _KB_VECTOR

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Return `self.vector` once per input text, regardless of its
        content."""
        return tuple(self.vector for _ in texts)


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


async def _seeded_kb_service(tmp_path: Path) -> KbService:
    """A real `KbService`, scoped to `tmp_path`, pre-seeded with one note
    via `add_note` -- backed by a real, temp-file-backed `KnowledgeStore`
    and a fixed-vector fake embedding client, so no real embedding model
    or network call is needed to prove the merge itself."""
    service = KbService(
        repo_root=tmp_path,
        config=KbConfig(),
        embedding_client=_FixedEmbeddingClient(),
        embedding_model_id="fake-embed",
        embedding_dim=len(_KB_VECTOR),
    )
    await service.add_note(_KB_NOTE_TEXT, tags=(), source_task="seed-task")
    return service


def _build_deps(tmp_path: Path, *, kb: KbService | None) -> LoopDeps:
    """Assemble a `LoopDeps` bundle scoped to `tmp_path`, varying only
    `kb` across this suite's two scenarios."""
    registry = _registry()
    return LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        kb=kb,
    )


def _write_matching_fixture_file(tmp_path: Path) -> None:
    """Write one repo file `_PATTERN` also matches via `rg`, so a real
    search finds exactly one ordinary hit alongside whatever the
    knowledge base contributes."""
    (tmp_path / "widgets.py").write_text("# widget factory\n", encoding="utf-8")


async def test_semantic_search_folds_a_real_kb_hit_into_the_next_tool_role_message(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo file the pattern also matches, and `deps.kb`
    set to a real `KbService` pre-seeded with one note, when a mock
    server scripts a `search` tool call with `semantic: true` followed by
    a plain no-more-tools reply, then the task completes and the second
    model call's own request body carries both an ordinary `rg` line and
    a `[kb score=...]` line for the seeded note.
    """
    _write_matching_fixture_file(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_SEARCH_SEMANTIC, _DONE_CASSETTE],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    kb = await _seeded_kb_service(tmp_path)
    deps = _build_deps(tmp_path, kb=kb)

    result = await run_task(f"search for {_PATTERN}", deps, task_id="sys-p060-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert len(captured) == 2
    second_request = captured[1]
    assert b"widgets.py" in second_request
    assert b"[kb score=" in second_request
    assert _KB_NOTE_TEXT.encode("utf-8") in second_request


async def test_semantic_search_degrades_silently_when_kb_is_none(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the identical scripted `semantic: true` call but `deps.kb`
    left at `None`, when the task runs, then it still completes and the
    second model call's own request body carries the ordinary `rg` line
    but no `[kb score=...]` section at all -- a disabled knowledge base
    never crashes or blocks a semantic search, it just narrows the
    result to `rg` alone.
    """
    _write_matching_fixture_file(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_SEARCH_SEMANTIC, _DONE_CASSETTE],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    deps = _build_deps(tmp_path, kb=None)

    result = await run_task(f"search for {_PATTERN}", deps, task_id="sys-p060-2")

    assert result.reason == TerminationReason.TASK_COMPLETE
    second_request = captured[1]
    assert b"widgets.py" in second_request
    assert b"[kb score=" not in second_request
