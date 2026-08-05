"""Acceptance suite proving the Definition of Done's own "retrieval
measurably improves a scripted repeat-task benchmark (fewer turns/
tokens on second run)" clause as one executable scenario.

A first task discovers a repo convention the hard way -- spending an
extra turn on a `read_file` call before it can act correctly -- and its
own writeback commits a note stating that convention directly. A
second, related task then runs twice against the identical fixture
repo: once with the knowledge base left out of it entirely (replaying
the same discovery-turn cassette shape the first task needed), and once
with a real `KbService` already holding the first task's own note,
replaying a shorter cassette that skips the discovery turn because the
answer already arrived pre-seeded in `kb_context`. Both
`LoopResult.turns_used` and the cumulative token counts `deps.meter`
recorded come out strictly lower for the with-retrieval run -- the
literal DoD claim, encoded as an assertion rather than left as
narrative.

Every collaborator this composes -- `run_task`, `KbService`,
`kestrel.kb.retrieval.build_kb_context`, `kestrel.kb.writeback.
propose_learnings`/`commit_learnings`, `CostMeter` -- already has its
own dedicated suite elsewhere; this module's own job is wiring them
together into the same shape a real two-task session would actually
take.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.agent.walkthrough import build_walkthrough
from kestrel.config import KbConfig
from kestrel.cost.meter import CostMeter
from kestrel.kb.retrieval import build_kb_context
from kestrel.kb.service import DEFAULT_EMBEDDING_DIM, KbService
from kestrel.kb.writeback import ProposedLearning, commit_learnings, propose_learnings
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p067, pytest.mark.dod_phase_5, pytest.mark.acceptance]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_LEARNINGS_ONE = _CASSETTES / "learnings_one.sse"

_MODEL_ID = "glm-5.2"
_TASK_A_DESCRIPTION = (
    "figure out this repo's own docstring convention before touching anything"
)
_TASK_B_DESCRIPTION = "apply this repo's own docstring convention to a new module"
_LEARNING_TEXT = (
    "This repo's src/ modules keep their own docstring to one summary line, "
    "never a multi-paragraph block"
)
# A fixed, non-zero, `DEFAULT_EMBEDDING_DIM`-long vector -- the same trick
# `test_p063_cli_run_kb_end_to_end.py`'s own `_KB_VECTOR` plays at the store
# layer, applied here at the embedding-client layer instead: a note
# committed from Task A's own learning and a query embedded from Task B's
# own, unrelated description always score as a perfect cosine match, so
# this scenario needs no real semantic model to prove retrieval finds what
# writeback committed.
_KB_VECTOR: tuple[float, ...] = (1.0,) + (0.0,) * (DEFAULT_EMBEDDING_DIM - 1)


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry every phase of this
    scenario shares -- only the mock server each phase's own
    `KESTREL_OPENROUTER_BASE_URL` points at changes between them, never
    the registry entry itself."""
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


def _write_fixture_repo(repo_root: Path) -> None:
    """Write the one file `toolcall_read_file.sse`'s own scripted
    `read_file` call names (`src/greet.py`), standing in for whichever
    real file a model would read to discover this repo's own
    convention."""
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )


def _build_deps(
    client: LiteLLMClient,
    registry: Registry,
    repo_root: Path,
    *,
    kb: KbService | None,
) -> LoopDeps:
    """Assemble one task's own `LoopDeps`, scoped to `repo_root` and this
    scenario's shared registry -- `kb` is the only field that varies
    between this suite's three phases."""
    return LoopDeps(
        client=client,
        registry=registry,
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        kb=kb,
    )


def _cumulative_tokens(deps: LoopDeps) -> int:
    """Sum of every recorded turn's input-plus-output tokens for one
    task's own meter -- the DoD's own "tokens" half of "fewer turns/
    tokens on second run"."""
    return sum(turn.input_tokens + turn.output_tokens for turn in deps.meter.turns)


@dataclass(frozen=True, slots=True)
class _FixedVectorEmbeddingClient:
    """A stub `EmbeddingClient` (see `kestrel.kb.embeddings.
    EmbeddingClient`) returning one hand-picked, non-zero vector for
    every input, ignoring the text embedded entirely -- see this
    module's own docstring for why that is enough to prove the
    retrieval round trip without a real semantic model.
    """

    vector: tuple[float, ...]

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Return `self.vector` once per entry in `texts`, ignoring both
        `texts` and `model_id` entirely."""
        return tuple(self.vector for _ in texts)


async def test_retrieval_yields_fewer_turns_and_tokens_than_rediscovering_it(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given Task A's own writeback committed a note stating a repo
    convention it had to spend a `read_file` turn discovering, when a
    related Task B runs twice against the same fixture repo -- once with
    no knowledge base at all, replaying the identical discovery-turn
    cassette shape, and once with a real `KbService` already holding
    Task A's note, replaying a shorter cassette that skips the discovery
    turn -- then the with-retrieval run finishes in strictly fewer turns
    and spends strictly fewer cumulative tokens than the without-
    retrieval run.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _write_fixture_repo(repo_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    registry = _registry()

    # --- Task A: discovers the convention the hard way, then writes it back ---
    task_a_base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE, _DONE_CASSETTE, _LEARNINGS_ONE]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", task_a_base_url)
    task_a_client = LiteLLMClient(registry)
    deps_a = _build_deps(task_a_client, registry, repo_dir, kb=None)

    result_a = await run_task(_TASK_A_DESCRIPTION, deps_a, task_id="p067-bench-task-a")

    assert result_a.reason == TerminationReason.TASK_COMPLETE
    assert result_a.turns_used == 2

    kb = KbService(
        repo_root=repo_dir,
        config=KbConfig(),
        embedding_client=_FixedVectorEmbeddingClient(_KB_VECTOR),
        embedding_model_id="nomic-embed-text",
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )
    walkthrough = build_walkthrough(
        result_a,
        task_id="p067-bench-task-a",
        undo=deps_a.undo,
        verification_reports=deps_a.verification_reports,
    )
    proposed = await propose_learnings(
        walkthrough, client=task_a_client, model_id=_MODEL_ID
    )
    assert proposed == (ProposedLearning(text=_LEARNING_TEXT, tags=("style", "docs")),)
    committed = await commit_learnings(
        proposed, decisions=["approve"], task_id="p067-bench-task-a", kb=kb
    )
    assert len(committed) == 1

    # --- Task B without retrieval: pays the discovery turn again ---
    task_b_without_base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE, _DONE_CASSETTE]
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", task_b_without_base_url)
    task_b_without_client = LiteLLMClient(registry)
    deps_b_without = _build_deps(task_b_without_client, registry, repo_dir, kb=None)

    result_b_without = await run_task(
        _TASK_B_DESCRIPTION, deps_b_without, task_id="p067-bench-task-b-without"
    )

    assert result_b_without.reason == TerminationReason.TASK_COMPLETE
    assert result_b_without.turns_used == 2

    # --- Task B with retrieval: the answer already arrived in kb_context ---
    task_b_with_base_url = mock_openai_server(cassette_sequence=[_DONE_CASSETTE])
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", task_b_with_base_url)
    task_b_with_client = LiteLLMClient(registry)
    deps_b_with = _build_deps(task_b_with_client, registry, repo_dir, kb=kb)
    kb_context = await build_kb_context(_TASK_B_DESCRIPTION, kb=kb)
    assert kb_context is not None
    assert _LEARNING_TEXT in kb_context

    result_b_with = await run_task(
        _TASK_B_DESCRIPTION,
        deps_b_with,
        task_id="p067-bench-task-b-with",
        kb_context=kb_context,
    )

    assert result_b_with.reason == TerminationReason.TASK_COMPLETE
    assert result_b_with.turns_used == 1

    assert result_b_with.turns_used < result_b_without.turns_used
    assert _cumulative_tokens(deps_b_with) < _cumulative_tokens(deps_b_without)
