"""Unit tests for `kestrel.task_setup.build_task_deps`'s own
knowledge-base wiring: `[kb].enabled` (default `True`) resolves the
`"embed"` task class through `kestrel.router.policy.resolve_model_id`
and constructs a real `kestrel.kb.service.KbService`, wired into
`LoopDeps.kb`; disabling it leaves that field at `agent.loop`'s own
`None` default. Building the collaborator itself never makes a model
call -- one only happens once a task's own turn actually reads from or
writes to the knowledge base.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import KbConfig, KestrelConfig
from kestrel.kb.embeddings import OllamaEmbeddingClient
from kestrel.kb.service import DEFAULT_EMBEDDING_DIM, KbService
from kestrel.registry.loader import load_registry
from kestrel.registry.model import ModelEntry, Registry
from kestrel.task_setup import build_task_deps

pytestmark = [pytest.mark.p059, pytest.mark.unit]

_MODEL_ID = "glm-5.2"


def _registry_without_a_local_route() -> Registry:
    """A single-entry `Registry` carrying no `"local"`-tagged embed
    route -- used only by the disabled-kb case below, where no embed
    resolution should happen regardless of what the registry contains."""
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


def test_kb_enabled_by_default_builds_a_real_kb_service_against_nomic_embed_text(
    tmp_path: Path,
) -> None:
    """Given the default config (`[kb].enabled` left at `True`) and the
    packaged default registry, when a task's deps are built, then
    `deps.kb` is a real `KbService` whose `embedding_model_id` resolved
    to `"nomic-embed-text"` -- the only `"local"`-tagged entry the
    default `[router.policy].embed` mapping can reach -- scoped to this
    task's own repo, config, and `DEFAULT_EMBEDDING_DIM`."""
    registry = load_registry()
    config = KestrelConfig()
    setup = build_task_deps(
        config=config,
        registry=registry,
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert isinstance(setup.deps.kb, KbService)
    assert setup.deps.kb.embedding_model_id == "nomic-embed-text"
    assert setup.deps.kb.repo_root == tmp_path
    assert setup.deps.kb.config is config.kb
    assert setup.deps.kb.embedding_dim == DEFAULT_EMBEDDING_DIM
    assert isinstance(setup.deps.kb.embedding_client, OllamaEmbeddingClient)


def test_kb_disabled_leaves_deps_kb_at_the_agent_loop_default(tmp_path: Path) -> None:
    """Given `[kb].enabled = False`, when a task's deps are built, then
    `deps.kb` is `None` -- `agent.loop`'s own default, unchanged by a
    disabled knowledge base, and no embed resolution or `KbService`
    construction ever runs."""
    config = KestrelConfig(kb=KbConfig(enabled=False))
    setup = build_task_deps(
        config=config,
        registry=_registry_without_a_local_route(),
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert setup.deps.kb is None


@pytest.mark.cost_regression
def test_building_kb_service_makes_no_model_call_and_prices_nothing(
    tmp_path: Path,
) -> None:
    """Given `[kb].enabled` at its default, when a task's deps are
    built, then no cost is priced and no turn is recorded -- the same
    zero-turn baseline `test_p038_task_setup.py` already pins for every
    other collaborator this function builds -- proving that
    constructing `KbService` alone never embeds anything; an embedding
    call only happens once a task's own turn actually reads from or
    writes to the knowledge base."""
    registry = load_registry()
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=registry,
        model_id=_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert setup.deps.kb is not None
    assert setup.meter.turns == ()
    assert setup.deps.meter.session_usd == Decimal(0)
