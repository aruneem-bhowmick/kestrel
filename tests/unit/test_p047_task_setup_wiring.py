"""Unit tests for `kestrel.task_setup.build_task_deps`'s own self-critique
wiring: `[managers.self_critique].enabled` (default `True`) routes
`LoopDeps.self_critique_fn` through `kestrel.router.policy.resolve_model_id`
and `kestrel.agent.critique.make_self_critique_fn`; disabling it falls
back to `agent.loop`'s own always-approve default.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.critique import model_self_critique
from kestrel.agent.loop import _default_self_critique
from kestrel.config import KestrelConfig, ManagersConfig, SelfCritiqueConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.task_setup import build_task_deps

pytestmark = [pytest.mark.p047, pytest.mark.unit, pytest.mark.sanity]

_MAIN_MODEL_ID = "glm-5.2"
_CHEAP_MODEL_ID = "glm-5.2-cheap"


def _registry() -> Registry:
    """A two-entry `Registry`: the main model, plus a `"cheap"`-tagged
    entry distinct from it, so a test can tell whether self-critique
    routing actually consulted the tag rather than defaulting to the
    main model id."""
    main_entry = ModelEntry(
        id=_MAIN_MODEL_ID,
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
    cheap_entry = ModelEntry(
        id=_CHEAP_MODEL_ID,
        backend="openrouter",
        provider_model="z-ai/glm-5.2-cheap",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.10"),
        usd_per_mtok_output=Decimal("0.20"),
        usd_per_mtok_cached=Decimal("0.02"),
        supports_tools=True,
        supports_cache=True,
        tags=frozenset({"cheap"}),
    )
    return Registry(
        models={_MAIN_MODEL_ID: main_entry, _CHEAP_MODEL_ID: cheap_entry},
        source=None,
    )


def test_self_critique_enabled_by_default_routes_to_the_cheap_tagged_entry(
    tmp_path: Path,
) -> None:
    """Given the default config (`[managers.self_critique].enabled`
    left at `True`), when a task's deps are built, then
    `deps.self_critique_fn` is bound to `model_self_critique`, itself
    bound to the registry's own `"cheap"`-tagged entry -- not the main
    task's own model id -- and to `deps.client` itself, not a second
    client instance."""
    setup = build_task_deps(
        config=KestrelConfig(),
        registry=_registry(),
        model_id=_MAIN_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    self_critique_fn = setup.deps.self_critique_fn
    assert self_critique_fn.func is model_self_critique  # type: ignore[attr-defined]
    assert self_critique_fn.keywords["model_id"] == _CHEAP_MODEL_ID  # type: ignore[attr-defined]
    assert self_critique_fn.keywords["client"] is setup.deps.client  # type: ignore[attr-defined]


def test_self_critique_disabled_falls_back_to_the_always_approve_default(
    tmp_path: Path,
) -> None:
    """Given `[managers.self_critique].enabled = False`, when a task's
    deps are built, then `deps.self_critique_fn` is `agent.loop`'s own
    `_default_self_critique` -- the exact behavior every caller had
    before this config key existed."""
    config = KestrelConfig(
        managers=ManagersConfig(self_critique=SelfCritiqueConfig(enabled=False))
    )
    setup = build_task_deps(
        config=config,
        registry=_registry(),
        model_id=_MAIN_MODEL_ID,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id="task-1",
    )

    assert setup.deps.self_critique_fn is _default_self_critique
