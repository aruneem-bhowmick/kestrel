"""Unit tests for `kestrel.router.policy.resolve_model_id`: the pure
task-class-to-model-id lookup that generalizes the agent loop's own
budget-degradation "sorted ids, first tagged match, else fall back"
rule to any tag a policy names.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kestrel.config import RouterPolicyConfig
from kestrel.registry.model import ModelEntry, Registry, Tag
from kestrel.router.policy import TaskClass, resolve_model_id

pytestmark = [pytest.mark.p046, pytest.mark.unit]

_BACKEND = "openrouter"
_FALLBACK_MODEL_ID = "fallback-model"


def _entry(model_id: str, *, tags: frozenset[Tag] = frozenset()) -> ModelEntry:
    """Build one minimally-specified registry entry carrying `tags`,
    with otherwise arbitrary but valid rates and limits -- the tests in
    this module only ever inspect ids and tags."""
    return ModelEntry(
        id=model_id,
        backend=_BACKEND,
        provider_model=f"z-ai/{model_id}",
        api_key_env="OPENROUTER_API_KEY",
        context_window=100_000,
        max_output=8_192,
        usd_per_mtok_input=Decimal("1.00"),
        usd_per_mtok_output=Decimal("2.00"),
        usd_per_mtok_cached=Decimal("0.50"),
        supports_tools=True,
        supports_cache=True,
        tags=tags,
    )


def _registry(*entries: ModelEntry) -> Registry:
    """A `Registry` carrying exactly the given entries, keyed by their
    own ids."""
    return Registry(models={entry.id: entry for entry in entries}, source=None)


@pytest.mark.sanity
def test_resolves_to_the_registry_entry_carrying_the_policy_tag() -> None:
    """Given a registry with one `"planner"`-tagged and one `"cheap"`-
    tagged entry, when resolving `"plan"` against a policy mapping
    `"plan"` to `"planner"`, then the planner entry's id is returned."""
    registry = _registry(
        _entry("planner-model", tags=frozenset({"planner"})),
        _entry("cheap-model", tags=frozenset({"cheap"})),
    )
    policy: dict[TaskClass, Tag] = {
        "plan": "planner",
        "execute": "executor",
        "critique": "cheap",
        "trivial": "cheap",
        "embed": "local",
    }

    result = resolve_model_id(
        "plan",
        registry=registry,
        policy=policy,
        fallback_model_id=_FALLBACK_MODEL_ID,
    )

    assert result == "planner-model"


@pytest.mark.sanity
def test_ties_resolve_to_the_alphabetically_first_id_deterministically() -> None:
    """Given two entries both tagged `"cheap"`, when resolving `"trivial"`
    twice in a row, then the alphabetically-first id wins both times --
    proving the tie-break is deterministic, not accidental."""
    registry = _registry(
        _entry("zebra-cheap", tags=frozenset({"cheap"})),
        _entry("alpha-cheap", tags=frozenset({"cheap"})),
    )
    policy: dict[TaskClass, Tag] = {
        "plan": "planner",
        "execute": "executor",
        "critique": "cheap",
        "trivial": "cheap",
        "embed": "local",
    }

    first = resolve_model_id(
        "trivial",
        registry=registry,
        policy=policy,
        fallback_model_id=_FALLBACK_MODEL_ID,
    )
    second = resolve_model_id(
        "trivial",
        registry=registry,
        policy=policy,
        fallback_model_id=_FALLBACK_MODEL_ID,
    )

    assert first == "alpha-cheap"
    assert second == "alpha-cheap"


@pytest.mark.sanity
def test_no_matching_tag_falls_back_to_fallback_model_id() -> None:
    """Given a registry with no `"local"`-tagged entry, when resolving
    `"embed"`, then `fallback_model_id` is returned unchanged rather
    than raising or picking an unrelated entry."""
    registry = _registry(
        _entry("planner-model", tags=frozenset({"planner"})),
        _entry("cheap-model", tags=frozenset({"cheap"})),
    )
    policy: dict[TaskClass, Tag] = {
        "plan": "planner",
        "execute": "executor",
        "critique": "cheap",
        "trivial": "cheap",
        "embed": "local",
    }

    result = resolve_model_id(
        "embed",
        registry=registry,
        policy=policy,
        fallback_model_id=_FALLBACK_MODEL_ID,
    )

    assert result == _FALLBACK_MODEL_ID


@pytest.mark.sanity
def test_all_five_task_classes_resolve_against_default_policy() -> None:
    """Given a registry with one entry per tag and `RouterPolicyConfig()
    .as_mapping()` used directly (not a hand-built dict), when every
    task class is resolved, then each returns the entry carrying its
    own default tag -- proving the config's own defaults are wired
    correctly end to end."""
    registry = _registry(
        _entry("planner-entry", tags=frozenset({"planner"})),
        _entry("executor-entry", tags=frozenset({"executor"})),
        _entry("cheap-entry", tags=frozenset({"cheap"})),
        _entry("local-entry", tags=frozenset({"local"})),
    )
    policy = RouterPolicyConfig().as_mapping()

    assert (
        resolve_model_id(
            "plan",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
        == "planner-entry"
    )
    assert (
        resolve_model_id(
            "execute",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
        == "executor-entry"
    )
    assert (
        resolve_model_id(
            "critique",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
        == "cheap-entry"
    )
    assert (
        resolve_model_id(
            "trivial",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
        == "cheap-entry"
    )
    assert (
        resolve_model_id(
            "embed",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
        == "local-entry"
    )


def test_unknown_task_class_raises_key_error() -> None:
    """Given a `policy` mapping that omits an entry for the requested
    task class, when resolving it, then `KeyError` is raised rather
    than silently degrading -- a missing policy entry is a caller-side
    configuration bug, not a routing outcome."""
    registry = _registry(_entry("only-entry", tags=frozenset({"planner"})))
    policy: dict[TaskClass, Tag] = {"plan": "planner"}

    with pytest.raises(KeyError):
        resolve_model_id(
            "execute",
            registry=registry,
            policy=policy,
            fallback_model_id=_FALLBACK_MODEL_ID,
        )
