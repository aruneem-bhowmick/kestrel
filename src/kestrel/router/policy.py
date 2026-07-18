"""Task-class routing: resolve a named task class to a real model id.

`resolve_model_id` generalizes `kestrel.agent.loop._find_cheap_entry`'s
own "sorted registry ids, first entry carrying the wanted tag, fallback
when none does" rule from one hardcoded tag (`"cheap"`) to any `Tag` a
caller-supplied policy names for any of the five task classes this
module recognizes. Nothing in this module calls a model or reads
`kestrel.toml` itself -- both a real `Registry` and a resolved
`Mapping[TaskClass, Tag]` (see `kestrel.config.RouterPolicyConfig`) are
supplied by the caller, keeping this a pure, easily tested function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from kestrel.registry.model import Registry, Tag

TaskClass = Literal["plan", "execute", "critique", "trivial", "embed"]


def resolve_model_id(
    task_class: TaskClass,
    *,
    registry: Registry,
    policy: Mapping[TaskClass, Tag],
    fallback_model_id: str,
) -> str:
    """The first (sorted by id, for determinism) registry entry
    carrying the `Tag` `policy[task_class]` names, or `fallback_model_id`
    when no entry does -- generalizes `agent.loop._find_cheap_entry`'s
    own "sorted ids, first match, else fall back" rule from one
    hardcoded tag to any tag a policy names. Never raises: an
    unreachable policy (e.g. no `"local"`-tagged entry exists yet)
    degrades to `fallback_model_id` rather than failing routing
    outright.

    Raises:
        KeyError: `task_class` has no entry in `policy` -- a genuine
            caller-side configuration bug (every real `RouterPolicyConfig
            .as_mapping()` covers all five classes by construction), not
            a routing outcome to degrade gracefully from.
    """
    tag = policy[task_class]
    for model_id in registry.ids():
        if tag in registry.get(model_id).tags:
            return model_id
    return fallback_model_id
