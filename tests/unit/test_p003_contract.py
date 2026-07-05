"""Contract test: model registry entries round-trip losslessly.

Registry entries are Kestrel's first stable data contract -- every later
prompt (the provider adapter, the cost meter, the REPL) reads them by
field, so a value that silently changed shape between validation and
serialization would corrupt everything downstream.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p003, pytest.mark.api]

_DIRECT_BACKEND_ENDPOINTS = {
    "zai": "https://api.z.ai/api/paas/v4",
    "ollama": "http://localhost:11434",
}


@st.composite
def _valid_model_entry_dicts(draw: st.DrawFn) -> dict[str, object]:
    """Build a models.toml-shaped dict guaranteed to pass ModelEntry
    validation, covering every backend and an arbitrary tag/rate
    combination."""
    backend = draw(st.sampled_from(["openrouter", "zai", "anthropic", "ollama"]))
    rate = st.decimals(
        min_value=0, max_value=1000, places=6, allow_nan=False, allow_infinity=False
    )

    entry: dict[str, object] = {
        "id": draw(st.text(min_size=1, max_size=20).filter(str.strip)),
        "backend": backend,
        "provider_model": draw(st.text(min_size=1, max_size=40).filter(str.strip)),
        "context_window": draw(st.integers(min_value=1, max_value=2_000_000)),
        "max_output": draw(st.integers(min_value=1, max_value=200_000)),
        "usd_per_mtok_input": draw(rate),
        "usd_per_mtok_output": draw(rate),
        "usd_per_mtok_cached": draw(rate),
        "supports_tools": draw(st.booleans()),
        "supports_cache": draw(st.booleans()),
        "tags": draw(
            st.lists(
                st.sampled_from(["planner", "executor", "cheap", "local"]),
                unique=True,
            )
        ),
    }
    if backend in _DIRECT_BACKEND_ENDPOINTS:
        entry["endpoint"] = _DIRECT_BACKEND_ENDPOINTS[backend]
    return entry


@given(_valid_model_entry_dicts())
def test_model_entry_round_trips_through_validation(raw: dict[str, object]) -> None:
    """Given any dict shaped like a valid models.toml entry, when it is
    validated into a ModelEntry and dumped back to JSON, then re-validating
    that JSON reproduces an identical ModelEntry -- the schema has no
    silent lossy corners."""
    entry = ModelEntry.model_validate(raw)

    reloaded = ModelEntry.model_validate(json.loads(entry.model_dump_json()))

    assert reloaded == entry
