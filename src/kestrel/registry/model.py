"""Model registry schema: ``models.toml`` entries, validated and frozen.

Two fields exist beyond the minimum schema needed to describe a model's
capabilities and price: ``provider_model`` (the name the backend itself
uses for the model -- required to actually place a call) and
``api_key_env`` (the *name* of the environment variable holding the
credential, never the credential itself). The registry inherits the
config loader's no-secrets-on-disk rule: it names where a key lives, it
never carries the key.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

Backend = Literal["openrouter", "zai", "anthropic", "ollama"]
Tag = Literal["planner", "executor", "cheap", "local"]

_ENDPOINT_REQUIRED_BACKENDS: frozenset[str] = frozenset({"zai", "ollama"})
_CODING_PLAN_PATH_SEGMENT = "/api/coding/"


class ModelEntry(BaseModel):
    """A single validated, immutable model registry entry.

    Attributes:
        id: Kestrel-stable key used everywhere else in the codebase to
            refer to this model (e.g. ``"glm-5.2"``).
        backend: Which provider integration routes calls to this entry.
        provider_model: The name the backend itself uses for the model
            (e.g. ``"zai-org/GLM-5.2"``); required to actually place a call.
        endpoint: Base URL for the backend's API. Required for direct
            backends (``"zai"``, ``"ollama"``); unused for aggregators
            that resolve routing internally.
        api_key_env: Name of the environment variable holding the API
            key -- never the key itself.
        context_window: Maximum total tokens (prompt + completion).
        max_output: Maximum completion tokens in a single turn.
        usd_per_mtok_input: Price per million input tokens.
        usd_per_mtok_output: Price per million output tokens.
        usd_per_mtok_cached: Price per million cache-hit input tokens.
        supports_tools: Whether the backend accepts tool/function schemas.
        supports_cache: Whether the backend supports prompt caching.
        precision_pin: Quantization pinned by the operator (e.g.
            ``"fp8"``), or ``None`` to leave precision provider-managed.
        tags: Routing hints (planner/executor role, relative cost,
            local-vs-hosted) consumed by later model-selection logic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    backend: Backend
    provider_model: str
    endpoint: str | None = None
    api_key_env: str | None = None
    context_window: int = Field(gt=0)
    max_output: int = Field(gt=0)
    usd_per_mtok_input: Decimal = Field(ge=0)
    usd_per_mtok_output: Decimal = Field(ge=0)
    usd_per_mtok_cached: Decimal = Field(ge=0)
    supports_tools: bool
    supports_cache: bool
    precision_pin: str | None = None
    tags: frozenset[Tag] = frozenset()

    @field_serializer("tags")
    def _serialize_tags_sorted(self, tags: frozenset[Tag]) -> list[str]:
        """Serialize tags in sorted order.

        A plain ``frozenset`` iterates in an order that depends on
        Python's per-process string hash seed, which would make any JSON
        dump of this model (logs, golden-file fixtures, future API
        payloads) non-deterministic between runs unless the output order
        is pinned explicitly.
        """
        return sorted(tags)

    @model_validator(mode="after")
    def _require_endpoint_for_direct_backends(self) -> ModelEntry:
        """Reject direct backends that omit the base URL they call."""
        if self.backend in _ENDPOINT_REQUIRED_BACKENDS and self.endpoint is None:
            raise ValueError(
                f"entry '{self.id}': backend '{self.backend}' requires an 'endpoint'"
            )
        return self

    @model_validator(mode="after")
    def _reject_coding_plan_endpoint(self) -> ModelEntry:
        """Enforce the ToS guard: Coding-Plan routes cannot be expressed.

        Z.ai's Coding-Plan quota is contractually restricted to recognized
        coding tools; Kestrel is a custom application and must only ever
        route through per-token/MaaS endpoints (spec section 2.2).
        """
        if self.endpoint is not None and _CODING_PLAN_PATH_SEGMENT.lower() in self.endpoint.lower():
            raise ValueError(
                f"entry '{self.id}': Coding-Plan endpoints are excluded by "
                "spec section 2.2; use a per-token endpoint."
            )
        return self


class RegistryError(Exception):
    """A ``models.toml`` parse or validation failure.

    Rendered as a human-readable message naming the file, the offending
    entry's position, and the field (or reason) at fault.
    """


class UnknownModelError(RegistryError):
    """Raised by :meth:`Registry.get` for an id with no matching entry."""

    def __init__(self, model_id: str, *, available: list[str]) -> None:
        """Store the requested id alongside every id that is available."""
        super().__init__(
            f"unknown model id '{model_id}'; available: {', '.join(available)}"
        )
        self.model_id = model_id
        self.available = available


class Registry(BaseModel):
    """The fully validated, immutable model registry for a session."""

    model_config = ConfigDict(frozen=True)

    models: Mapping[str, ModelEntry]
    source: Path | None

    @model_validator(mode="after")
    def _wrap_models_immutable(self) -> Registry:
        object.__setattr__(self, "models", MappingProxyType(self.models))
        return self

    @field_serializer("models")
    def _serialize_models(self, models: Mapping[str, ModelEntry]) -> dict[str, ModelEntry]:
        return dict(models)

    def get(self, model_id: str) -> ModelEntry:
        """Look up an entry by id.

        Raises:
            UnknownModelError: ``model_id`` has no matching entry; the
                error names every id that *is* available.
        """
        try:
            return self.models[model_id]
        except KeyError:
            raise UnknownModelError(model_id, available=self.ids()) from None

    def ids(self) -> list[str]:
        """Return every registered model id, sorted for stable display."""
        return sorted(self.models)
