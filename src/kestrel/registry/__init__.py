"""Model registry: ``models.toml`` schema, loader, and validation errors."""

from kestrel.registry.loader import load_registry
from kestrel.registry.model import (
    Backend,
    ModelEntry,
    Registry,
    RegistryError,
    Tag,
    UnknownModelError,
)

__all__ = [
    "Backend",
    "ModelEntry",
    "Registry",
    "RegistryError",
    "Tag",
    "UnknownModelError",
    "load_registry",
]
