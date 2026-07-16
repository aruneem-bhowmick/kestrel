"""Shared fixtures for unit tests that load a real config/registry file
pair from disk.

Only the tests that actually touch the filesystem or the environment
(config- and registry-backed doctor/CLI coverage) need these; anything
defining its own same-named fixture locally continues to shadow this
module's version, as pytest fixture resolution always prefers the
closer-scoped definition.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel import config as kestrel_config
from kestrel.config import KestrelConfig
from kestrel.registry import loader as registry_loader
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp


@pytest.fixture
def user_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user config
    directory, so tests never touch (or depend on) the real home directory.
    """
    return tmp_path_factory.mktemp("userconfig")


@pytest.fixture(autouse=True)
def _isolated_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_config_dir: Path
) -> None:
    """Chdir into an empty directory, clear ``$KESTREL_CONFIG`` and every
    known credential variable, and point both the config and registry
    user-config-dir lookups at an empty temp directory."""
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        kestrel_config.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )
    monkeypatch.setattr(
        registry_loader.platformdirs,
        "user_config_dir",
        lambda appname: str(user_config_dir),  # noqa: ARG005
    )


@pytest.fixture
def write_config() -> Callable[..., Path]:
    """Return a factory that writes a ``kestrel.toml`` + ``models.toml``
    pair under a given directory and returns the config path."""

    def _write(tmp_path: Path, models_toml: str, *, default_model: str) -> Path:
        """Write the pair, returning the ``kestrel.toml`` path."""
        models_file = tmp_path / "models.toml"
        models_file.write_text(models_toml, encoding="utf-8")

        kestrel_toml = tmp_path / "kestrel.toml"
        kestrel_toml.write_text(
            f"""\
[general]
default_model = "{default_model}"

[paths]
models_file = "{models_file.as_posix()}"
""",
            encoding="utf-8",
        )
        return kestrel_toml

    return _write


@pytest.fixture
def kestrel_app_factory(tmp_path: Path) -> Callable[[], KestrelApp]:
    """Return a factory building a fresh, minimally configured
    `KestrelApp` -- one registry entry, default config, no KESTREL.md --
    for tests that exercise the app's own layout, focus, and status-bar
    wiring without driving a real task."""

    def _build() -> KestrelApp:
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
        registry = Registry(models={"glm-5.2": entry}, source=None)
        return KestrelApp(
            config=KestrelConfig(),
            registry=registry,
            model_id="glm-5.2",
            kestrel_md=None,
            repo_root=tmp_path,
        )

    return _build
