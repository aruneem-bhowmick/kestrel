"""Unit tests for `KestrelApp.action_show_kb_info`: a disabled knowledge
base notifies its own dedicated message without ever constructing a
`KbService`, and an enabled one reports a real note count read back
from a store seeded directly through `KnowledgeStore.add_note`,
bypassing the app entirely.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.tui.app as app_module
from kestrel.config import KbConfig, KestrelConfig
from kestrel.kb import store as kb_store
from kestrel.kb.store import KnowledgeNote, KnowledgeStore, resolve_kb_path
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p066, pytest.mark.unit]

_DIM = 768  # DEFAULT_EMBEDDING_DIM


@pytest.fixture
def global_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh, empty directory standing in for the real per-user data
    directory a global-namespace store would otherwise resolve to."""
    return tmp_path_factory.mktemp("globaldata")


@pytest.fixture(autouse=True)
def _patch_global_path(monkeypatch: pytest.MonkeyPatch, global_data_dir: Path) -> None:
    """Point `resolve_kb_path`'s own global-path lookup at `global_data_dir`
    so no test in this module ever touches a real per-user directory."""
    monkeypatch.setattr(
        kb_store.platformdirs,
        "user_data_dir",
        lambda appname: str(global_data_dir),  # noqa: ARG005
    )


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry -- `action_show_kb_
    info` never actually calls the embedding client, so this is the
    only entry any case here needs."""
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


def _app(tmp_path: Path, *, kb_enabled: bool) -> KestrelApp:
    """A `KestrelApp` scoped to `tmp_path`, with `[kb].enabled` set to
    `kb_enabled` and every other setting at its default."""
    return KestrelApp(
        config=KestrelConfig(kb=KbConfig(enabled=kb_enabled)),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )


def _seed_note(db_path: Path, *, repo: str) -> None:
    """Insert one note directly into the store at `db_path`, bypassing
    both `KestrelApp` and `KbService` entirely."""
    store = KnowledgeStore(db_path=db_path, embedding_dim=_DIM)
    try:
        store.add_note(
            KnowledgeNote(
                id=None,
                text="seeded note",
                embedding=(1.0,) + (0.0,) * (_DIM - 1),
                repo=repo,
                tags=(),
                source_task="seed",
                timestamp=0.0,
            )
        )
    finally:
        store.close()


def test_disabled_kb_notifies_and_never_constructs_a_kb_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `[kb].enabled=False`, when `/kb` runs, then the disabled
    message is notified and no worker is even started -- checked by
    making the app module's own `build_kb_service` name raise if it is
    ever called, and never touching a mounted app or `run_worker` at
    all, since the disabled branch returns before either would be
    needed."""

    def _must_not_be_called(*args: object, **kwargs: object) -> None:
        """Fail loudly if `action_show_kb_info` ever reaches this."""
        raise AssertionError("build_kb_service must not be called while kb is disabled")

    monkeypatch.setattr(app_module, "build_kb_service", _must_not_be_called)
    app = _app(tmp_path, kb_enabled=False)
    notifications: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **_: notifications.append(message)
    )

    app.action_show_kb_info()

    assert notifications == ["Knowledge base is disabled ([kb].enabled = false)."]


async def test_enabled_kb_notifies_the_real_per_repo_note_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `[kb].enabled=True` (the default) and two notes seeded
    directly into the per-repo store, when `/kb` runs, then the
    notified message names `"2 note(s)"` and omits any mention of the
    global namespace, since `global_namespace` is `False` by default.

    `action_show_kb_info` only schedules a worker -- a mounted app is
    required for `run_worker` to accept it at all, and the count itself
    lands once that worker's own `asyncio.to_thread` call resolves, so
    the assertions wait on `workers.wait_for_complete()` first.
    """
    repo = str(tmp_path.resolve())
    db_path = resolve_kb_path(tmp_path, global_=False)
    _seed_note(db_path, repo=repo)
    _seed_note(db_path, repo=repo)
    app = _app(tmp_path, kb_enabled=True)
    notifications: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **_: notifications.append(message)
    )

    async with app.run_test() as pilot:
        pilot.app.action_show_kb_info()
        await pilot.app.workers.wait_for_complete()

    assert len(notifications) == 1
    assert "2 note(s)" in notifications[0]
    assert "global" not in notifications[0]


async def test_enabled_kb_with_global_namespace_reports_both_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `[kb].global_namespace=True`, one note in the per-repo
    store, and two in the global store, when `/kb` runs, then the
    notified message names both counts."""
    repo = str(tmp_path.resolve())
    _seed_note(resolve_kb_path(tmp_path, global_=False), repo=repo)
    global_db = resolve_kb_path(tmp_path, global_=True)
    _seed_note(global_db, repo=repo)
    _seed_note(global_db, repo=repo)
    app = KestrelApp(
        config=KestrelConfig(kb=KbConfig(enabled=True, global_namespace=True)),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )
    notifications: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **_: notifications.append(message)
    )

    async with app.run_test() as pilot:
        pilot.app.action_show_kb_info()
        await pilot.app.workers.wait_for_complete()

    assert len(notifications) == 1
    assert "1 note(s) in this repo" in notifications[0]
    assert "2 in the global namespace" in notifications[0]


async def test_kb_service_error_while_counting_notifies_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a store that fails to open (its own `__init__` raises,
    modeling a disk or permissions failure), when `/kb` runs, then the
    raw error never escapes the worker -- it surfaces as a single
    warning-severity notification instead."""

    def _raise(*args: object, **kwargs: object) -> None:
        """Stand in for `KnowledgeStore.__init__`, always failing."""
        raise RuntimeError("disk full")

    monkeypatch.setattr(KnowledgeStore, "__init__", _raise)
    app = _app(tmp_path, kb_enabled=True)
    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *, severity="information", **_: notifications.append(
            (message, severity)
        ),
    )

    async with app.run_test() as pilot:
        pilot.app.action_show_kb_info()
        await pilot.app.workers.wait_for_complete()

    assert len(notifications) == 1
    message, severity = notifications[0]
    assert severity == "warning"
    assert "failed to open" in message
