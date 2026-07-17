"""Tests for the ctrl+p command palette: `KestrelCommandProvider.search`
driven directly against a mounted `KestrelApp`.

Every case here mounts a `KestrelApp` through `run_test()` -- a
`command.Provider` requires a real `Screen`/`App` context per Textual's
own contract (`self.app`, `self.screen` read from the screen the
provider was constructed against) -- so nothing here is a pure unit
test in the sense the rest of this package's `unit` marker usually
implies; it is still fast and network-free, matching every other
`ui`-marked suite in this directory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from textual.command import Hit

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import ConversationPane, KestrelApp
from kestrel.tui.commands import KestrelCommandProvider

pytestmark = [pytest.mark.p041, pytest.mark.ui]


def _model_entry(model_id: str) -> ModelEntry:
    """A minimal, cheap OpenRouter-routed `ModelEntry` for `model_id`,
    carrying only the fields `Registry`/`KestrelApp` actually require."""
    return ModelEntry(
        id=model_id,
        backend="openrouter",
        provider_model=f"z-ai/{model_id}",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )


def _registry(*model_ids: str) -> Registry:
    """A `Registry` carrying one `_model_entry` per id in `model_ids`."""
    return Registry(
        models={model_id: _model_entry(model_id) for model_id in model_ids},
        source=None,
    )


def _app(tmp_path: Path, *model_ids: str) -> KestrelApp:
    """A `KestrelApp` rooted at `tmp_path`, registered with one entry
    per id in `model_ids` and starting active on the first one."""
    return KestrelApp(
        config=KestrelConfig(),
        registry=_registry(*model_ids),
        model_id=model_ids[0],
        kestrel_md=None,
        repo_root=tmp_path,
    )


async def _hits(provider: KestrelCommandProvider, query: str) -> list[Hit]:
    """Drain `provider.search(query)`'s async generator into a plain
    list, so its results can be asserted on with ordinary list/count
    operations."""
    results: AsyncIterator[Hit] = provider.search(query)
    return [hit async for hit in results]


@pytest.mark.sanity
async def test_model_query_yields_one_hit_per_registered_id_and_switches_model(
    tmp_path: Path,
) -> None:
    """Given a registry with two models, when the palette is searched
    for "model", then it yields exactly one `/model <id>` hit per
    registered id; and when the second hit's own command runs, then
    `app.active_model_id` updates to that id."""
    app = _app(tmp_path, "glm-5.2", "glm-5.2-mini")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)

        hits = await _hits(provider, "model")
        # Textual's fuzzy matcher scores subsequences, not just
        # contiguous substrings, so "model" also weakly matches
        # unrelated candidates like "/mode plan" (m-o-d-e...-l).
        # Filtering to the "/model "-prefixed hits isolates the
        # property this case actually cares about: one hit per
        # registered id.
        model_hits = [
            hit
            for hit in hits
            if hit.text is not None and hit.text.startswith("/model ")
        ]

        assert sorted(hit.text for hit in model_hits if hit.text is not None) == [
            "/model glm-5.2",
            "/model glm-5.2-mini",
        ]

        mini_hit = next(hit for hit in model_hits if hit.text == "/model glm-5.2-mini")
        mini_hit.command()  # type: ignore[operator]

        assert pilot.app.active_model_id == "glm-5.2-mini"


@pytest.mark.sanity
async def test_mode_query_yields_both_spellings_and_switches_mode(
    tmp_path: Path,
) -> None:
    """Given a freshly mounted app in its default "fast" mode, when the
    palette is searched for "plan", then it yields both `/mode plan`
    and `/plan`; and when either hit's own command runs, then
    `app.mode_manager.mode` reads back as "plan"."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)

        hits = await _hits(provider, "plan")
        texts = {hit.text for hit in hits}

        assert "/mode plan" in texts
        assert "/plan" in texts

        target = next(hit for hit in hits if hit.text == "/plan")
        target.command()  # type: ignore[operator]

        assert pilot.app.mode_manager.mode == "plan"


async def test_kb_query_yields_a_hit_that_reports_unavailability(
    tmp_path: Path,
) -> None:
    """Given a mounted app, when the palette is searched for "kb", then
    it yields exactly one hit whose own command calls `app.notify` with
    a message naming the knowledge base as unavailable."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)
        notified: list[str] = []
        pilot.app.notify = notified.append  # type: ignore[method-assign]

        hits = await _hits(provider, "kb")

        assert [hit.text for hit in hits] == ["/kb"]
        hits[0].command()  # type: ignore[operator]

        assert len(notified) == 1
        assert "not available" in notified[0]


async def test_approve_query_yields_a_hit_that_reports_automatic_approval(
    tmp_path: Path,
) -> None:
    """Given a mounted app, when the palette is searched for "approve",
    then it yields exactly one hit whose own command calls `app.notify`
    with a message naming approvals as automatic."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)
        notified: list[str] = []
        pilot.app.notify = notified.append  # type: ignore[method-assign]

        hits = await _hits(provider, "approve")

        assert [hit.text for hit in hits] == ["/approve"]
        hits[0].command()  # type: ignore[operator]

        assert len(notified) == 1
        assert "automatically" in notified[0]


async def test_cost_query_yields_a_hit_reporting_no_turns_recorded_yet(
    tmp_path: Path,
) -> None:
    """Given a mounted app that has not yet run any task, when the
    palette is searched for "cost" and that hit's own command runs,
    then the conversation pane gains a "no turns recorded yet" line,
    since `KestrelApp._last_meter` is still unset."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)

        hits = await _hits(provider, "cost")

        assert [hit.text for hit in hits] == ["/cost"]
        hits[0].command()  # type: ignore[operator]

        conversation = pilot.app.query_one("#conversation", ConversationPane)
        lines = [strip.text for strip in conversation.lines]
        assert "no turns recorded yet" in lines


async def test_undo_query_yields_a_hit_warning_when_no_task_is_current(
    tmp_path: Path,
) -> None:
    """Given a mounted app with no task ever submitted, when the
    palette is searched for "undo" and that hit's own command runs,
    then `app.notify` fires a warning instead of attempting a revert."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)
        notified: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, *, severity="information", **_: (  # type: ignore[method-assign]
            notified.append((message, severity))
        )

        hits = await _hits(provider, "undo")

        assert [hit.text for hit in hits] == ["/undo"]
        hits[0].command()  # type: ignore[operator]

        assert notified == [("no task to undo yet", "warning")]


async def test_resume_query_yields_no_hits_without_a_sessions_directory(
    tmp_path: Path,
) -> None:
    """Given a repo with no `.kestrel/sessions/` directory at all, when
    the palette is searched for "resume", then it yields zero hits."""
    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)

        hits = await _hits(provider, "resume")

        assert hits == []


async def test_resume_query_yields_one_hit_per_journal_file(
    tmp_path: Path,
) -> None:
    """Given a repo with exactly one session journal on disk, when the
    palette is searched for "resume", then it yields exactly one hit
    naming that journal's own filename stem as the task id to
    resume."""
    sessions_dir = tmp_path / ".kestrel" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "task-abc123.jsonl").write_text("", encoding="utf-8")

    app = _app(tmp_path, "glm-5.2")
    async with app.run_test() as pilot:
        provider = KestrelCommandProvider(pilot.app.screen)

        hits = await _hits(provider, "resume")

        assert [hit.text for hit in hits] == ["/resume task-abc123"]
