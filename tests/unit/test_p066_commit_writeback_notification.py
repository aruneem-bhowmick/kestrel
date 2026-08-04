"""Unit test for `KestrelApp._commit_writeback`'s own notified count:
the number of *approved* learnings, not the number of `KnowledgeNote`
copies `commit_learnings` returns -- which is larger than
`len(approved)` whenever `[kb].global_namespace` is on, since
`commit_learnings` returns one persisted copy per store a learning
landed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import KestrelConfig
from kestrel.kb.store import KnowledgeNote
from kestrel.kb.writeback import ProposedLearning
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui.app import KestrelApp

pytestmark = [pytest.mark.p066, pytest.mark.unit]


class _TwoCopiesPerLearningKbService:
    """Stands in for a `KbService` with `global_namespace` enabled:
    `add_note` returns two persisted `KnowledgeNote`s per call, one per
    store a learning landed in -- the exact shape that would make
    `len(committed)` double-count `len(approved)` if the notification
    read the wrong one."""

    async def add_note(
        self, text: str, *, tags: Sequence[str], source_task: str
    ) -> tuple[KnowledgeNote, ...]:
        """Return two `KnowledgeNote`s sharing identical fields, standing
        in for a per-repo-plus-global write."""
        note = KnowledgeNote(
            id=1,
            text=text,
            embedding=(0.0,),
            repo="repo",
            tags=tuple(tags),
            source_task=source_task,
            timestamp=0.0,
        )
        return (note, note)


def _app(tmp_path: Path) -> KestrelApp:
    """A minimal `KestrelApp` scoped to `tmp_path` -- `_commit_writeback`
    never touches anything requiring a mounted screen once `notify` is
    monkeypatched, so no `run_test()` is needed here."""
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
    return KestrelApp(
        config=KestrelConfig(),
        registry=Registry(models={"glm-5.2": entry}, source=None),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )


async def test_notified_count_is_the_approved_count_not_the_persisted_copy_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two approved learnings committed against a `KbService`
    that returns two persisted copies per learning (the
    `global_namespace`-enabled shape), when the commit finishes, then
    the notified count is 2 (the approved learnings), not 4 (the
    persisted copies)."""
    app = _app(tmp_path)
    notifications: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **_: notifications.append(message)
    )
    approved = (
        ProposedLearning(text="first", tags=()),
        ProposedLearning(text="second", tags=()),
    )
    kb = _TwoCopiesPerLearningKbService()

    await app._commit_writeback(approved, task_id="task-1", kb=kb)  # type: ignore[arg-type]

    assert notifications == ["kb: committed 2 learning(s)"]
