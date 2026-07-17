"""Textual command-palette provider surfacing every KES-TUI-003 entry.

`KestrelCommandProvider` is the sole bridge between the nine named
commands -- `/model`, `/mode`, `/plan`, `/fast`, `/undo`, `/cost`,
`/resume`, `/approve`, and `/kb` -- and Textual's own `ctrl+p` command
palette: it enumerates the candidate text for each one and lets the
palette's built-in fuzzy matcher rank and highlight them, rather than
this module reimplementing any of that ranking itself. Every match's
`command` callback delegates straight back to a `KestrelApp` method, so
this file holds no state and no business logic of its own -- it is
purely the candidate list and the wiring between a matched string and
the action it names.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import partial
from typing import TYPE_CHECKING, cast

from textual.command import Hit, Provider

if TYPE_CHECKING:
    from kestrel.tui.app import KestrelApp


class KestrelCommandProvider(Provider):
    """Yields one `Hit` per KES-TUI-003 command whose candidate text
    scores above zero against the palette's current query.

    Registered onto `KestrelApp.COMMANDS` (see `app.py`), alongside
    Textual's own built-in providers, so pressing `ctrl+p` and typing
    any of `/model`, `/mode`, `/plan`, `/fast`, `/undo`, `/cost`,
    `/resume`, `/approve`, or `/kb` finds and runs it through the same
    keyboard-first flow every other command palette entry uses.
    """

    async def search(self, query: str) -> AsyncIterator[Hit]:
        """Match `query` against every command this provider knows
        about and yield a `Hit` for each one that scores above zero.

        Five groups of candidates are considered, in order: one
        `/model <id>` per id in `app.registry`; `/mode plan`/`/plan`
        and `/mode fast`/`/fast` (both spellings, so either reads
        naturally); the four fixed, argument-free commands `/undo`,
        `/cost`, `/approve`, and `/kb`; and one `/resume <task_id>` per
        task with a session journal already on disk. `app.registry`,
        `app.list_resumable_task_ids()`, and the fixed command set are
        all read fresh on every call, so a palette search always
        reflects the app's current state rather than a snapshot taken
        when this provider was constructed.
        """
        app = cast("KestrelApp", self.app)
        matcher = self.matcher(query)

        for model_id in app.registry.ids():
            text = f"/model {model_id}"
            if (score := matcher.match(text)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(text),
                    partial(app.action_switch_model, model_id),
                )

        for mode in ("plan", "fast"):
            for text in (f"/mode {mode}", f"/{mode}"):
                if (score := matcher.match(text)) > 0:
                    yield Hit(
                        score,
                        matcher.highlight(text),
                        partial(app.action_set_mode, mode),
                    )

        for text, callback in (
            ("/undo", app.action_undo_current_task),
            ("/cost", app.action_show_cost),
            ("/approve", app.action_show_approve_info),
            ("/kb", app.action_show_kb_info),
        ):
            if (score := matcher.match(text)) > 0:
                yield Hit(score, matcher.highlight(text), callback)

        for task_id in app.list_resumable_task_ids():
            text = f"/resume {task_id}"
            if (score := matcher.match(text)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(text),
                    partial(app.action_resume_task, task_id),
                )
