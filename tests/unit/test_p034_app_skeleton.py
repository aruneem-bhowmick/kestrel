"""Sanity checks for the KestrelApp Textual skeleton: it mounts cleanly,
every pane resolves at its documented widget id, its stylesheet parses
without error, and the compose tree matches the documented status-bar
over two-column layout.

Every case here runs entirely in Textual's own headless test harness
(`App.run_test`) -- no network access, no sandbox, no model call -- so
the whole file stays well under the sanity gate's 30-second budget.
"""

from __future__ import annotations

import pytest
from textual.css.scalar import Unit
from textual.css.stylesheet import StylesheetParseError
from textual.widgets import Input

from kestrel.tui.app import (
    ArtifactPane,
    ConversationPane,
    DiffPane,
    KestrelApp,
    StatusBar,
    ToolLogPane,
)

pytestmark = [pytest.mark.p034, pytest.mark.ui, pytest.mark.sanity]


async def test_app_mounts_cleanly() -> None:
    """Given a freshly constructed KestrelApp, when it is run under
    Textual's headless test harness, then it starts and stops without
    raising."""
    async with KestrelApp().run_test():
        pass


async def test_stylesheet_parses_without_error() -> None:
    """Given KestrelApp's own kestrel.tcss, when the app mounts, then no
    StylesheetParseError is raised -- a failure here would surface as
    an exception out of run_test rather than a normal assertion, so
    this case exists to name that failure mode explicitly."""
    try:
        async with KestrelApp().run_test():
            pass
    except StylesheetParseError as exc:
        pytest.fail(f"kestrel.tcss failed to parse: {exc}")


async def test_every_pane_id_resolves() -> None:
    """Given a mounted KestrelApp, when each documented widget id is
    looked up, then `query_one` resolves it to the expected pane type
    rather than raising `NoMatches`."""
    async with KestrelApp().run_test() as pilot:
        pilot.app.query_one("#status_bar", StatusBar)
        pilot.app.query_one("#conversation", ConversationPane)
        pilot.app.query_one("#task_input", Input)
        pilot.app.query_one("#artifact", ArtifactPane)
        pilot.app.query_one("#tool_log", ToolLogPane)
        pilot.app.query_one("#diff", DiffPane)


async def test_status_bar_docks_top() -> None:
    """Given a mounted KestrelApp, when the status bar's own computed
    style is read, then it is docked to the top of the screen."""
    async with KestrelApp().run_test() as pilot:
        status_bar = pilot.app.query_one("#status_bar", StatusBar)
        assert status_bar.styles.dock == "top"


async def test_left_right_column_widths() -> None:
    """Given a mounted KestrelApp, when the two body columns' own
    computed widths are read, then the left column is 2fr and the
    right column is 1fr, matching the documented layout ratio."""
    async with KestrelApp().run_test() as pilot:
        left_width = pilot.app.query_one("#left_column").styles.width
        right_width = pilot.app.query_one("#right_column").styles.width

        assert left_width is not None
        assert left_width.value == 2.0
        assert left_width.unit == Unit.FRACTION

        assert right_width is not None
        assert right_width.value == 1.0
        assert right_width.unit == Unit.FRACTION


async def test_panes_are_split_across_the_documented_columns() -> None:
    """Given a mounted KestrelApp, when each pane's ancestry is
    inspected, then the conversation pane and task input live under the
    left column, and the artifact, tool-log, and diff panes live under
    the right column (the tool-log pane nested one level deeper, inside
    its own Collapsible)."""
    async with KestrelApp().run_test() as pilot:
        left_column = pilot.app.query_one("#left_column")
        right_column = pilot.app.query_one("#right_column")

        assert pilot.app.query_one("#conversation").parent is left_column
        assert pilot.app.query_one("#task_input").parent is left_column

        assert pilot.app.query_one("#artifact").parent is right_column
        assert pilot.app.query_one("#diff").parent is right_column
        assert right_column in pilot.app.query_one("#tool_log").ancestors
