"""Unit tests for `kestrel.tui.app.ToolLogPane`'s live rendering methods
and `kestrel.tui.app.DiffPane.show_diff`: exact line formats, the
argument-summary length cap, and unified-diff rendering for an ordinary
edit as well as a file's creation or deletion -- all driven directly
against a freshly constructed widget, with no mounted Textual app
needed, since `RichLog.write`/`Static.update` are safe to call before a
widget is ever mounted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.syntax import Syntax

from kestrel.provider.events import ToolCallEvent
from kestrel.tui.app import DiffPane, ToolLogPane

pytestmark = [pytest.mark.p039, pytest.mark.unit, pytest.mark.sanity]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p039_tool_log_lines.golden"
)


def _spy_write(pane: ToolLogPane, written: list[str]) -> None:
    """Replace `pane.write` with a spy recording each call's own first
    (content) argument verbatim, instead of the real `RichLog.write` --
    keeps these tests independent of Textual's own deferred-render
    machinery for an unmounted widget."""
    pane.write = written.append  # type: ignore[method-assign]


def test_append_started_writes_the_documented_arrow_line() -> None:
    """Given a tool call with a short argument payload, when
    `append_started` writes it, then the pane receives exactly
    `"-> {name}({summary})"` with the summary rendered verbatim."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    call = ToolCallEvent(
        id="call-1", name="read_file", arguments_json='{"path": "a.py"}'
    )

    pane.append_started(call)

    assert written == ['-> read_file({"path": "a.py"})']


def test_append_started_sanitizes_the_argument_summary() -> None:
    """Given a tool call whose argument payload carries a hostile
    terminal escape sequence, when `append_started` writes it, then the
    escape bytes are stripped before the line is ever written."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    call = ToolCallEvent(id="call-1", name="edit_file", arguments_json="x\x1b[2Jy")

    pane.append_started(call)

    assert written == ["-> edit_file(xy)"]


def test_append_started_caps_a_long_summary_at_120_chars_plus_ellipsis() -> None:
    """Given a tool call whose argument payload exceeds 120 characters,
    when `append_started` writes it, then the summary is truncated to
    exactly the first 120 characters with a trailing `"..."` -- never
    the full, possibly enormous, payload."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    long_args = "x" * 200
    call = ToolCallEvent(id="call-1", name="execute", arguments_json=long_args)

    pane.append_started(call)

    assert len(written) == 1
    line = written[0]
    assert line == f"-> execute({'x' * 120}...)"


def test_append_started_leaves_a_short_summary_uncapped() -> None:
    """Given a tool call whose argument payload is exactly 120
    characters, when `append_started` writes it, then no trailing
    `"..."` is appended -- the cap only ever bites when the summary is
    actually longer than the limit."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    exact_args = "y" * 120
    call = ToolCallEvent(id="call-1", name="search", arguments_json=exact_args)

    pane.append_started(call)

    assert written == [f"-> search({'y' * 120})"]


def test_append_finished_writes_the_documented_arrow_line() -> None:
    """Given a tool call and its own elapsed time, when
    `append_finished` writes it, then the pane receives exactly
    `"<- {name} ({elapsed_s:.1f}s)"`, rounded to one decimal place."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    call = ToolCallEvent(id="call-1", name="verify", arguments_json="{}")

    pane.append_finished(call, elapsed_s=2.34)

    assert written == ["<- verify (2.3s)"]


@pytest.mark.regression
def test_tool_log_lines_match_golden_snapshot() -> None:
    """One canonical `append_started`/`append_finished` rendering pair
    matches a pinned snapshot byte-for-byte, so an accidental wording or
    formatting change shows up here instead of silently drifting."""
    pane = ToolLogPane()
    written: list[str] = []
    _spy_write(pane, written)
    call = ToolCallEvent(
        id="call-1", name="read_file", arguments_json='{"path": "src/greet.py"}'
    )

    pane.append_started(call)
    pane.append_finished(call, elapsed_s=1.5)

    rendered = "\n".join(written) + "\n"
    assert rendered == _GOLDEN_FILE.read_text(encoding="utf-8")


def test_show_diff_renders_unified_diff_with_minus_and_plus_lines() -> None:
    """Given a simple before/after content pair, when `show_diff`
    renders them, then the resulting `Syntax` object's own plain text
    carries the expected removed and added lines."""
    pane = DiffPane()

    pane.show_diff("greet.py", "a\nb\nc\n", "a\nX\nc\n")

    assert isinstance(pane.content, Syntax)
    plain_text = pane.content.code
    assert "-b" in plain_text
    assert "+X" in plain_text


def test_show_diff_created_file_has_no_before_content() -> None:
    """Given `before=None` (the mutation created the file), when
    `show_diff` renders it, then it does not raise and the diff shows
    only additions."""
    pane = DiffPane()

    pane.show_diff("new_file.py", None, "print('hello')\n")

    plain_text = pane.content.code
    assert "+print('hello')" in plain_text
    assert "-print" not in plain_text


def test_show_diff_deleted_file_has_no_after_content() -> None:
    """Given `after=None` (the mutation deleted the file), when
    `show_diff` renders it, then it does not raise and the diff shows
    only removals."""
    pane = DiffPane()

    pane.show_diff("gone.py", "print('bye')\n", None)

    plain_text = pane.content.code
    assert "-print('bye')" in plain_text
    assert "+print" not in plain_text
