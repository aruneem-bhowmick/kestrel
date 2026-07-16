"""Tests for the status bar's pure-rendering StatusSnapshot/render_status_line
pair, and the StatusBar widget's own show() hook over that rendering.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.tui.app import KestrelApp, StatusBar
from kestrel.tui.status import StatusSnapshot, render_status_line

pytestmark = [pytest.mark.p037, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p037_status_line.golden"
)


def _snapshot(**overrides: Any) -> StatusSnapshot:
    """Build a fully populated StatusSnapshot, overriding only the
    fields a test cares about. Defaults to a mid-session shape (some
    context used, no caps set) so each test only states what it's
    actually varying.
    """
    fields: dict[str, Any] = {
        "model_id": "glm-5.2",
        "mode": "fast",
        "effort": "high",
        "context_used_tokens": 50_000,
        "context_window": 200_000,
        "session_usd": Decimal("1.2345"),
        "session_cap_usd": None,
        "day_usd": Decimal("12.3456"),
        "day_cap_usd": None,
    }
    fields.update(overrides)
    return StatusSnapshot(**fields)


@pytest.mark.sanity
def test_no_caps_render_bare_dollar_segments() -> None:
    """Given a snapshot with no session or day cap configured, when it
    is rendered, then both spend segments show a bare `$X.XXXX` with no
    trailing `/ cap` text."""
    line = render_status_line(_snapshot(session_cap_usd=None, day_cap_usd=None))

    assert "session $1.2345 ·" in line
    assert "day $12.3456" in line
    assert "/ cap" not in line


@pytest.mark.sanity
def test_both_caps_render_exact_cap_suffix() -> None:
    """Given a snapshot with both a session and a day cap configured,
    when it is rendered, then each spend segment carries its own
    ` / cap $Y.YYYY` suffix, exactly."""
    line = render_status_line(
        _snapshot(
            session_cap_usd=Decimal("5.0000"),
            day_cap_usd=Decimal("50.0000"),
        )
    )

    assert "session $1.2345 / cap $5.0000 ·" in line
    assert "day $12.3456 / cap $50.0000" in line


@pytest.mark.sanity
def test_no_billed_turn_renders_placeholder_context_usage() -> None:
    """Given a snapshot whose context_used_tokens is None (no turn has
    billed yet), when it is rendered, then the context segment reads
    `ctx --% (--/200000)` rather than dividing by anything."""
    line = render_status_line(
        _snapshot(context_used_tokens=None, context_window=200_000)
    )

    assert "ctx --% (--/200000)" in line


@pytest.mark.sanity
def test_billed_turn_renders_exact_context_percentage() -> None:
    """Given context_used_tokens=50_000 against a context_window of
    200_000, when rendered, then the context segment reads exactly
    `ctx 25% (50000/200000)`."""
    line = render_status_line(
        _snapshot(context_used_tokens=50_000, context_window=200_000)
    )

    assert "ctx 25% (50000/200000)" in line


@pytest.mark.sanity
def test_zero_context_window_renders_placeholder_instead_of_dividing() -> None:
    """Given a billed turn but a non-positive context_window (a
    caller-supplied value the registry itself would never produce),
    when rendered, then the context segment falls back to the same
    `--` placeholders used when no turn has billed, rather than
    raising a division-by-zero error."""
    line = render_status_line(_snapshot(context_used_tokens=50_000, context_window=0))

    assert "ctx --% (--/0)" in line


@pytest.mark.sanity
def test_dollar_formatting_matches_cost_meter_convention() -> None:
    """Given Decimal spend figures with more than four decimal places
    of underlying precision, when rendered, then both dollar segments
    round for display to exactly four decimal places -- the same
    `:.4f` convention `format_cost_line` uses, and the same rounding."""
    line = render_status_line(
        _snapshot(
            session_usd=Decimal("1.23455"),
            session_cap_usd=None,
            day_usd=Decimal("12.34565"),
            day_cap_usd=None,
        )
    )

    assert f"session ${Decimal('1.23455'):.4f}" in line
    assert f"day ${Decimal('12.34565'):.4f}" in line


@pytest.mark.sanity
def test_mode_and_effort_render_verbatim() -> None:
    """Given mode/effort pairs from both PLAN/max and FAST/high, when
    rendered, then each renders verbatim as `{mode}/{effort}` with no
    reformatting."""
    plan_line = render_status_line(_snapshot(mode="plan", effort="max"))
    fast_line = render_status_line(_snapshot(mode="fast", effort="high"))

    assert "plan/max" in plan_line
    assert "fast/high" in fast_line


@pytest.mark.regression
def test_status_line_matches_golden_snapshot() -> None:
    """One canonical render_status_line rendering -- both caps set, a
    real context percentage -- matches a pinned snapshot byte-for-byte,
    so an accidental wording or formatting change shows up here instead
    of silently drifting."""
    snapshot = _snapshot(
        model_id="glm-5.2",
        mode="plan",
        effort="max",
        context_used_tokens=50_000,
        context_window=200_000,
        session_usd=Decimal("1.2345"),
        session_cap_usd=Decimal("5.0000"),
        day_usd=Decimal("12.3456"),
        day_cap_usd=Decimal("50.0000"),
    )

    line = render_status_line(snapshot)

    assert line + "\n" == _GOLDEN_FILE.read_text(encoding="utf-8")


@pytest.mark.ui
async def test_status_bar_show_renders_via_render_status_line() -> None:
    """Given a mounted KestrelApp, when StatusBar.show() is called with
    a hand-built StatusSnapshot, then the widget's own rendered content
    matches render_status_line's output for that same snapshot exactly."""
    sample = _snapshot(mode="plan", effort="max")

    async with KestrelApp().run_test() as pilot:
        status_bar = pilot.app.query_one("#status_bar", StatusBar)
        status_bar.show(sample)

        assert str(status_bar.render()) == render_status_line(sample)
