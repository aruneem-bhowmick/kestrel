"""Tests for the PLAN/FAST mode manager: default state, mode switching,
its effort mapping, and model-override round-trips.
"""

from __future__ import annotations

import pytest

from kestrel.managers.mode import ModeManager

pytestmark = [pytest.mark.p035, pytest.mark.unit]


@pytest.mark.sanity
def test_default_manager_starts_in_fast_mode_with_high_effort() -> None:
    """Given a freshly constructed ModeManager, when nothing has been
    changed, then it starts in "fast" mode, maps to "high" effort, and
    carries no model override."""
    manager = ModeManager()

    assert manager.mode == "fast"
    assert manager.effort() == "high"
    assert manager.model_override is None


@pytest.mark.sanity
def test_switching_to_plan_mode_maps_to_max_effort() -> None:
    """Given a default ModeManager, when set_mode("plan") is called,
    then the mode reads back as "plan" and effort() reports "max"."""
    manager = ModeManager()

    manager.set_mode("plan")

    assert manager.mode == "plan"
    assert manager.effort() == "max"


def test_switching_back_to_fast_mode_restores_high_effort() -> None:
    """Given a ModeManager already switched to "plan", when set_mode
    is called again with "fast", then effort() reports "high" again --
    the mapping is re-derived from the current mode on every call, not
    cached from the first switch."""
    manager = ModeManager()
    manager.set_mode("plan")

    manager.set_mode("fast")

    assert manager.mode == "fast"
    assert manager.effort() == "high"
