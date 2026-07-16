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


def test_set_model_override_then_clear_round_trips() -> None:
    """Given a default ModeManager, when set_model_override sets an id
    and a later call clears it with None, then model_override reflects
    each change in turn without disturbing mode or effort."""
    manager = ModeManager()

    manager.set_model_override("glm-5.2")
    assert manager.model_override == "glm-5.2"

    manager.set_model_override(None)
    assert manager.model_override is None
    assert manager.mode == "fast"
    assert manager.effort() == "high"


def test_effort_by_mode_mapping_is_immutable() -> None:
    """Given a ModeManager's own internal effort mapping, when a caller
    attempts to mutate it directly, then the assignment raises TypeError
    -- the mapping is a MappingProxyType, not a plain dict, so it cannot
    be tampered with through any public or private surface."""
    manager = ModeManager()

    with pytest.raises(TypeError):
        manager._effort_by_mode["fast"] = "max"  # type: ignore[index]


def test_two_managers_do_not_share_mutable_mode_state() -> None:
    """Given two independently constructed ModeManagers, when one has
    its mode switched, then the other is unaffected -- mode is per-
    instance state, not shared through the default effort mapping."""
    first = ModeManager()
    second = ModeManager()

    first.set_mode("plan")

    assert first.mode == "plan"
    assert second.mode == "fast"
    assert second.effort() == "high"


def test_effort_by_mode_is_not_a_constructor_argument() -> None:
    """Given ModeManager's constructor, when called with an
    _effort_by_mode keyword argument, then it raises TypeError -- the
    field is excluded from __init__ so a caller cannot hand every
    instance a mutable, per-instance mapping in place of the shared
    default."""
    with pytest.raises(TypeError):
        ModeManager(_effort_by_mode={"plan": "high", "fast": "high"})  # type: ignore[call-arg]


def test_repr_omits_the_internal_effort_mapping() -> None:
    """Given a default ModeManager, when it is repr'd, then the output
    names its public fields but not the internal effort mapping -- an
    implementation detail, not part of the object's displayed identity."""
    manager = ModeManager()

    assert "_effort_by_mode" not in repr(manager)
    assert "mode=" in repr(manager)
