"""Tests for `CostMeter`'s cache-hit ratio aggregate and its low-hit-rate
alert: the ratio's `None`-when-empty behavior, exact Decimal arithmetic at
and around the 50% boundary, and every gate `cache_alert` must pass before
it warns.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.cost.meter import CostMeter
from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p028, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p028_cache_alert.golden"
)


def _entry(**overrides: Any) -> ModelEntry:
    """Build a valid glm-5.2-shaped ModelEntry, overriding only the fields
    a test cares about -- most commonly `supports_cache`.
    """
    fields: dict[str, Any] = {
        "id": "glm-5.2",
        "backend": "openrouter",
        "provider_model": "zai-org/GLM-5.2",
        "api_key_env": "OPENROUTER_API_KEY",
        "context_window": 200_000,
        "max_output": 16_384,
        "usd_per_mtok_input": Decimal("0.60"),
        "usd_per_mtok_output": Decimal("2.20"),
        "usd_per_mtok_cached": Decimal("0.11"),
        "supports_tools": True,
        "supports_cache": True,
    }
    fields.update(overrides)
    return ModelEntry(**fields)


def _record(meter: CostMeter, entry: ModelEntry, usages: list[UsageEvent]) -> None:
    """Record each of `usages` against `meter`/`entry`, in order."""
    for usage in usages:
        meter.record(usage, entry)


@pytest.mark.sanity
def test_no_turns_yields_no_ratio() -> None:
    """Given a meter with no recorded turns, when the cache-hit ratio is
    read, then it is `None` -- there is nothing to divide."""
    meter = CostMeter()

    assert meter.cache_hit_ratio() is None


@pytest.mark.sanity
def test_ratio_at_exactly_fifty_percent_is_an_exact_decimal_half() -> None:
    """Given three turns whose cached/input tokens sum to exactly 50%,
    when the ratio is read, then it equals `Decimal("0.5")` exactly --
    computed in Decimal arithmetic end to end, never touching a binary
    float that could round it to something merely close to one half."""
    meter = CostMeter()
    entry = _entry()
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=25),
            UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=25),
            UsageEvent(input_tokens=0, output_tokens=5, cached_tokens=0),
        ],
    )

    assert meter.cache_hit_ratio() == Decimal("0.5")


@pytest.mark.sanity
def test_cache_alert_fires_below_threshold_with_enough_turns() -> None:
    """Given a cache-capable entry, three turns whose ratio is 20%, when
    the alert is checked, then it returns a message naming the measured
    percentage."""
    meter = CostMeter()
    entry = _entry(supports_cache=True)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=60),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
        ],
    )

    alert = meter.cache_alert(entry)

    assert alert is not None
    assert "20" in alert


def test_cache_alert_is_none_below_the_minimum_turn_count() -> None:
    """Given the same 20% ratio but only two turns recorded, when the
    alert is checked, then it is `None` -- a session this short never had
    a prior turn's prefix to hit against, so it would false-alarm on
    every task's first turn or two."""
    meter = CostMeter()
    entry = _entry(supports_cache=True)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=60),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
        ],
    )

    assert meter.cache_alert(entry) is None


def test_cache_alert_is_none_when_backend_does_not_support_cache() -> None:
    """Given the same 20% ratio and enough turns, but an entry whose
    backend does not support caching, when the alert is checked, then it
    is `None` -- a low ratio there is expected, not a regression."""
    meter = CostMeter()
    entry = _entry(supports_cache=False)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=60),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
        ],
    )

    assert meter.cache_alert(entry) is None


def test_cache_alert_is_none_exactly_at_the_boundary() -> None:
    """Given a ratio of exactly 50% (the alert threshold itself), when
    the alert is checked, then it is `None` -- the gate is `< 0.50`, not
    `<= 0.50`, so the boundary value itself never alerts."""
    meter = CostMeter()
    entry = _entry(supports_cache=True)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=25),
            UsageEvent(input_tokens=50, output_tokens=10, cached_tokens=25),
            UsageEvent(input_tokens=0, output_tokens=5, cached_tokens=0),
        ],
    )

    assert meter.cache_alert(entry) is None


def test_cache_alert_is_none_above_the_threshold() -> None:
    """Given a cache-capable entry, enough turns, and a ratio comfortably
    above 50%, when the alert is checked, then it is `None` -- a healthy
    cache-hit rate never warns."""
    meter = CostMeter()
    entry = _entry(supports_cache=True)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=80),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=80),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=80),
        ],
    )

    assert meter.cache_alert(entry) is None


def test_cache_alert_is_none_when_every_turn_billed_zero_input_tokens() -> None:
    """Given a cache-capable entry with enough turns recorded, but every
    turn synthesized zeroed usage (no real call ever billed an input
    token), when the alert is checked, then it is `None` -- `cache_alert`
    must not treat an undefined ratio as a below-threshold one."""
    meter = CostMeter()
    entry = _entry(supports_cache=True)
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0),
            UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0),
            UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0),
        ],
    )

    assert meter.cache_hit_ratio() is None
    assert meter.cache_alert(entry) is None


@pytest.mark.regression
def test_cache_alert_rendering_matches_golden_snapshot() -> None:
    """One canonical `cache_alert` rendering -- a 20% ratio on a
    cache-capable entry -- matches a pinned snapshot byte-for-byte, so an
    accidental wording or formatting change shows up here instead of
    silently drifting."""
    meter = CostMeter()
    entry = _entry()
    _record(
        meter,
        entry,
        [
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=60),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
            UsageEvent(input_tokens=100, output_tokens=10, cached_tokens=0),
        ],
    )

    alert = meter.cache_alert(entry)

    assert alert is not None
    assert alert + "\n" == _GOLDEN_FILE.read_text(encoding="utf-8")
