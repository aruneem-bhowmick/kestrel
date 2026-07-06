"""Tests for turn cost computation, session accumulation, and the printed
per-turn cost line.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kestrel.cost import CostMeter, TurnCost, compute_turn_cost, format_cost_line
from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry

pytestmark = [pytest.mark.p007, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p007_cost_line.golden"
)


def _entry(**overrides: Any) -> ModelEntry:
    """Build a valid glm-5.2-shaped ModelEntry, overriding only the fields
    a test cares about. Defaults to the packaged default's own rates so
    hand-computed expectations elsewhere line up with the shipped registry.
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


@pytest.mark.sanity
def test_compute_turn_cost_matches_hand_computed_example() -> None:
    """Given 1,000,000 input tokens (250,000 of them cached) and 500,000
    output tokens at the default glm-5.2 rates, when priced, then the
    result is exactly the hand-computed total: 750k uncached input at
    $0.60/Mtok, 250k cached input at $0.11/Mtok, and 500k output at
    $2.20/Mtok."""
    usage = UsageEvent(
        input_tokens=1_000_000, output_tokens=500_000, cached_tokens=250_000
    )

    assert compute_turn_cost(usage, _entry()) == Decimal("1.577500")


@pytest.mark.sanity
def test_zero_usage_costs_nothing() -> None:
    """Given a turn with no tokens of any kind, when priced, then the cost
    is exactly zero rather than raising or dividing by zero."""
    usage = UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=0)

    assert compute_turn_cost(usage, _entry()) == Decimal("0.000000")


@pytest.mark.sanity
def test_negative_tokens_raise_value_error() -> None:
    """Given a UsageEvent with any negative token counts, when priced,
    then compute_turn_cost raises ValueError."""
    for usage in [
        UsageEvent(input_tokens=-1, output_tokens=0, cached_tokens=0),
        UsageEvent(input_tokens=0, output_tokens=-1, cached_tokens=0),
        UsageEvent(input_tokens=0, output_tokens=0, cached_tokens=-1),
    ]:
        with pytest.raises(ValueError, match="must be non-negative"):
            compute_turn_cost(usage, _entry())


@pytest.mark.sanity
def test_cached_tokens_exceeding_input_tokens_raises_value_error() -> None:
    """Given a UsageEvent reporting more cached tokens than input tokens,
    when priced, then compute_turn_cost raises ValueError instead of
    silently computing a nonsensical (negative uncached-input) cost --
    malformed provider data must be loud, not mispriced."""
    usage = UsageEvent(input_tokens=10, output_tokens=0, cached_tokens=11)

    with pytest.raises(ValueError, match="cached_tokens"):
        compute_turn_cost(usage, _entry())


def test_quantization_rounds_half_even_down_at_an_exact_tie() -> None:
    """Given a raw cost that lands exactly halfway between two six-decimal
    values whose lower neighbor has an even last digit, when priced, then
    ROUND_HALF_EVEN rounds down to that even neighbor rather than always
    rounding halfway values up."""
    entry = _entry(
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0.5"),
        usd_per_mtok_cached=Decimal("0"),
    )
    usage = UsageEvent(input_tokens=0, output_tokens=1, cached_tokens=0)

    assert compute_turn_cost(usage, entry) == Decimal("0.000000")


def test_quantization_rounds_half_even_up_at_an_exact_tie() -> None:
    """Given a raw cost that lands exactly halfway between two six-decimal
    values whose upper neighbor has an even last digit, when priced, then
    ROUND_HALF_EVEN rounds up to that even neighbor -- confirming the
    rounding mode is genuinely banker's rounding, not a one-directional
    tie-break in disguise."""
    entry = _entry(
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0.5"),
        usd_per_mtok_cached=Decimal("0"),
    )
    usage = UsageEvent(input_tokens=0, output_tokens=3, cached_tokens=0)

    assert compute_turn_cost(usage, entry) == Decimal("0.000002")


def test_meter_session_usd_equals_sum_of_recorded_turns() -> None:
    """Given three turns recorded in sequence, when the session total is
    read back, then it equals the sum of each turn's own cost -- the
    accumulator does no rounding or arithmetic beyond what compute_turn_cost
    already did per turn."""
    meter = CostMeter()
    entry = _entry()
    usages = [
        UsageEvent(1_000, 500, 0),
        UsageEvent(2_000, 750, 100),
        UsageEvent(0, 0, 0),
    ]

    for usage in usages:
        meter.record(usage, entry)

    expected_total = sum(
        (compute_turn_cost(usage, entry) for usage in usages), start=Decimal(0)
    )
    assert meter.session_usd == expected_total.quantize(Decimal("0.000001"))


def test_meter_turns_is_an_ordered_immutable_tuple() -> None:
    """Given two turns recorded in sequence, when the turn history is read
    back, then it is a tuple (not a mutable list) containing exactly the
    two TurnCost objects returned by record(), in recording order."""
    meter = CostMeter()
    entry = _entry()

    first = meter.record(UsageEvent(10, 5, 0), entry)
    second = meter.record(UsageEvent(20, 5, 0), entry)

    assert meter.turns == (first, second)
    assert isinstance(meter.turns, tuple)


@pytest.mark.sanity
def test_format_cost_line_omits_cached_segment_when_zero() -> None:
    """Given a turn with no cached tokens, when rendered, then the
    "(cached:N)" segment is omitted entirely rather than printed as
    "(cached:0)"."""
    turn = TurnCost(
        model_id="glm-5.2-zai",
        input_tokens=40,
        output_tokens=6,
        cached_tokens=0,
        usd=Decimal("0.000037"),
    )

    line = format_cost_line(turn, session_usd=Decimal("1.577537"))

    assert line == "in:40 out:6 · $0.0000 turn · $1.5775 session"


def test_format_cost_line_includes_cached_segment_when_nonzero() -> None:
    """Given a turn with cached tokens, when rendered, then the
    "(cached:N)" segment appears between the input and output counts."""
    turn = TurnCost(
        model_id="glm-5.2",
        input_tokens=1_000_000,
        output_tokens=500_000,
        cached_tokens=250_000,
        usd=Decimal("1.577500"),
    )

    line = format_cost_line(turn, session_usd=Decimal("1.577500"))

    assert line == (
        "in:1000000 (cached:250000) out:500000 · $1.5775 turn · $1.5775 session"
    )


def test_format_cost_line_zero_usage_turn_is_visibly_suspicious() -> None:
    """Given a synthesized zero-usage turn (as emitted when a backend
    never reports usage), when rendered, then the line reads "in:0 out:0
    ... $0.0000 turn ..." -- a missing-cost bug is never silent."""
    turn = TurnCost(
        model_id="glm-5.2",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        usd=Decimal("0.000000"),
    )

    line = format_cost_line(turn, session_usd=Decimal("1.577537"))

    assert line == "in:0 out:0 · $0.0000 turn · $1.5775 session"


@pytest.mark.cost_regression
def test_canonical_turn_band() -> None:
    """The canonical mock turn (42 input / 7 output / 0 cached tokens, at
    the packaged glm-5.2 default rates) must cost exactly
    Decimal("0.000041"); any drift in the pricing formula, the default
    rates, or the rounding mode fails this test rather than surfacing
    later as an unexplained cost-band violation."""
    usage = UsageEvent(input_tokens=42, output_tokens=7, cached_tokens=0)

    assert compute_turn_cost(usage, _entry()) == Decimal("0.000041")


@pytest.mark.regression
@pytest.mark.acceptance
def test_cost_line_format_matches_golden_snapshot() -> None:
    """The exact per-turn line the REPL prints, for four canonical shapes
    (a cache hit, no cache hit, a zero-usage turn, and a large turn),
    must match a pinned snapshot byte-for-byte -- this text is the DoD's
    "prints usage/cost per turn" clause made concrete, and an accidental
    formatting change should surface here rather than downstream."""
    meter = CostMeter()
    entry = _entry()
    entry_zai = _entry(id="glm-5.2-zai")

    lines = [
        format_cost_line(
            meter.record(UsageEvent(1_000_000, 500_000, 250_000), entry),
            meter.session_usd,
        ),
        format_cost_line(
            meter.record(UsageEvent(40, 6, 0), entry_zai), meter.session_usd
        ),
        format_cost_line(meter.record(UsageEvent(0, 0, 0), entry), meter.session_usd),
        format_cost_line(
            meter.record(UsageEvent(50_000_000, 16_384, 10_000_000), entry),
            meter.session_usd,
        ),
    ]

    assert "\n".join(lines) + "\n" == _GOLDEN_FILE.read_text(encoding="utf-8")


_RATE = st.decimals(
    min_value=0, max_value=1000, places=6, allow_nan=False, allow_infinity=False
)


@pytest.mark.api
@given(
    input_tokens=st.integers(min_value=0, max_value=10**9),
    output_tokens=st.integers(min_value=0, max_value=10**9),
    cached_fraction=st.floats(min_value=0.0, max_value=1.0),
    in_rate=_RATE,
    out_rate=_RATE,
    cached_rate=_RATE,
)
def test_cost_is_non_negative_and_monotonic_in_input_and_output(
    input_tokens: int,
    output_tokens: int,
    cached_fraction: float,
    in_rate: Decimal,
    out_rate: Decimal,
    cached_rate: Decimal,
) -> None:
    """Given any non-negative token counts (cached kept <= input) and any
    non-negative rates, when priced, then the cost is never negative, and
    it never decreases when either input_tokens or output_tokens grows
    with everything else held fixed.

    Growing cached_tokens alone is deliberately not asserted here: unlike
    input and output tokens, its rate multiplies a *shifted* share of the
    input count rather than adding a new one, so whether cost rises or
    falls with cached_tokens depends on whether the cached rate is above
    or below the input rate. Registry validation only warns (rather than
    rejects) when the cached rate exceeds the input rate, so no
    monotonicity direction holds unconditionally for that field.
    """
    entry = _entry(
        usd_per_mtok_input=in_rate,
        usd_per_mtok_output=out_rate,
        usd_per_mtok_cached=cached_rate,
    )
    cached_tokens = int(input_tokens * cached_fraction)

    baseline = compute_turn_cost(
        UsageEvent(input_tokens, output_tokens, cached_tokens), entry
    )
    assert baseline >= 0

    grown_output = compute_turn_cost(
        UsageEvent(input_tokens, output_tokens + 1, cached_tokens), entry
    )
    assert grown_output >= baseline

    grown_input = compute_turn_cost(
        UsageEvent(input_tokens + 1, output_tokens, cached_tokens), entry
    )
    assert grown_input >= baseline
