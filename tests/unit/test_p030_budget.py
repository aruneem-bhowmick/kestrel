"""Tests for the budget-cap classifier: session/day/month USD caps with
soft (warn/degrade) and hard (halt) thresholds.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kestrel.managers.budget import (
    BudgetLimits,
    BudgetManager,
    BudgetStatus,
)

pytestmark = [pytest.mark.p030, pytest.mark.unit]

_STATUS_RANK = {BudgetStatus.OK: 0, BudgetStatus.SOFT: 1, BudgetStatus.HARD: 2}


@pytest.mark.sanity
def test_no_caps_configured_always_ok() -> None:
    """Given a BudgetManager with every cap left unset, when checked
    against any spend, then the status is always OK and no cap is
    named, regardless of how large the spend figures are."""
    manager = BudgetManager()

    event = manager.check(
        spent_session=Decimal("1000000"),
        spent_day=Decimal("1000000"),
        spent_month=Decimal("1000000"),
    )

    assert event.status == BudgetStatus.OK
    assert event.tripped_cap is None


@pytest.mark.sanity
def test_spend_below_soft_threshold_is_ok() -> None:
    """Given a session cap of $10 and spend of $5 (below the default 80%
    soft threshold), when checked, then the status is OK."""
    manager = BudgetManager(limits=BudgetLimits(session_usd=Decimal("10")))

    event = manager.check(
        spent_session=Decimal("5"), spent_day=Decimal("0"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.OK
    assert event.tripped_cap is None


@pytest.mark.sanity
def test_spend_at_soft_threshold_trips_soft() -> None:
    """Given a session cap of $10 and spend of exactly $8 (80% of the
    cap, the default soft_threshold), when checked, then the status is
    SOFT and the session cap is named -- the boundary itself trips."""
    manager = BudgetManager(limits=BudgetLimits(session_usd=Decimal("10")))

    event = manager.check(
        spent_session=Decimal("8"), spent_day=Decimal("0"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.SOFT
    assert event.tripped_cap == "session"


@pytest.mark.sanity
def test_spend_at_cap_trips_hard() -> None:
    """Given a session cap of $10 and spend of exactly $10, when checked,
    then the status is HARD -- reaching the cap itself, not just
    exceeding it, already trips."""
    manager = BudgetManager(limits=BudgetLimits(session_usd=Decimal("10")))

    event = manager.check(
        spent_session=Decimal("10"), spent_day=Decimal("0"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.HARD
    assert event.tripped_cap == "session"


@pytest.mark.sanity
def test_spend_just_below_cap_is_soft_not_hard() -> None:
    """Given a session cap of $10 and spend of $9.99, when checked, then
    the status is SOFT, not HARD -- the strict >= boundary at the cap
    itself is tested precisely, not just its rounded neighborhood."""
    manager = BudgetManager(limits=BudgetLimits(session_usd=Decimal("10")))

    event = manager.check(
        spent_session=Decimal("9.99"),
        spent_day=Decimal("0"),
        spent_month=Decimal("0"),
    )

    assert event.status == BudgetStatus.SOFT
    assert event.tripped_cap == "session"


def test_day_cap_trips_independently_of_an_ok_session_cap() -> None:
    """Given a session cap that is OK and a day cap that is HARD, when
    checked, then the worst status wins and the day cap is named --
    each cap is classified independently before the worst is picked."""
    manager = BudgetManager(
        limits=BudgetLimits(session_usd=Decimal("100"), day_usd=Decimal("10"))
    )

    event = manager.check(
        spent_session=Decimal("1"), spent_day=Decimal("10"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.HARD
    assert event.tripped_cap == "day"


def test_tied_hard_caps_name_session_by_priority_order() -> None:
    """Given both the session and day caps tied at HARD, when checked,
    then tripped_cap names "session" -- the first-priority cap in
    session/day/month order, not whichever happened to be classified
    last."""
    manager = BudgetManager(
        limits=BudgetLimits(session_usd=Decimal("10"), day_usd=Decimal("10"))
    )

    event = manager.check(
        spent_session=Decimal("10"), spent_day=Decimal("10"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.HARD
    assert event.tripped_cap == "session"


def test_tied_hard_caps_name_day_over_month() -> None:
    """Given day and month caps tied at HARD with the session cap OK,
    when checked, then tripped_cap names "day" -- the priority order
    holds for every pair of caps, not just session-versus-day."""
    manager = BudgetManager(
        limits=BudgetLimits(
            session_usd=Decimal("100"), day_usd=Decimal("10"), month_usd=Decimal("10")
        )
    )

    event = manager.check(
        spent_session=Decimal("1"), spent_day=Decimal("10"), spent_month=Decimal("10")
    )

    assert event.status == BudgetStatus.HARD
    assert event.tripped_cap == "day"


def test_custom_soft_threshold_shifts_where_soft_trips() -> None:
    """Given a session cap of $10 and a custom soft_threshold of 0.5,
    when checked against spend of $5, then the status is SOFT -- the
    configured threshold fraction, not a hardcoded 80%, decides the
    boundary."""
    manager = BudgetManager(
        limits=BudgetLimits(session_usd=Decimal("10"), soft_threshold=Decimal("0.5"))
    )

    event = manager.check(
        spent_session=Decimal("5"), spent_day=Decimal("0"), spent_month=Decimal("0")
    )

    assert event.status == BudgetStatus.SOFT
    assert event.tripped_cap == "session"


def test_six_decimal_precision_compares_exactly() -> None:
    """Given a cap and spend both expressed to six decimal places that
    differ by exactly one unit in the last place, when checked, then
    the classification reflects that exact difference rather than
    being swallowed by float rounding."""
    manager = BudgetManager(limits=BudgetLimits(session_usd=Decimal("10.000000")))

    just_under = manager.check(
        spent_session=Decimal("9.999999"),
        spent_day=Decimal("0"),
        spent_month=Decimal("0"),
    )
    at_cap = manager.check(
        spent_session=Decimal("10.000000"),
        spent_day=Decimal("0"),
        spent_month=Decimal("0"),
    )

    assert just_under.status == BudgetStatus.SOFT
    assert at_cap.status == BudgetStatus.HARD


_SPEND = st.decimals(
    min_value=0, max_value=1000, places=6, allow_nan=False, allow_infinity=False
)


@given(
    spent_session=_SPEND,
    spent_day=_SPEND,
    spent_month=_SPEND,
    delta=_SPEND,
)
def test_status_is_monotonically_non_decreasing_in_each_spend_argument(
    spent_session: Decimal,
    spent_day: Decimal,
    spent_month: Decimal,
    delta: Decimal,
) -> None:
    """Given any Decimal spend figures against fixed caps, when one
    spend argument grows by a non-negative delta with the other two
    held fixed, then the resulting status never decreases -- OK <=
    SOFT <= HARD as an ordering -- in each of the three spend
    arguments independently. This guards against float-style drift or
    an inverted comparison sneaking into the classifier."""
    manager = BudgetManager(
        limits=BudgetLimits(
            session_usd=Decimal("10"), day_usd=Decimal("20"), month_usd=Decimal("50")
        )
    )

    baseline = manager.check(
        spent_session=spent_session, spent_day=spent_day, spent_month=spent_month
    )

    grown_session = manager.check(
        spent_session=spent_session + delta,
        spent_day=spent_day,
        spent_month=spent_month,
    )
    assert _STATUS_RANK[grown_session.status] >= _STATUS_RANK[baseline.status]

    grown_day = manager.check(
        spent_session=spent_session,
        spent_day=spent_day + delta,
        spent_month=spent_month,
    )
    assert _STATUS_RANK[grown_day.status] >= _STATUS_RANK[baseline.status]

    grown_month = manager.check(
        spent_session=spent_session,
        spent_day=spent_day,
        spent_month=spent_month + delta,
    )
    assert _STATUS_RANK[grown_month.status] >= _STATUS_RANK[baseline.status]
