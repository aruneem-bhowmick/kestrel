"""Classify spend against per-scope USD budget caps.

`BudgetManager` is a pure classifier: given how much has been spent this
session, day, and month, it says whether each cap sits under budget, past
a soft (warn/degrade) threshold, or past a hard (halt) one. It holds no
opinion about where those spend figures came from -- it never reads a
file or tracks spend itself, mirroring `kestrel.cost.CostMeter`'s own
precedent of staying decoupled from persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final, Literal


class BudgetStatus(StrEnum):
    """Where one checked spend figure sits relative to its configured cap."""

    OK = "ok"
    SOFT = "soft"
    HARD = "hard"


_STATUS_RANK: Final[dict[BudgetStatus, int]] = {
    BudgetStatus.OK: 0,
    BudgetStatus.SOFT: 1,
    BudgetStatus.HARD: 2,
}


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """USD caps. `None` means "no cap" for that scope -- never trips.

    Attributes:
        session_usd: Cap on this task/session's own spend.
        day_usd: Cap on spend across the current UTC day.
        month_usd: Cap on spend across the current UTC month.
        soft_threshold: Fraction of a cap counted as the SOFT
            (warn/degrade) boundary, e.g. 0.8 means SOFT trips at 80%
            of whichever cap it belongs to.
    """

    session_usd: Decimal | None = None
    day_usd: Decimal | None = None
    month_usd: Decimal | None = None
    soft_threshold: Decimal = Decimal("0.8")


@dataclass(frozen=True, slots=True)
class BudgetEvent:
    """The result of one `BudgetManager.check` call.

    Attributes:
        status: The worst status across all three configured caps.
        tripped_cap: Which cap ("session", "day", or "month") produced
            `status`; `None` when `status` is OK. When more than one cap
            ties at the worst status, names whichever comes first in
            session/day/month priority order.
        spent_session: The session spend this check was given.
        spent_day: The day spend this check was given.
        spent_month: The month spend this check was given.
    """

    status: BudgetStatus
    tripped_cap: Literal["session", "day", "month"] | None
    spent_session: Decimal
    spent_day: Decimal
    spent_month: Decimal


class BudgetManager:
    """Classifies spend against `BudgetLimits`; holds no state of its own
    beyond the limits it was constructed with."""

    def __init__(self, *, limits: BudgetLimits = BudgetLimits()) -> None:
        """Store `limits` as the caps every `check` call classifies spend
        against. Defaults to `BudgetLimits()`, under which every cap is
        unset and every check therefore returns OK.
        """
        self._limits = limits

    def check(
        self, *, spent_session: Decimal, spent_day: Decimal, spent_month: Decimal
    ) -> BudgetEvent:
        """For each configured cap, HARD when spent >= cap; SOFT when
        spent >= cap * soft_threshold; OK otherwise (an unconfigured,
        `None` cap is always OK and never named as `tripped_cap`).
        Returns a `BudgetEvent` naming the worst status (HARD > SOFT >
        OK) and, on a tie, the first-priority (session, then day, then
        month) cap that reached it.
        """
        session_status = self._classify(spent_session, self._limits.session_usd)
        day_status = self._classify(spent_day, self._limits.day_usd)
        month_status = self._classify(spent_month, self._limits.month_usd)
        worst_status = max(
            session_status, day_status, month_status, key=_STATUS_RANK.__getitem__
        )

        tripped_cap: Literal["session", "day", "month"] | None = None
        if worst_status != BudgetStatus.OK:
            if session_status == worst_status:
                tripped_cap = "session"
            elif day_status == worst_status:
                tripped_cap = "day"
            else:
                tripped_cap = "month"

        return BudgetEvent(
            status=worst_status,
            tripped_cap=tripped_cap,
            spent_session=spent_session,
            spent_day=spent_day,
            spent_month=spent_month,
        )

    def _classify(self, spent: Decimal, cap: Decimal | None) -> BudgetStatus:
        """Classify one cap in isolation: HARD once `spent` reaches `cap`,
        SOFT once it reaches `cap * soft_threshold`, OK otherwise. Always
        OK when `cap` is `None` -- an unconfigured cap never trips.
        """
        if cap is None:
            return BudgetStatus.OK
        if spent >= cap:
            return BudgetStatus.HARD
        if spent >= cap * self._limits.soft_threshold:
            return BudgetStatus.SOFT
        return BudgetStatus.OK
