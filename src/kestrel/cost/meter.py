"""Price and accumulate token usage into USD using registry rates.

Every registry entry (:class:`~kestrel.registry.model.ModelEntry`) already
carries its own ``Decimal`` per-million-token rates; this module is the
only place that multiplies those rates against a completed turn's token
counts. All arithmetic stays in ``Decimal`` end to end -- rates parse from
TOML as ``Decimal`` and never touch a binary float, so a session total is
exactly the sum of its turns, not an accumulation of float rounding error.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry

_SIX_DP = Decimal("0.000001")
_TOKENS_PER_MTOK = Decimal(1_000_000)

# Below this fraction of billed input tokens landing as cache hits, on a
# backend that advertises cache support, is treated as a prefix-structure
# regression worth flagging rather than an expected cold-cache session.
_CACHE_ALERT_THRESHOLD: Final[Decimal] = Decimal("0.50")
# A session shorter than this never had a prior turn's prefix to hit
# against, so alerting on its ratio would false-alarm on every task's
# first turn or two rather than catch a real regression.
_CACHE_ALERT_MIN_TURNS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class TurnCost:
    """The priced result of one completed turn.

    Attributes:
        model_id: The registry id active when this turn was priced.
        input_tokens: Prompt tokens billed for this turn.
        output_tokens: Completion tokens billed for this turn.
        cached_tokens: The subset of ``input_tokens`` billed at the cache
            rate.
        usd: Total cost for this turn, quantized to six decimal places.
    """

    model_id: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    usd: Decimal


def compute_turn_cost(usage: UsageEvent, entry: ModelEntry) -> Decimal:
    """Price one turn's token usage against a registry entry's rates.

    The formula is ``((input - cached) * in_rate + cached * cached_rate +
    output * out_rate) / 1_000_000``, computed entirely in ``Decimal`` and
    quantized to six decimal places with banker's rounding
    (``ROUND_HALF_EVEN``), which is the rounding convention least biased
    across many accumulated turns.

    Raises:
        ValueError: Any token count in ``usage`` is negative, or
            ``usage.cached_tokens`` exceeds ``usage.input_tokens``. A
            backend reporting negative counts or more cached tokens than
            input tokens is malformed; pricing it anyway would silently
            produce a wrong number instead of surfacing the bad data.
    """
    if usage.input_tokens < 0 or usage.output_tokens < 0 or usage.cached_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative (got "
            f"input_tokens={usage.input_tokens}, "
            f"output_tokens={usage.output_tokens}, "
            f"cached_tokens={usage.cached_tokens})"
        )

    if usage.cached_tokens > usage.input_tokens:
        raise ValueError(
            f"cached_tokens ({usage.cached_tokens}) exceeds input_tokens "
            f"({usage.input_tokens}); refusing to price malformed usage data"
        )

    uncached_input_tokens = usage.input_tokens - usage.cached_tokens
    raw_usd = (
        Decimal(uncached_input_tokens) * entry.usd_per_mtok_input
        + Decimal(usage.cached_tokens) * entry.usd_per_mtok_cached
        + Decimal(usage.output_tokens) * entry.usd_per_mtok_output
    ) / _TOKENS_PER_MTOK
    return raw_usd.quantize(_SIX_DP, rounding=ROUND_HALF_EVEN)


class CostMeter:
    """Accumulates priced turns across one REPL session."""

    def __init__(self) -> None:
        """Start with no recorded turns."""
        self._turns: list[TurnCost] = []

    def record(self, usage: UsageEvent, entry: ModelEntry) -> TurnCost:
        """Price ``usage`` against ``entry``, append it, and return it.

        The returned :class:`TurnCost` is also retained in :attr:`turns`
        and folded into :attr:`session_usd`.
        """
        turn = TurnCost(
            model_id=entry.id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            usd=compute_turn_cost(usage, entry),
        )
        self._turns.append(turn)
        return turn

    @property
    def session_usd(self) -> Decimal:
        """Sum of every recorded turn's cost, quantized to six decimal places.

        Recomputed from :attr:`turns` on each access rather than tracked
        as separate running state, so it can never drift out of sync with
        the turn history it is supposed to summarize.
        """
        total = sum((turn.usd for turn in self._turns), start=Decimal(0))
        return total.quantize(_SIX_DP, rounding=ROUND_HALF_EVEN)

    @property
    def turns(self) -> tuple[TurnCost, ...]:
        """Every recorded turn, in the order :meth:`record` was called."""
        return tuple(self._turns)

    def cache_hit_ratio(self) -> Decimal | None:
        """``sum(cached_tokens) / sum(input_tokens)`` across every recorded
        turn, as a ``Decimal`` in ``[0, 1]``. ``None`` when zero input
        tokens have been recorded yet (an empty session, or one whose only
        turns synthesized zeroed usage) -- there is no meaningful ratio to
        report or alert on before any real turn has happened.
        """
        total_input = sum(turn.input_tokens for turn in self._turns)
        if total_input == 0:
            return None
        total_cached = sum(turn.cached_tokens for turn in self._turns)
        return Decimal(total_cached) / Decimal(total_input)

    def cache_alert(self, entry: ModelEntry) -> str | None:
        """A one-line warning naming the measured ratio (as a percentage)
        when ALL of: ``entry.supports_cache`` is ``True``, at least
        :data:`_CACHE_ALERT_MIN_TURNS` turns have been recorded, and
        :meth:`cache_hit_ratio` is below :data:`_CACHE_ALERT_THRESHOLD`.
        ``None`` in every other case -- including when
        ``entry.supports_cache`` is ``False``, since a low ratio there is
        expected rather than a regression, and when fewer than
        :data:`_CACHE_ALERT_MIN_TURNS` turns exist, since a one- or
        two-turn session (the first call never has a prior prefix to hit)
        would otherwise false-alarm on every task.
        """
        if not entry.supports_cache or len(self._turns) < _CACHE_ALERT_MIN_TURNS:
            return None
        ratio = self.cache_hit_ratio()
        if ratio is None or ratio >= _CACHE_ALERT_THRESHOLD:
            return None
        return (
            f"cache-hit ratio for {entry.id} is {ratio * 100:.0f}%, below "
            f"the {_CACHE_ALERT_THRESHOLD * 100:.0f}% alert threshold -- "
            "possible prefix-structure regression"
        )


def format_cost_line(turn: TurnCost, session_usd: Decimal) -> str:
    """Render the REPL's per-turn usage/cost line.

    Format: ``in:{input} (cached:{cached}) out:{output} · ${usd:.4f} turn
    · ${session:.4f} session`` -- the ``(cached:N)`` segment is omitted
    entirely when ``cached == 0``, since most turns have no cache hits and
    the segment would otherwise be noise. Both dollar figures round for
    *display* to four decimal places; the underlying values stay priced to
    six. A turn with zero usage (a synthesized ``UsageEvent(0, 0, 0)``,
    emitted when a backend never reports usage) renders as ``in:0 out:0 ·
    $0.0000 turn · ...`` -- visibly suspicious by design, so a missing-cost
    bug is never silent.
    """
    cached_segment = f" (cached:{turn.cached_tokens})" if turn.cached_tokens else ""
    return (
        f"in:{turn.input_tokens}{cached_segment} out:{turn.output_tokens} "
        f"· ${turn.usd:.4f} turn · ${session_usd:.4f} session"
    )
