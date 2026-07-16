"""Renders the TUI status bar's one-line summary from a plain snapshot.

`StatusSnapshot` is a frozen value the caller builds fresh on every
update; it carries no behavior of its own and reads no state directly
from `CostMeter`, `ModeManager`, or `BudgetManager` -- the caller
already has those numbers and hands them over verbatim.
`render_status_line` is a pure formatting function over that value,
with no Textual dependency, so both can be unit-tested without
mounting a widget at all. `kestrel.tui.app.StatusBar.show` is the only
place `render_status_line` is called from inside a live widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kestrel.managers.mode import Mode
from kestrel.provider.base import Effort

_PCT_MULTIPLIER = Decimal(100)


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Everything StatusBar needs to render one line. Built fresh by
    the caller on every update -- StatusBar itself holds no state
    beyond the last snapshot it was shown.

    Attributes:
        model_id: The registry id the next task/turn will be sent to.
        mode: The active ModeManager mode.
        effort: The active ModeManager effort() result for `mode`.
        context_used_tokens: The most recently billed turn's own
            input_tokens (the same signal `should_compact` reads), or
            None before any turn has billed.
        context_window: The active model's own context_window.
        session_usd: The current session's running CostMeter total.
        session_cap_usd: The configured session budget cap, or None.
        day_usd: Historical-plus-session day spend.
        day_cap_usd: The configured day budget cap, or None.
    """

    model_id: str
    mode: Mode
    effort: Effort
    context_used_tokens: int | None
    context_window: int
    session_usd: Decimal
    session_cap_usd: Decimal | None
    day_usd: Decimal
    day_cap_usd: Decimal | None


def _format_cap_suffix(cap_usd: Decimal | None) -> str:
    """Render a spend segment's cap suffix: ` / cap $X.XXXX` when
    `cap_usd` is set, or an empty string when it is `None` -- matching
    `CostMeter`/`BudgetManager`'s own "None means no cap" convention.
    """
    if cap_usd is None:
        return ""
    return f" / cap ${cap_usd:.4f}"


def render_status_line(snapshot: StatusSnapshot) -> str:
    """Render one status-bar line:
    `{model_id} · {mode}/{effort} · ctx {pct}% ({used}/{window}) ·
    session ${session_usd:.4f}{cap} · day ${day_usd:.4f}{cap}`.
    `{pct}`/`{used}` render as `--` when `context_used_tokens is None`
    (no turn has billed yet). Each `{cap}` segment is
    ` / cap $X.XXXX` when that scope's own cap is set, and empty
    (bare `$X.XXXX`) when it is `None` -- matching CostMeter/
    BudgetManager's own "None means no cap" convention throughout
    this codebase.
    """
    if snapshot.context_used_tokens is None:
        ctx_pct = "--"
        ctx_used = "--"
    else:
        pct = (
            Decimal(snapshot.context_used_tokens)
            / Decimal(snapshot.context_window)
            * _PCT_MULTIPLIER
        )
        ctx_pct = f"{pct:.0f}"
        ctx_used = str(snapshot.context_used_tokens)

    session_cap = _format_cap_suffix(snapshot.session_cap_usd)
    day_cap = _format_cap_suffix(snapshot.day_cap_usd)

    return (
        f"{snapshot.model_id} · {snapshot.mode}/{snapshot.effort} · "
        f"ctx {ctx_pct}% ({ctx_used}/{snapshot.context_window}) · "
        f"session ${snapshot.session_usd:.4f}{session_cap} · "
        f"day ${snapshot.day_usd:.4f}{day_cap}"
    )
