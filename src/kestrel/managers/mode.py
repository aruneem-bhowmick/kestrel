"""A session's live PLAN-vs-FAST interaction-mode state.

PLAN favors deliberate, higher-effort reasoning; FAST favors quicker,
more direct turns. `ModeManager` is pure in-memory state: which mode is
active, the effort level that mode maps to, and an optional model-id
override -- nothing here talks to a UI toolkit or the agent loop
itself, so it can be constructed and tested in complete isolation from
both.

No caller today reads `ModeManager.effort()` to change what effort
level an actual turn is sent at. This module only tracks the state;
wiring it into a real model call is a separate concern for whatever
builds on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from kestrel.provider.base import Effort

Mode = Literal["plan", "fast"]

_DEFAULT_EFFORT_BY_MODE: Final[Mapping[Mode, Effort]] = MappingProxyType(
    {"plan": "max", "fast": "high"}
)


@dataclass
class ModeManager:
    """A session's active mode, its effort mapping, and any model override.

    Attributes:
        mode: The active mode; `"fast"` by default.
        model_override: A registry id overriding the session's own
            default model, or `None` to use whatever the session was
            started with.
    """

    mode: Mode = "fast"
    model_override: str | None = None
    _effort_by_mode: Mapping[Mode, Effort] = field(
        default=_DEFAULT_EFFORT_BY_MODE, init=False, repr=False
    )

    def effort(self) -> Effort:
        """The `Effort` `self.mode` currently maps to."""
        return self._effort_by_mode[self.mode]

    def set_mode(self, mode: Mode) -> None:
        """Switch the active mode.

        No side effect beyond the field itself -- a caller that
        displays or otherwise reacts to the mode is responsible for
        refreshing anything derived from it.
        """
        self.mode = mode

    def set_model_override(self, model_id: str | None) -> None:
        """Set (or, with `None`, clear) the model-override id.

        `model_id` is stored as given, with no check that it names a
        real registry entry -- that validation belongs to whichever
        caller actually resolves the override into a model, not to
        this constructor-adjacent setter.
        """
        self.model_override = model_id
