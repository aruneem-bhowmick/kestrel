"""Gates destructive tool actions behind interactive approval.

`ApprovalManager` classifies a proposed action by `DestructiveKind`
(deleting files, force-pushing, changing file permissions, turning on
network access, or writing outside the repository) and, for anything
not already allowlisted or approved earlier in the session, asks an
injectable decision function whether to proceed. The default decision
function, `_prompt_stdin`, prompts on the real terminal; tests inject a
scripted one instead, following the same `input_fn`-style dependency
injection `kestrel.repl` already uses for its own read-eval-print loop.

A decision of `"once"` allows exactly the request that triggered it and
nothing more -- an identical follow-up request prompts again. `"always"`
additionally remembers the request's kind as approved for the rest of
this manager's lifetime, so nothing of that kind prompts again through
the same instance. Anything else, including `"deny"`, raises
`ApprovalDenied`, which a caller invoking `check` directly is
responsible for translating into its own normal refusal rather than
letting it escape unhandled.

`network_on` and `out_of_repo_write` are declared kinds with no live
caller wiring them up yet: `kestrel.tools.execute`'s sandbox always
runs with networking disabled, and `kestrel.tools.edit_file` refuses an
out-of-repo write outright rather than offering it as an approvable
escalation. Both are recognized here regardless, so a future caller
that does wire up either path can name it without inventing a parallel
vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

DestructiveKind = Literal[
    "delete", "force_push", "chmod", "network_on", "out_of_repo_write"
]
ApprovalDecision = Literal["once", "always", "deny"]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One proposed destructive action awaiting a decision.

    Attributes:
        kind: Which `DestructiveKind` this request belongs to.
        summary: A one-line, human-readable description naming the
            exact action being proposed.
        detail: The exact command or diff the action would run --
            everything a reviewer needs to make an informed decision,
            verbatim.
    """

    kind: DestructiveKind
    summary: str
    detail: str


class ApprovalDenied(Exception):
    """Raised by `ApprovalManager.check` when a request is denied.

    `str(self)` is the request's own `summary`. A caller that invokes
    `check` directly is responsible for catching this and surfacing it
    as an ordinary refusal in its own idiom, rather than letting it
    escape as an unhandled exception.
    """


def _prompt_stdin(
    request: ApprovalRequest, *, input_fn: Callable[[str], str] = input
) -> ApprovalDecision:
    """Render `request` on the real terminal and read one line of reply.

    Prints `request.summary` and `request.detail`, then reads a single
    line via `input_fn` (defaulting to the built-in `input`). The reply
    is matched case-insensitively: `y`/`yes` decides `"once"`, `always`
    decides `"always"`, and everything else -- including an empty line
    -- decides `"deny"`. This is a plain-terminal placeholder for
    interactive approval, not a real modal UI.
    """
    print(request.summary)
    print(request.detail)
    reply = input_fn("Approve? [y]es / [a]lways / [N]o: ").strip().lower()
    if reply in ("y", "yes"):
        return "once"
    if reply in ("a", "always"):
        return "always"
    return "deny"


class ApprovalManager:
    """Classifies destructive actions and gates them behind approval.

    Every request this manager is asked to `check` is either
    short-circuited (already allowlisted, or already approved
    `"always"` earlier this session) or handed to `decide_fn` for a
    fresh decision. Nothing here classifies an action itself -- that is
    each caller's own job (see `kestrel.tools.execute`'s pattern-table
    classification of shell commands); this manager only ever sees the
    `ApprovalRequest` a caller has already built.
    """

    def __init__(
        self,
        *,
        allowlist: frozenset[DestructiveKind] = frozenset(),
        decide_fn: Callable[[ApprovalRequest], ApprovalDecision] = _prompt_stdin,
    ) -> None:
        """`allowlist` is the set of kinds pre-approved for every
        request this session; `decide_fn` is called only for kinds not
        in `allowlist` and not already session-approved. Defaults to a
        real stdin y/n/always prompt rendering `request.summary` and
        `request.detail`.
        """
        self._allowlist = allowlist
        self._decide_fn = decide_fn
        self._session_approved: set[DestructiveKind] = set()

    def check(self, request: ApprovalRequest) -> None:
        """No-op if `request.kind` is already allowlisted or
        session-approved. Otherwise calls `decide_fn(request)`: `"once"`
        returns without recording anything; `"always"` records the kind
        as session-approved (so a later request of the same kind
        short-circuits without prompting again) and returns; anything
        else -- including `"deny"` -- raises
        `ApprovalDenied(request.summary)`.
        """
        if request.kind in self._allowlist or request.kind in self._session_approved:
            return
        decision = self._decide_fn(request)
        if decision == "once":
            return
        if decision == "always":
            self._session_approved.add(request.kind)
            return
        raise ApprovalDenied(request.summary)
