"""Budget-capped live smoke test proving a real cache-capable backend
actually reuses its cached compute across two calls that share an
identical, byte-stable leading prefix -- the live counterpart to the
mock-backed cache-hit-ratio scenario in
``tests/acceptance/test_p033_dod_phase_2.py``, and this project's own
live evidence for its cost-accounting requirement (a real turn's
computed cost is a small, plausible positive number, cross-checked the
only way possible without an independent invoice API to diff against).

This is the one place in the suite that drives the provider client
directly (rather than the tool-calling agent loop or the REPL) against a
real endpoint, so it can send the exact same two requests back to back
without depending on a live model's own turn-taking to produce them.
Spending discipline mirrors the rest of the live suite
(``tests/e2e/test_p005_live_openrouter.py``,
``tests/e2e/test_p023_dod_live.py``): both calls together are asserted
well under the project's $0.50-per-run policy ceiling.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal

import pytest

from kestrel.agent.loop import _SYSTEM_PROMPT
from kestrel.cost.meter import CostMeter
from kestrel.kestrel_md import KestrelMd, VerifyCommands
from kestrel.provider.base import Message
from kestrel.provider.cache import build_stable_prefix, mark_cache_breakpoints
from kestrel.provider.events import UsageEvent
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry

pytestmark = [
    pytest.mark.p033,
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.dod_phase_2,
]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_MODEL_ID = "glm-5.2"
_COLLECTION_TIMEOUT_S = 30.0
# Well under the $0.50/run policy ceiling described in the module docstring.
_BUDGET_CEILING_USD = Decimal("0.10")

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1 and {_API_KEY_ENV} to run the live cache-hit smoke test"
)

# A cache-capable backend generally only activates automatic prompt caching
# once the shared prefix crosses its own, provider-specific minimum size --
# a short one-line KESTREL.md would risk never crossing that floor and
# reporting a false-negative cache miss. This is deliberately long and
# realistic-looking (the kind of conventions document a real repo would
# actually carry) rather than padding, so the two live calls below exercise
# the exact code path a real task's stable prefix goes through.
_LIVE_KESTREL_MD_TEXT = """\
# Project conventions

This repository follows a small set of house conventions that every
change, human- or agent-authored, is expected to honor. They exist to
keep review fast and behavior predictable across a codebase with many
independent contributors working on overlapping modules.

## Code style

Prefer small, single-purpose functions over large ones that branch on
many unrelated conditions. A function that needs a paragraph of comments
to explain what it does is usually a function that should be split into
two or three smaller ones with names that make the comment unnecessary.
Keep line length under 100 characters. Avoid deeply nested conditionals;
prefer early returns over an `if`/`else` ladder more than two levels
deep. Name booleans as predicates (`is_ready`, `has_children`) rather
than nouns, so a call site reads like a sentence.

Type every public function signature. Internal helper functions should
also be typed unless the inference is completely unambiguous from
context. Avoid `Any` except at true external boundaries (deserializing
untrusted JSON, a third-party library with no stubs) -- and even then,
narrow it to a concrete type as soon as possible after the boundary.

## Testing

Every new function gets at least one direct unit test and, where it
touches a real external seam (a filesystem, a subprocess, a network
call), at least one integration-level test that exercises that seam for
real rather than through a mock. Tests should read as documentation:
given some setup, when some action happens, then some specific,
checkable outcome follows. Avoid asserting on incidental details (exact
log wording, internal field ordering) that would make the test brittle
to a refactor that doesn't change actual behavior.

Prefer many small, focused test functions over one large test that
walks through an entire scenario end to end, unless the scenario itself
genuinely requires that sequencing to be meaningful (e.g. testing that a
resumed task picks up exactly where a halted one left off).

## Error handling

Raise a specific, named exception type at the point something goes
wrong, with a message that names the offending value and what was
expected instead. Never swallow an exception silently -- if a caller
genuinely doesn't care about a particular failure mode, that should be
an explicit, commented decision, not an empty `except` block. Prefer
returning a typed result object over raising for an outcome that is a
normal, expected part of a function's contract (a lookup that legitimately
found nothing, a validation that legitimately failed) -- reserve raising
for a caller-side contract violation or an environment failure.

## Commit and review discipline

Keep commits small and focused on one logical change each. A commit
message should explain *why* a change was made, not just restate what
changed -- the diff already shows what changed. Group related test
additions with the implementation they cover rather than deferring all
tests to one giant commit at the end of a body of work.

## Architecture notes

This project is organized around a small number of narrow interfaces
that most of the code depends on, with the concrete implementations
behind them kept swappable. New functionality should extend an existing
interface where one already fits, rather than reaching around it to call
a concrete implementation directly. When a new capability genuinely
doesn't fit any existing seam, introduce a new one deliberately, with a
docstring explaining the boundary it draws and why.

Keep expensive or side-effecting work (network calls, subprocess
invocations, disk writes) behind explicit, injectable collaborators
rather than reached for as global state or module-level singletons --
this is what keeps the test suite hermetic and fast without sacrificing
real integration coverage where it matters.
"""


@pytest.mark.skipif(
    os.environ.get(_LIVE_TESTS_ENV) != "1" or not os.environ.get(_API_KEY_ENV),
    reason=_SKIP_REASON,
)
async def test_dod_live_second_call_reports_a_real_cache_hit() -> None:
    """Given the packaged default registry -- no override, the same
    defaults a fresh install resolves to -- and a real OpenRouter
    credential, when the exact same byte-stable prefix (a real,
    substantial KESTREL.md folded into the system message) plus an
    identical opening user turn is sent twice in a row, then the second
    call's reported `cached_tokens` is greater than zero -- a real cache
    hit against a real cache-capable backend, not merely the builder
    function producing identical bytes in isolation -- and the combined
    metered cost of both calls stays under the budget ceiling, with each
    call's own cost a small, plausible positive number.
    """
    registry = load_registry()
    entry = registry.get(_MODEL_ID)
    client = LiteLLMClient(registry)
    kestrel_md = KestrelMd(
        raw_text=_LIVE_KESTREL_MD_TEXT, verify_commands=VerifyCommands()
    )
    prefix = mark_cache_breakpoints(
        build_stable_prefix(_SYSTEM_PROMPT, kestrel_md), entry
    )
    opening_turn: Message = {
        "role": "user",
        "content": "Reply with exactly: kestrel",
    }
    messages: list[Message] = [*prefix, opening_turn]

    async def _call() -> UsageEvent:
        """Stream one completion for `messages` and return its usage event."""
        events = [
            event
            async for event in client.complete(
                messages=messages,
                tools=None,
                model_id=_MODEL_ID,
                effort="high",
                stream=True,
            )
        ]
        return next(event for event in events if isinstance(event, UsageEvent))

    first_usage = await asyncio.wait_for(_call(), timeout=_COLLECTION_TIMEOUT_S)
    second_usage = await asyncio.wait_for(_call(), timeout=_COLLECTION_TIMEOUT_S)

    assert second_usage.cached_tokens > 0

    meter = CostMeter()
    first_cost = meter.record(first_usage, entry)
    second_cost = meter.record(second_usage, entry)

    assert first_cost.usd > 0
    assert second_cost.usd > 0
    assert meter.session_usd < _BUDGET_CEILING_USD
