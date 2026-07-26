"""Integration tests: `kestrel.kb.writeback.propose_learnings` driven
against a real `LiteLLMClient` and a mock chat-completions server,
proving the real request/response round trip -- not just the pure
line-parsing logic already covered at the unit level.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import TerminationReason
from kestrel.agent.walkthrough import Walkthrough
from kestrel.cost import compute_turn_cost
from kestrel.kb.writeback import ProposedLearning, WritebackError, propose_learnings
from kestrel.provider.events import UsageEvent
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p062, pytest.mark.integration]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_LEARNINGS_TWO = _CASSETTES / "learnings_two.sse"
_LEARNINGS_NONE = _CASSETTES / "learnings_none.sse"

_MODEL_ID = "glm-5.2-cheap"
_PROVIDER_MODEL = "z-ai/glm-5.2-cheap"


def _registry() -> Registry:
    """A single-entry `Registry` naming the "cheap"-tagged model a real
    writeback proposal call would be routed to."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend="openrouter",
        provider_model=_PROVIDER_MODEL,
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.10"),
        usd_per_mtok_output=Decimal("0.20"),
        usd_per_mtok_cached=Decimal("0.02"),
        supports_tools=True,
        supports_cache=True,
        tags=frozenset({"cheap"}),
    )
    return Registry(models={_MODEL_ID: entry}, source=None)


def _walkthrough() -> Walkthrough:
    """A small, otherwise-arbitrary `Walkthrough` naming one touched
    file, standing in for a real finished task's own summary."""
    return Walkthrough(
        task_id="int-p062-1",
        reason=TerminationReason.TASK_COMPLETE,
        turns_used=2,
        total_usd=Decimal("0.01"),
        touched_paths=("src/greet.py",),
        verification=None,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> LiteLLMClient:
    """A `LiteLLMClient` bound to `_registry`, with a fake API key set."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    return LiteLLMClient(_registry())


async def test_two_well_formed_lines_parse_and_the_malformed_one_is_dropped(
    client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a cassette scripting three `LEARNING:` lines -- two
    well-formed, one deliberately missing its `| TAGS:` separator --
    when `propose_learnings` runs, then exactly the two well-formed
    learnings come back, in order, and the malformed line contributes
    nothing rather than raising."""
    captured: list[bytes] = []
    base_url = mock_openai_server(_LEARNINGS_TWO, capture=captured)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    result = await propose_learnings(_walkthrough(), client=client, model_id=_MODEL_ID)

    assert result == (
        ProposedLearning(
            text="This repo's Makefile targets assume tabs, not spaces, "
            "for recipe indentation",
            tags=("style", "make"),
        ),
        ProposedLearning(
            text="Run `uv run pytest -m sanity` before pushing any change here",
            tags=("testing",),
        ),
    )

    assert len(captured) == 1
    assert json.loads(captured[0])["stream"] is False


async def test_a_none_reply_produces_an_empty_tuple(
    client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a cassette scripting the literal `"NONE"` reply, when
    `propose_learnings` runs, then it returns an empty tuple."""
    base_url = mock_openai_server(_LEARNINGS_NONE)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    result = await propose_learnings(_walkthrough(), client=client, model_id=_MODEL_ID)

    assert result == ()


async def test_a_rejected_call_surfaces_as_writeback_error(
    client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a backend that rejects the request outright (401, mapped to
    the unretriable `AuthError`), when `propose_learnings` runs, then the
    raw provider error never escapes -- it surfaces as `WritebackError`
    instead."""
    base_url = mock_openai_server(status_code=401)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    with pytest.raises(WritebackError, match="model call failed"):
        await propose_learnings(_walkthrough(), client=client, model_id=_MODEL_ID)


@pytest.mark.cost_regression
@pytest.mark.parametrize(
    ("cassette", "expected_cost"),
    [
        (_LEARNINGS_TWO, Decimal("0.000034")),
        (_LEARNINGS_NONE, Decimal("0.000020")),
    ],
    ids=["two_learnings", "none"],
)
async def test_a_learnings_cassettes_own_usage_prices_within_an_unsurprising_band(
    client: LiteLLMClient,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    cassette: Path,
    expected_cost: Decimal,
) -> None:
    """Given each of the two writeback cassettes in turn, when the raw
    usage figures a real proposal call against it would produce are
    priced via `compute_turn_cost` against the same "cheap" rate card
    `test_p047_critique_scripted.py` already exercises for a similarly
    short, capped completion, then it prices to a small, pinned amount
    well under a cent -- proof that a proposal call's own token
    footprint stays unsurprising in size, not merely that it succeeds.
    That figure is never billed through a `CostMeter` here:
    `propose_learnings` itself has no meter to bill through, matching
    this suite's own scope fence that a caller, not this module, is
    responsible for accounting a proposal call's spend."""
    entry = _registry().get(_MODEL_ID)
    base_url = mock_openai_server(cassette)
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    events = [
        event
        async for event in client.complete(
            [{"role": "user", "content": "irrelevant"}],
            None,
            _MODEL_ID,
            "high",
            stream=False,
            max_tokens=256,
        )
    ]
    usage = next(e for e in events if isinstance(e, UsageEvent))
    cost = compute_turn_cost(usage, entry)

    assert cost == expected_cost
    assert cost < Decimal("0.01")
