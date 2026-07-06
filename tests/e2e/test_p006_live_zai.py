"""Budget-capped live smoke test against the real Z.ai endpoint.

This is the zai-backend counterpart to the live OpenRouter smoke test: it
is the one place in the suite that spends real money against Z.ai's own
API, so it is opted into explicitly rather than run by default. The call
this test makes is expected to cost well under $0.01, and this file makes
exactly one such call -- the provider interface has no per-call token cap
yet, so the budget is kept by prompting for the shortest possible reply
rather than by a request parameter.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry

pytestmark = [pytest.mark.p006, pytest.mark.e2e, pytest.mark.live]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_API_KEY_ENV = "ZAI_API_KEY"
_MODEL_ID = "glm-5.2-zai"
_COLLECTION_TIMEOUT_S = 30.0

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1 and {_API_KEY_ENV} to run the live zai smoke test"
)


@pytest.mark.skipif(
    os.environ.get(_LIVE_TESTS_ENV) != "1" or not os.environ.get(_API_KEY_ENV),
    reason=_SKIP_REASON,
)
async def test_live_zai_completion_returns_text_and_usage() -> None:
    """Given the real Z.ai endpoint and a real credential, when a minimal
    completion is streamed, then it yields non-empty text and a usage
    event reporting tokens on both sides of the exchange."""
    registry = load_registry()
    client = LiteLLMClient(registry)

    async def _collect() -> list[Any]:
        """Run the live completion to exhaustion and return its full event list."""
        return [
            event
            async for event in client.complete(
                messages=[{"role": "user", "content": "Reply with exactly: kestrel"}],
                tools=None,
                model_id=_MODEL_ID,
                effort="high",
                stream=True,
            )
        ]

    events = await asyncio.wait_for(_collect(), timeout=_COLLECTION_TIMEOUT_S)

    text = "".join(event.text for event in events if hasattr(event, "text"))
    assert text.strip() != ""

    usage = next(event for event in events if hasattr(event, "input_tokens"))
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
