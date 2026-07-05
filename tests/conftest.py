"""Shared pytest fixtures for the Kestrel test suite."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

import pytest

# LiteLLM fetches its remote model-cost map over the network on import
# unless told not to; nothing in this suite depends on that map (Kestrel
# prices calls from its own registry rates), so importing it should never
# require -- or wait on -- a network round trip.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fixtures.mock_openai import MockOpenAIServer  # noqa: E402

logger = logging.getLogger("tests.conftest")


class MockOpenAIServerFactory(Protocol):
    """Callable returned by the ``mock_openai_server`` fixture."""

    def __call__(
        self,
        cassette_path: Path | None = None,
        *,
        status_code: int = 200,
        extra_headers: Mapping[str, str] | None = None,
    ) -> str:
        """Start a mock server for one canned behavior; return its base_url."""
        ...


@pytest.fixture
def mock_openai_server() -> Iterator[MockOpenAIServerFactory]:
    """Yield a factory for hermetic, per-test mock chat-completions servers.

    Call the yielded factory once per desired canned behavior -- a
    cassette replay, or a fixed error status -- to boot a fresh server and
    get back its ``base_url``. Every server started this way is torn down
    automatically at the end of the test.
    """
    servers: list[MockOpenAIServer] = []

    def _start(
        cassette_path: Path | None = None,
        *,
        status_code: int = 200,
        extra_headers: Mapping[str, str] | None = None,
    ) -> str:
        """Start one server for this test and return its base_url."""
        server = MockOpenAIServer(
            cassette_path=cassette_path,
            status_code=status_code,
            extra_headers=extra_headers,
        )
        server.start()
        servers.append(server)
        return server.base_url

    yield _start

    for server in servers:
        try:
            server.stop()
        except Exception:
            # One server failing to shut down cleanly shouldn't leave the
            # rest of the test session's servers running.
            logger.exception("mock_openai_server: server.stop() failed")
