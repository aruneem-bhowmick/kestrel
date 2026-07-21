"""Shared pytest fixtures for the Kestrel test suite."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import pytest

# LiteLLM fetches its remote model-cost map over the network on import
# unless told not to; nothing in this suite depends on that map (Kestrel
# prices calls from its own registry rates), so importing it should never
# require -- or wait on -- a network round trip.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402
from fixtures.mock_ollama import MockOllamaServer  # noqa: E402
from fixtures.mock_openai import MockOpenAIServer  # noqa: E402

logger = logging.getLogger("tests.conftest")


class MockOpenAIServerFactory(Protocol):
    """Callable returned by the ``mock_openai_server`` fixture."""

    def __call__(
        self,
        cassette_path: Path | None = None,
        *,
        cassette_sequence: Sequence[Path | int] | None = None,
        status_code: int = 200,
        extra_headers: Mapping[str, str] | None = None,
        capture: list[bytes] | None = None,
    ) -> str:
        """Start a mock server for one canned behavior; return its base_url.

        Pass ``cassette_sequence`` instead of ``cassette_path`` to script
        a different reply per request (the Nth request gets the Nth
        entry, clamped once the sequence runs out); passing both is a
        ``ValueError``, raised before the server starts. An entry may be
        a cassette ``Path`` or a bare ``int`` status code, so a sequence
        can script a transient failure (e.g. ``[429, some_cassette]``).

        Pass ``capture`` to also record every request's raw body into that
        list, in arrival order -- useful for asserting what a client
        actually sent (e.g. that conversation history survived a model
        swap) without reaching into the server's internals.
        """
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
        cassette_sequence: Sequence[Path | int] | None = None,
        status_code: int = 200,
        extra_headers: Mapping[str, str] | None = None,
        capture: list[bytes] | None = None,
    ) -> str:
        """Start one server for this test and return its base_url."""
        server = MockOpenAIServer(
            cassette_path=cassette_path,
            cassette_sequence=cassette_sequence,
            status_code=status_code,
            extra_headers=extra_headers,
            capture=capture,
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


class MockOllamaServerFactory(Protocol):
    """Callable returned by the ``mock_ollama_server`` fixture."""

    def __call__(
        self,
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        status_code: int = 200,
        capture: list[bytes] | None = None,
    ) -> str:
        """Start a mock Ollama embedding server for one canned behavior; return its base_url.

        Pass ``embeddings`` to script a successful ``{"embeddings": [...]}``
        reply; omit it to always answer with the fixed mock-provider-error
        body at ``status_code`` instead. Pass ``capture`` to also record
        every request's raw body into that list, in arrival order --
        mirroring ``mock_openai_server``'s own ``capture`` parameter.
        """
        ...


async def _close_litellm_module_level_aclient() -> None:
    """Close and drop litellm's own lazily-created, process-wide async HTTP
    client, if this test caused one to exist.

    litellm creates this client on first use and caches it directly on its
    own module (``litellm.module_level_aclient``), reusing that same
    instance -- and its pooled connections -- for the rest of the process.
    A test that embeds against a `mock_ollama_server` leaves that pool
    holding a connection to a server this fixture is about to tear down;
    left alone, the dead socket is only noticed whenever Python happens to
    garbage-collect it, which can land during an unrelated, later test and
    fail it under this suite's strict unraisable-exception handling.
    Dropping the cached attribute (rather than merely closing the client)
    lets litellm's own lazy-import machinery build a fresh one the next
    time anything needs it.
    """
    client = vars(litellm).get("module_level_aclient")
    if client is not None:
        await client.close()
        del litellm.module_level_aclient


@pytest.fixture
async def mock_ollama_server() -> AsyncIterator[MockOllamaServerFactory]:
    """Yield a factory for hermetic, per-test mock Ollama embedding servers.

    Call the yielded factory once per desired canned behavior -- a
    scripted embeddings reply, or a fixed error status -- to boot a fresh
    server and get back its ``base_url``. Every server started this way
    is torn down automatically at the end of the test, along with the
    litellm-internal client connection it was talking to.
    """
    servers: list[MockOllamaServer] = []

    def _start(
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        status_code: int = 200,
        capture: list[bytes] | None = None,
    ) -> str:
        """Start one server for this test and return its base_url."""
        server = MockOllamaServer(
            embeddings=embeddings, status_code=status_code, capture=capture
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
            logger.exception("mock_ollama_server: server.stop() failed")

    try:
        await _close_litellm_module_level_aclient()
    except Exception:
        logger.exception(
            "mock_ollama_server: closing litellm's module-level client failed"
        )


@pytest.fixture
def kestrel_executable() -> str:
    """Locate the installed ``kestrel`` console script for subprocess tests.

    ``uv run pytest`` puts the environment's script directory on ``PATH``,
    so :func:`shutil.which` finds it directly. As a fallback (e.g. a test
    runner invoking pytest without going through ``uv run``), the script
    lives alongside the interpreter currently running, since console
    scripts install into the same directory as their environment's Python.
    """
    found = shutil.which("kestrel")
    if found is not None:
        return found

    exe_dir = Path(sys.executable).parent
    for candidate_name in ("kestrel", "kestrel.exe"):
        candidate = exe_dir / candidate_name
        if candidate.exists():
            return str(candidate)

    pytest.fail(
        "kestrel console script not found on PATH or in the environment's "
        "script directory"
    )
