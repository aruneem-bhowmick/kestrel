"""A hermetic stand-in for an OpenAI-compatible chat completions endpoint.

Every integration test that exercises a real ``ProviderClient`` backend
needs something on the other end of the wire that behaves like a real
provider, without ever reaching the network. This module boots a genuine
Starlette application under uvicorn, bound to an ephemeral localhost port,
because the adapter under test uses LiteLLM's own HTTP client -- which
this test suite does not construct or otherwise get to intercept -- so
only a real listening socket looks like a real backend to it.

A server started here replays one of two things on every request,
regardless of path or body: a checked-in SSE cassette file (a successful
streaming completion), or a JSON error body at a given status code (a
provider failure). See ``tests/fixtures/cassettes/README.md`` for how the
cassette files themselves are structured and re-recorded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

_STARTUP_POLL_INTERVAL_S = 0.005
_STARTUP_TIMEOUT_S = 10.0


class MockOpenAIServer:
    """A single running mock chat-completions server.

    Replays either a fixed SSE cassette (``cassette_path`` set) or a fixed
    JSON error body (``cassette_path`` is ``None``) for every request it
    receives, regardless of the request's path or contents -- this suite
    only ever needs one canned behavior per server instance.
    """

    def __init__(
        self,
        *,
        cassette_path: Path | None,
        status_code: int,
        extra_headers: Mapping[str, str] | None,
    ) -> None:
        """Configure (without starting) a server for one canned behavior."""
        self._cassette_path = cassette_path
        self._status_code = status_code
        self._extra_headers = dict(extra_headers or {})
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def start(self) -> None:
        """Boot the server in a background thread and wait until it accepts connections.

        ``uvicorn.Server.run`` drives its own ``asyncio.run`` call, which
        owns the event loop's full lifecycle (creation, shutdown, and
        close) -- running it in a plain background thread, rather than
        threading an event loop through by hand, is what lets that
        lifecycle clean up correctly when the server stops.
        """
        app = Starlette(
            routes=[Route("/{path:path}", self._handle, methods=["GET", "POST"])]
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("mock OpenAI server did not start in time")
            time.sleep(_STARTUP_POLL_INTERVAL_S)

        self._thread = thread
        self._server = server
        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"

    def stop(self) -> None:
        """Signal the server to exit and wait for its thread to finish."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _handle(self, request: Request) -> Response:
        """Reply with the configured cassette or error, ignoring the request itself."""
        del request
        if self._cassette_path is None:
            return PlainTextResponse(
                '{"error": {"message": "mock provider error", "type": "mock_error"}}',
                status_code=self._status_code,
                media_type="application/json",
                headers=self._extra_headers,
            )
        body = self._cassette_path.read_text(encoding="utf-8")
        return PlainTextResponse(
            body,
            status_code=self._status_code,
            media_type="text/event-stream",
            headers=self._extra_headers,
        )
