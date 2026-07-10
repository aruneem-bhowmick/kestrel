"""A hermetic stand-in for an OpenAI-compatible chat completions endpoint.

Every integration test that exercises a real ``ProviderClient`` backend
needs something on the other end of the wire that behaves like a real
provider, without ever reaching the network. This module boots a genuine
Starlette application under uvicorn, bound to an ephemeral localhost port,
because the adapter under test uses LiteLLM's own HTTP client -- which
this test suite does not construct or otherwise get to intercept -- so
only a real listening socket looks like a real backend to it.

A server started here replays one of three things on every request: a
single checked-in SSE cassette file (the same reply for every call), an
ordered sequence of cassette files (a different reply per call, for
tests that script a multi-turn conversation), or a JSON error body at a
given status code (a provider failure). See
``tests/fixtures/cassettes/README.md`` for how the cassette files
themselves are structured and re-recorded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
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

    Replays a fixed SSE cassette (``cassette_path`` set), an ordered
    sequence of SSE cassettes (``cassette_sequence`` set, one per request
    in arrival order), or a fixed JSON error body (neither set) for every
    request it receives, regardless of the request's path or contents --
    this suite only ever needs one canned script per server instance.
    """

    def __init__(
        self,
        *,
        cassette_path: Path | None,
        cassette_sequence: Sequence[Path] | None = None,
        status_code: int,
        extra_headers: Mapping[str, str] | None,
        capture: list[bytes] | None = None,
    ) -> None:
        """Configure (without starting) a server for one canned behavior.

        ``cassette_path`` and ``cassette_sequence`` are mutually
        exclusive: passing both raises ``ValueError`` immediately, before
        any socket is opened, since the two modes disagree about what the
        Nth request should receive.

        When ``capture`` is given, every request's raw body is appended to
        it in arrival order -- letting a caller inspect what a client
        actually sent (e.g. to confirm conversation history round-tripped
        across a model swap) without turning this server into a
        request-aware simulator for every other test that has no need to
        look.
        """
        if cassette_path is not None and cassette_sequence is not None:
            raise ValueError(
                "cassette_path and cassette_sequence are mutually "
                "exclusive; pass at most one"
            )
        self._cassette_path = cassette_path
        self._cassette_sequence = (
            tuple(cassette_sequence) if cassette_sequence is not None else None
        )
        self._call_index = 0
        self._call_index_lock = threading.Lock()
        self._status_code = status_code
        self._extra_headers = dict(extra_headers or {})
        self._capture = capture
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
        """Record the request body (if configured to), then reply with the
        cassette selected for this call (or an error) -- the request
        otherwise plays no role in what gets replayed."""
        if self._capture is not None:
            self._capture.append(await request.body())
        cassette_path = self._next_cassette_path()
        if cassette_path is None:
            return PlainTextResponse(
                '{"error": {"message": "mock provider error", "type": "mock_error"}}',
                status_code=self._status_code,
                media_type="application/json",
                headers=self._extra_headers,
            )
        body = cassette_path.read_text(encoding="utf-8")
        return PlainTextResponse(
            body,
            status_code=self._status_code,
            media_type="text/event-stream",
            headers=self._extra_headers,
        )

    def _next_cassette_path(self) -> Path | None:
        """Pick the cassette to serve for the request that just arrived.

        A fixed ``cassette_path`` (or ``None``, for the error-only mode)
        never changes across calls. A ``cassette_sequence`` instead
        advances a call counter shared across requests -- guarded by a
        lock since uvicorn serves requests from its own event-loop thread
        -- and clamps to the final entry once every scripted reply has
        been served, so a caller that keeps sending requests past the end
        of the script replays the last scripted turn rather than this
        server raising.
        """
        if self._cassette_sequence is None:
            return self._cassette_path
        with self._call_index_lock:
            index = min(self._call_index, len(self._cassette_sequence) - 1)
            self._call_index += 1
        return self._cassette_sequence[index]
