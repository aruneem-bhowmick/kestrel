"""A hermetic stand-in for a local Ollama embedding endpoint.

Mirrors `mock_openai.py`'s own shape: a genuine Starlette application
under uvicorn, bound to an ephemeral localhost port, because the adapter
under test (`kestrel.kb.embeddings.OllamaEmbeddingClient`) reaches it
through litellm's own HTTP client -- which this suite does not construct
or otherwise get to intercept -- so only a real listening socket looks
like a real backend to it. Unlike `MockOpenAIServer` (which replays SSE
chat-completion cassettes for an arbitrary path), a server started here
only ever answers Ollama's batched embedding route, `POST /api/embed`,
with either a scripted `{"embeddings": [...]}` body or a fixed error
status.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

_STARTUP_POLL_INTERVAL_S = 0.005
_STARTUP_TIMEOUT_S = 10.0


class MockOllamaServer:
    """A single running mock Ollama embedding server.

    Replays a fixed `embeddings` payload (`embeddings` given) as
    `{"embeddings": [...]}` at `status_code`, or a fixed JSON error body
    (`embeddings` omitted) at `status_code` instead -- this suite only
    ever needs one canned script per server instance.
    """

    def __init__(
        self,
        *,
        embeddings: Sequence[Sequence[float]] | None,
        status_code: int = 200,
        capture: list[bytes] | None = None,
    ) -> None:
        """Configure (without starting) a server for one canned behavior.

        When `capture` is given, every request's raw body is appended to
        it in arrival order, mirroring `MockOpenAIServer`'s own `capture`
        parameter -- letting a caller inspect exactly what a client sent
        (e.g. that `"input"` and `"model"` round-tripped correctly)
        without turning this server into a request-aware simulator for
        every other test that has no need to look.
        """
        self._embeddings = (
            [list(vector) for vector in embeddings] if embeddings is not None else None
        )
        self._status_code = status_code
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
                raise RuntimeError("mock Ollama server did not start in time")
            time.sleep(_STARTUP_POLL_INTERVAL_S)

        self._thread = thread
        self._server = server
        port = server.servers[0].sockets[0].getsockname()[1]
        # No "/v1" suffix, unlike the OpenAI mock's own base_url: Ollama's
        # real endpoint is a bare host:port, and litellm appends "/api/embed"
        # itself when the api_base it was given doesn't already end with it.
        self.base_url = f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        """Signal the server to exit and wait for its thread to finish."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _handle(self, request: Request) -> Response:
        """Record the request body (if configured to), then reply with the
        scripted embeddings or the fixed error status for this server
        instance -- every request gets the same canned reply."""
        body = await request.body()
        if self._capture is not None:
            self._capture.append(body)
        if self._embeddings is None:
            return self._error_response()
        return JSONResponse(
            {"embeddings": self._embeddings}, status_code=self._status_code
        )

    def _error_response(self) -> Response:
        """Render the fixed mock-provider-error JSON body at ``status_code``."""
        return PlainTextResponse(
            '{"error": {"message": "mock ollama error", "type": "mock_error"}}',
            status_code=self._status_code,
            media_type="application/json",
        )
