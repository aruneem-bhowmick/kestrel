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
ordered sequence of replies (a different reply per call, for tests that
script a multi-turn conversation), or a JSON error body at a given
status code (a provider failure). The ordered-sequence mode itself can
mix cassette files with bare status codes, so a script can express
"fail with 429, then succeed" as a single sequence. See
``tests/fixtures/cassettes/README.md`` for how the cassette files
themselves are structured and re-recorded.

Every cassette is authored once, as a streamed chunk sequence -- but not
every caller streams. A request whose own JSON body sets ``"stream":
false`` gets the same cassette folded into one non-streaming
chat-completion object instead of replayed verbatim, since a real
OpenAI-compatible backend never answers a non-streaming request with an
``event-stream`` body: litellm's own client raises trying to parse one as
plain JSON. This keeps one checked-in fixture format serving both kinds
of call rather than maintaining a second cassette shape.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

_STARTUP_POLL_INTERVAL_S = 0.005
_STARTUP_TIMEOUT_S = 10.0


class MockOpenAIServer:
    """A single running mock chat-completions server.

    Replays a fixed SSE cassette (``cassette_path`` set), an ordered
    sequence of replies (``cassette_sequence`` set, one entry per request
    in arrival order -- each entry either a cassette file or a bare
    status code to fail that one request with), or a fixed JSON error
    body (neither set) for every request it receives, regardless of the
    request's path or contents -- this suite only ever needs one canned
    script per server instance.
    """

    def __init__(
        self,
        *,
        cassette_path: Path | None,
        cassette_sequence: Sequence[Path | int] | None = None,
        status_code: int,
        extra_headers: Mapping[str, str] | None,
        capture: list[bytes] | None = None,
    ) -> None:
        """Configure (without starting) a server for one canned behavior.

        ``cassette_path`` and ``cassette_sequence`` are mutually
        exclusive: passing both raises ``ValueError`` immediately, before
        any socket is opened, since the two modes disagree about what the
        Nth request should receive. ``cassette_sequence`` must also be
        non-empty -- an empty sequence has no entry for any request to
        clamp to, which would otherwise surface as an ``IndexError`` out
        of ``_next_reply`` on the first request instead of a clear error
        at construction time.

        Each entry in ``cassette_sequence`` is either a ``Path`` (that
        request replays the named cassette at ``status_code``, normally
        200) or an ``int`` (that request replays the fixed error body,
        described below, at that status code instead) -- letting one
        sequence script a transient failure followed by a successful
        reply, e.g. ``[429, some_cassette]``.

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
        if cassette_sequence is not None and len(cassette_sequence) == 0:
            raise ValueError("cassette_sequence must not be empty")
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
        cassette or status code selected for this call -- everything but
        the request's own ``stream`` field plays no role in what gets
        replayed."""
        body = await request.body()
        if self._capture is not None:
            self._capture.append(body)
        reply = self._next_reply()
        if isinstance(reply, int):
            return self._error_response(reply)
        if reply is None:
            return self._error_response(self._status_code)
        return self._cassette_response(reply, stream=_wants_stream(body))

    def _error_response(self, status_code: int) -> Response:
        """Render the fixed mock-provider-error JSON body at ``status_code``."""
        return PlainTextResponse(
            '{"error": {"message": "mock provider error", "type": "mock_error"}}',
            status_code=status_code,
            media_type="application/json",
            headers=self._extra_headers,
        )

    def _cassette_response(self, cassette_path: Path, *, stream: bool) -> Response:
        """Render ``cassette_path`` the way this request asked for: its raw
        bytes verbatim as an SSE response when ``stream`` is True, or the
        same chunks folded into one non-streaming chat-completion JSON
        object via :func:`_consolidate_cassette_chunks` when it is False.
        """
        cassette_text = cassette_path.read_text(encoding="utf-8")
        if stream:
            return PlainTextResponse(
                cassette_text,
                status_code=self._status_code,
                media_type="text/event-stream",
                headers=self._extra_headers,
            )
        return JSONResponse(
            _consolidate_cassette_chunks(cassette_text),
            status_code=self._status_code,
            headers=self._extra_headers,
        )

    def _next_reply(self) -> Path | int | None:
        """Pick the cassette or status code to serve for the request that just arrived.

        A fixed ``cassette_path`` (or ``None``, for the error-only mode)
        never changes across calls. A ``cassette_sequence`` instead
        advances a call counter shared across requests -- guarded by a
        lock since uvicorn serves requests from its own event-loop thread
        -- and clamps to the final entry once every scripted reply has
        been served, so a caller that keeps sending requests past the end
        of the script replays the last scripted entry rather than this
        server raising. A clamped-to or in-sequence ``int`` entry is
        returned as-is, letting the caller fail that one request with the
        named status code without disturbing the entries around it.
        """
        if self._cassette_sequence is None:
            return self._cassette_path
        with self._call_index_lock:
            index = min(self._call_index, len(self._cassette_sequence) - 1)
            self._call_index += 1
        return self._cassette_sequence[index]


def _wants_stream(body: bytes) -> bool:
    """Whether the request's own JSON body asked for a streamed reply.

    Defaults to True -- matching every request this suite sent before a
    non-streaming caller ever existed -- when ``body`` is not valid JSON
    or simply omits the field, exactly like the real backends this
    server stands in for.
    """
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, dict):
        return True
    return bool(parsed.get("stream", True))


def _consolidate_cassette_chunks(cassette_text: str) -> dict[str, Any]:
    """Fold a cassette's own streamed chunk sequence into one
    non-streaming chat-completion response object -- the shape litellm
    expects back for a request whose own body set ``"stream": false``.

    Concatenates every chunk's ``delta.content`` in arrival order into
    the reply's own ``message.content``, keeps the last non-null
    ``finish_reason`` seen, and carries the final chunk's own populated
    ``usage`` object through unchanged. Only plain-text completions are
    supported -- no cassette in this suite scripts a tool call for a
    non-streaming caller, so tool-call deltas are not folded here.
    """
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    completion_id = ""
    created = 0
    model = ""
    for line in cassette_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        completion_id = chunk.get("id") or completion_id
        created = chunk.get("created") or created
        model = chunk.get("model") or model
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                content_parts.append(content)
            reason = choices[0].get("finish_reason")
            if reason is not None:
                finish_reason = reason
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
    }
