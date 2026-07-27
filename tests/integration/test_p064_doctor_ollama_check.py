"""Integration tests: `doctor._check_ollama` against a real mock Ollama server.

Unlike a unit test stubbing out `OllamaEmbeddingClient` entirely, these
tests drive the check through a genuine HTTP round trip against
`mock_ollama_server`, proving the router-resolve-then-probe path
actually reaches a listening backend rather than only exercising the
check's own branching logic in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import litellm
import pytest

import kestrel.doctor as doctor_module
from kestrel.config import GeneralConfig, KestrelConfig
from kestrel.doctor import CheckResult, CheckStatus, _check_ollama
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p064, pytest.mark.integration]


def _registry_and_config(*, endpoint: str) -> tuple[Registry, KestrelConfig]:
    """Build a single-entry, ollama-backed registry and a config whose
    default router policy resolves the "embed" task class to it (via its
    own `"local"` tag)."""
    entry = ModelEntry(
        id="nomic-embed-text",
        backend="ollama",
        provider_model="nomic-embed-text",
        endpoint=endpoint,
        context_window=8192,
        max_output=1,
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=False,
        supports_cache=False,
        tags=frozenset({"local"}),
    )
    registry = Registry(models={"nomic-embed-text": entry}, source=None)
    config = KestrelConfig(general=GeneralConfig(default_model="nomic-embed-text"))
    return registry, config


def _check_ollama_closing_litellm_client_within_the_same_loop(
    monkeypatch: pytest.MonkeyPatch,
    registry: Registry,
    config: KestrelConfig,
    *,
    live: bool,
) -> CheckResult:
    """Run `_check_ollama`, wrapping its internal probe so litellm's own
    cached, process-wide async HTTP client is closed inside the very
    event loop that created it.

    `_check_ollama` drives its probe through its own private
    `asyncio.run()` call, exactly like `_check_endpoint` already does --
    opening and closing a fresh event loop around one diagnostic
    round trip, which is exactly right for a one-shot CLI invocation.
    litellm caches its own async client at module scope rather than
    per-call, though, so left alone it outlives that loop; closing it
    from any *other* loop later (e.g. a naive fixture teardown) cannot
    cleanly release the connection it opened. Wrapping the probe itself
    to close the client in its own last moment, before `asyncio.run()`
    tears the loop down, is what actually avoids leaking it.
    """
    original_probe = doctor_module._probe_ollama

    async def _probe_then_close_client(registry_: Registry, entry: ModelEntry) -> None:
        """Run the original probe, then close litellm's client in the
        same loop the probe just used."""
        try:
            await original_probe(registry_, entry)
        finally:
            client = vars(litellm).get("module_level_aclient")
            if client is not None:
                await client.close()
                del litellm.module_level_aclient

    monkeypatch.setattr(doctor_module, "_probe_ollama", _probe_then_close_client)
    return _check_ollama(registry, config, live=live)


def test_check_ollama_ok_against_a_real_mock_server(
    monkeypatch: pytest.MonkeyPatch,
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given a live-gated ollama check pointed at a real (mocked) Ollama
    server that answers embedding requests, when run with `live=True`,
    then it reports OK naming the resolved model id."""
    base_url = mock_ollama_server(embeddings=[[0.1, 0.2]])
    registry, config = _registry_and_config(endpoint=base_url)

    result = _check_ollama_closing_litellm_client_within_the_same_loop(
        monkeypatch, registry, config, live=True
    )

    assert result.status is CheckStatus.OK
    assert result.detail == "nomic-embed-text"


def test_check_ollama_fails_naming_the_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_ollama_server: Callable[..., str],
) -> None:
    """Given a live-gated ollama check pointed at a server that fails
    every request with a 500, when run with `live=True`, then it FAILs
    naming the embedding error rather than letting it escape."""
    base_url = mock_ollama_server(status_code=500)
    registry, config = _registry_and_config(endpoint=base_url)

    result = _check_ollama_closing_litellm_client_within_the_same_loop(
        monkeypatch, registry, config, live=True
    )

    assert result.status is CheckStatus.FAIL
    assert "nomic-embed-text" in result.detail


def test_check_ollama_skips_without_live_and_starts_no_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `live=False`, when the ollama check runs, then it SKIPs with
    "pass --live" without ever resolving a model id or reaching a
    network endpoint -- no `mock_ollama_server` fixture is even started
    for this test, proving no call is attempted."""
    registry, config = _registry_and_config(endpoint="http://unused.invalid")

    def _unexpected_resolve(*_args: object, **_kwargs: object) -> str:
        """Stand in for `resolve_model_id`, failing the test if reached."""
        raise AssertionError("resolve_model_id should not run when live=False")

    monkeypatch.setattr(doctor_module, "resolve_model_id", _unexpected_resolve)

    result = _check_ollama(registry, config, live=False)

    assert result.status is CheckStatus.SKIP
    assert result.detail == "pass --live"
