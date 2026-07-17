"""System test: `run_repl`, driven against a hermetic mock backend
through a real `LiteLLMClient`, streams turns, prints the per-turn cost
line, and hot-swaps models via ``/model`` without losing the running
session total.

Drives `run_repl` directly rather than through the packaged console
script: the REPL is a library entry point `kestrel.tui`'s own cockpit
(and anything else in-process) can call, not something the `kestrel`
command line itself launches, so a real client against a real mock
HTTP server, in-process, is the honest boundary for this suite to test
at -- the same one `kestrel run`'s own equivalent coverage already
draws for the agent loop.

Both the config and registry fixtures used here are generated per test
run (into ``tmp_path``) rather than committed statically, because the zai
route's endpoint must point at a freshly started mock server on an
ephemeral port -- a real deployment would set that same field to a
genuine endpoint URL, so pointing it at a test double is the honest,
registry-driven path rather than a special test seam.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.config import load_config
from kestrel.cost.meter import CostMeter, format_cost_line
from kestrel.provider.events import UsageEvent
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry
from kestrel.registry.model import ModelEntry
from kestrel.repl import run_repl

pytestmark = [
    pytest.mark.p008,
    pytest.mark.system,
    pytest.mark.acceptance,
    pytest.mark.cost_regression,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"


def _scripted_input(*lines: str) -> Callable[[str], str]:
    """Build an `input_fn` for `run_repl` that yields `lines` in order
    and raises `EOFError` once exhausted -- the same contract a real
    piped stdin gives the loop once the script runs out."""
    iterator = iter(lines)

    def _next_line(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return _next_line


def _write_system_config(tmp_path: Path, *, zai_endpoint: str) -> Path:
    """Write a temp ``kestrel.toml`` + ``models.toml`` pair and return the
    config path, with the zai entry's endpoint pointed at ``zai_endpoint``."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(
        f"""\
[[models]]
id = "glm-5.2"
backend = "openrouter"
provider_model = "z-ai/glm-5.2"
api_key_env = "OPENROUTER_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true

[[models]]
id = "glm-5.2-zai"
backend = "zai"
provider_model = "glm-5.2"
endpoint = "{zai_endpoint}"
api_key_env = "ZAI_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true
""",
        encoding="utf-8",
    )

    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "glm-5.2"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def _rate_matched_entry(*, id: str, backend: str, endpoint: str | None) -> ModelEntry:
    """Rebuild the fixture registry entry for ``id`` so the exact cost
    line it must produce can be computed via the real pricing functions
    instead of a hand-copied literal."""
    return ModelEntry(
        id=id,
        backend=backend,  # type: ignore[arg-type]
        provider_model="z-ai/glm-5.2" if backend == "openrouter" else "glm-5.2",
        endpoint=endpoint,
        api_key_env="OPENROUTER_API_KEY" if backend == "openrouter" else "ZAI_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )


def test_repl_streams_and_hot_swaps_model_preserving_session_total(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a REPL script that sends one turn, hot-swaps to the zai
    route via ``/model``, sends a second turn, and quits, when driven
    through `run_repl` against two hermetic mock backends, then both
    turns' streamed text and exact cost lines appear in order, and the
    loop exits 0."""
    openrouter_base = mock_openai_server(_CASSETTES / "openrouter_glm52_hello.sse")
    zai_base = mock_openai_server(_CASSETTES / "zai_glm52_hello.sse")
    config_path = _write_system_config(tmp_path, zai_endpoint=zai_base)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-zai")
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", openrouter_base)
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)

    config, _source = load_config(config_path)
    registry = load_registry(config.paths.models_file)
    client = LiteLLMClient(registry)
    out = io.StringIO()

    exit_code = run_repl(
        config,
        registry,
        client,
        "glm-5.2",
        input_fn=_scripted_input("hello", "/model glm-5.2-zai", "hello again", "/quit"),
        out=out,
    )

    assert exit_code == 0

    meter = CostMeter()
    openrouter_entry = _rate_matched_entry(
        id="glm-5.2", backend="openrouter", endpoint=None
    )
    zai_entry = _rate_matched_entry(id="glm-5.2-zai", backend="zai", endpoint=zai_base)
    first_turn = meter.record(UsageEvent(42, 7, 0), openrouter_entry)
    first_line = format_cost_line(first_turn, meter.session_usd)
    second_turn = meter.record(UsageEvent(40, 6, 0), zai_entry)
    second_line = format_cost_line(second_turn, meter.session_usd)

    stdout = out.getvalue()
    assert "Hello from GLM-5.2" in stdout
    assert first_line in stdout
    assert "Hello from Z.ai GLM" in stdout
    assert second_line in stdout
    assert stdout.index("Hello from GLM-5.2") < stdout.index("Hello from Z.ai GLM")
    assert stdout.index(first_line) < stdout.index(second_line)
