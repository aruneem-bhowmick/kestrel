"""System test: the installed console script streams turns from a hermetic
mock backend, prints the per-turn cost line, and hot-swaps models via
``/model`` without losing the running session total.

Both the config and registry fixtures used here are generated per test
run (into ``tmp_path``) rather than committed statically, because the zai
route's endpoint must point at a freshly started mock server on an
ephemeral port -- a real deployment would set that same field to a
genuine endpoint URL, so pointing it at a test double is the honest,
registry-driven path rather than a special test seam.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.cost.meter import CostMeter, format_cost_line
from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry

pytestmark = [
    pytest.mark.p008,
    pytest.mark.system,
    pytest.mark.acceptance,
    pytest.mark.cost_regression,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TIMEOUT_S = 30.0


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
    kestrel_executable: str,
) -> None:
    """Given a REPL script that sends one turn, hot-swaps to the zai
    route via ``/model``, sends a second turn, and quits, when run
    against two hermetic mock backends, then both turns' streamed text
    and exact cost lines appear in order, and the process exits 0."""
    openrouter_base = mock_openai_server(_CASSETTES / "openrouter_glm52_hello.sse")
    zai_base = mock_openai_server(_CASSETTES / "zai_glm52_hello.sse")
    config_path = _write_system_config(tmp_path, zai_endpoint=zai_base)

    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["ZAI_API_KEY"] = "sk-test-zai"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    script = "hello\n/model glm-5.2-zai\nhello again\n/quit\n"

    result = subprocess.run(
        [kestrel_executable, "--config", str(config_path)],
        input=script,
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    meter = CostMeter()
    openrouter_entry = _rate_matched_entry(
        id="glm-5.2", backend="openrouter", endpoint=None
    )
    zai_entry = _rate_matched_entry(id="glm-5.2-zai", backend="zai", endpoint=zai_base)
    first_turn = meter.record(UsageEvent(42, 7, 0), openrouter_entry)
    first_line = format_cost_line(first_turn, meter.session_usd)
    second_turn = meter.record(UsageEvent(40, 6, 0), zai_entry)
    second_line = format_cost_line(second_turn, meter.session_usd)

    stdout = result.stdout
    assert "Hello from GLM-5.2" in stdout
    assert first_line in stdout
    assert "Hello from Z.ai GLM" in stdout
    assert second_line in stdout
    assert stdout.index("Hello from GLM-5.2") < stdout.index("Hello from Z.ai GLM")
    assert stdout.index(first_line) < stdout.index(second_line)
