"""Acceptance suite encoding the initial milestone's checkable
Definition-of-Done clauses as executable scenarios.

Each test below stands in for one machine-checkable clause: a REPL built
from the packaged entry point streams a real-shaped completion, prints a
per-turn usage/cost line in the project's canonical format, hot-swaps
models mid-session without losing conversation history, and the Jetson
provisioning guide walks a fresh install all the way to a running REPL.
Every scenario here runs against the hermetic mock backend (see
``tests/fixtures/mock_openai.py``); the one clause that requires a real
provider call has its own opt-in twin in ``tests/e2e/test_p011_dod_live.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.cost.meter import CostMeter, format_cost_line
from kestrel.provider.events import UsageEvent
from kestrel.registry.model import ModelEntry
from kestrel.repl import SYSTEM_PROMPT

pytestmark = [pytest.mark.p011, pytest.mark.acceptance, pytest.mark.dod_phase_0]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TIMEOUT_S = 30.0

_COST_LINE_RE = re.compile(
    r"in:(?P<input>\d+)(?: \(cached:(?P<cached>\d+)\))? out:(?P<output>\d+) "
    r"· \$(?P<turn_usd>\d+\.\d{4}) turn · \$(?P<session_usd>\d+\.\d{4}) session"
)

_PROVISIONING_DOC_PATH = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "provisioning-jetson.md"
)
_EXPECTED_PROVISIONING_SECTIONS = [
    "Prerequisites",
    "Flash JetPack 6.2",
    "NVMe setup",
    "Power mode",
    "Python & uv",
    "Kestrel install",
    "Flight check",
    "Ollama (deferred)",
]
_SECTION_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def _write_system_config(tmp_path: Path, *, zai_endpoint: str) -> Path:
    """Write a temp ``kestrel.toml`` + ``models.toml`` pair naming one
    openrouter route and one zai route (pointed at ``zai_endpoint``), and
    return the config path."""
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


def _repl_env(openrouter_base: str) -> dict[str, str]:
    """Build the subprocess environment for a REPL run against the
    hermetic mock backends: real credentials are never needed, so both
    API key variables are set to fixed test values, and the OpenRouter
    route is redirected at the mock server via its documented test seam.
    """
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["ZAI_API_KEY"] = "sk-test-zai"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


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


def test_dod_repl_streams_completion(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a REPL script that sends one turn and quits, when run against
    a hermetic OpenRouter-shaped mock backend, then the streamed
    completion text reaches stdout and the process exits cleanly.

    This is the mock-backend twin of the "streams a GLM-5.2 completion via
    OpenRouter" checklist clause; the live twin proving the real
    OpenRouter path is ``tests/e2e/test_p011_dod_live.py``.
    """
    openrouter_base = mock_openai_server(_CASSETTES / "openrouter_glm52_hello.sse")
    zai_base = mock_openai_server(_CASSETTES / "zai_glm52_hello.sse")
    config_path = _write_system_config(tmp_path, zai_endpoint=zai_base)

    result = subprocess.run(
        [kestrel_executable, "--config", str(config_path)],
        input="hello\n/quit\n",
        capture_output=True,
        encoding="utf-8",
        env=_repl_env(openrouter_base),
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Hello from GLM-5.2" in result.stdout


@pytest.mark.cost_regression
def test_dod_prints_usage_cost_per_turn(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given the same one-turn REPL script, when it completes, then stdout
    carries a cost line in the project's canonical format with nonzero
    token counts on both sides of the exchange, and that line matches
    exactly what the real pricing functions compute from the cassette's
    own usage figures -- not merely a line shaped like a cost line.
    """
    openrouter_base = mock_openai_server(_CASSETTES / "openrouter_glm52_hello.sse")
    zai_base = mock_openai_server(_CASSETTES / "zai_glm52_hello.sse")
    config_path = _write_system_config(tmp_path, zai_endpoint=zai_base)

    result = subprocess.run(
        [kestrel_executable, "--config", str(config_path)],
        input="hello\n/quit\n",
        capture_output=True,
        encoding="utf-8",
        env=_repl_env(openrouter_base),
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    match = _COST_LINE_RE.search(result.stdout)
    assert match is not None, result.stdout
    assert int(match["input"]) > 0
    assert int(match["output"]) > 0

    meter = CostMeter()
    entry = _rate_matched_entry(id="glm-5.2", backend="openrouter", endpoint=None)
    turn = meter.record(UsageEvent(42, 7, 0), entry)
    expected_line = format_cost_line(turn, meter.session_usd)
    assert expected_line in result.stdout


def test_dod_model_hotswap(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a REPL script that sends one turn, hot-swaps to the zai route
    via ``/model``, sends a second turn, and quits, when run against two
    hermetic mock backends, then the second turn's reply comes from the
    zai-endpoint mock, both turns are priced under their own model's
    rates with the session total carried across the swap, and the second
    request's captured body shows the first exchange was sent along with
    it -- proving history survives the hot-swap, not just that both calls
    happened.
    """
    zai_requests: list[bytes] = []
    openrouter_base = mock_openai_server(_CASSETTES / "openrouter_glm52_hello.sse")
    zai_base = mock_openai_server(
        _CASSETTES / "zai_glm52_hello.sse", capture=zai_requests
    )
    config_path = _write_system_config(tmp_path, zai_endpoint=zai_base)

    script = "hello\n/model glm-5.2-zai\nhello again\n/quit\n"
    result = subprocess.run(
        [kestrel_executable, "--config", str(config_path)],
        input=script,
        capture_output=True,
        encoding="utf-8",
        env=_repl_env(openrouter_base),
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Hello from GLM-5.2" in result.stdout
    assert "Hello from Z.ai GLM" in result.stdout
    assert result.stdout.index("Hello from GLM-5.2") < result.stdout.index(
        "Hello from Z.ai GLM"
    )

    meter = CostMeter()
    openrouter_entry = _rate_matched_entry(
        id="glm-5.2", backend="openrouter", endpoint=None
    )
    zai_entry = _rate_matched_entry(id="glm-5.2-zai", backend="zai", endpoint=zai_base)
    first_turn = meter.record(UsageEvent(42, 7, 0), openrouter_entry)
    first_line = format_cost_line(first_turn, meter.session_usd)
    second_turn = meter.record(UsageEvent(40, 6, 0), zai_entry)
    second_line = format_cost_line(second_turn, meter.session_usd)
    assert first_line in result.stdout
    assert second_line in result.stdout

    assert len(zai_requests) == 1
    second_request_messages = json.loads(zai_requests[0])["messages"]
    sent_texts = [message.get("content", "") for message in second_request_messages]
    assert sent_texts == [
        SYSTEM_PROMPT,
        "hello",
        "Hello from GLM-5.2",
        "hello again",
    ]


def test_dod_provisioning_doc_complete() -> None:
    """Given the committed Jetson provisioning guide, when checked against
    the "provisioning doc" clause of the exit criteria, then it still has
    every required section in order -- re-asserting the guide's own
    structural contract -- and explicitly walks the reader through both
    ``uv run kestrel doctor`` and ``uv run kestrel``, the two commands a
    fresh install is supposed to end at.
    """
    text = _PROVISIONING_DOC_PATH.read_text(encoding="utf-8")
    assert _SECTION_HEADING_RE.findall(text) == _EXPECTED_PROVISIONING_SECTIONS

    lines = text.splitlines()
    assert "uv run kestrel doctor" in lines
    # A plain substring check would pass on "uv run kestrel doctor" alone;
    # this must be its own line to prove the guide separately walks the
    # reader to the bare REPL command, not just the doctor subcommand.
    assert "uv run kestrel" in lines
