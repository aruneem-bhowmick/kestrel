"""Red-team system test: hostile terminal escape sequences in model output
must never reach the real terminal `run_repl` prints to.

Drives `run_repl` directly against a real `LiteLLMClient` and a real
mock HTTP server rather than through the packaged console script -- the
REPL is a library entry point other code (the TUI, this suite) calls
in-process, not something `kestrel` itself launches, so a real client
against a real mock server, in-process, is the honest boundary to red-
team at.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.config import load_config
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry
from kestrel.repl import run_repl

pytestmark = [pytest.mark.p008, pytest.mark.system, pytest.mark.redteam]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_ANSI_CASSETTE = _CASSETTES / "openrouter_glm52_ansi.sse"


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


def _write_system_config(tmp_path: Path) -> Path:
    """Write a temp ``kestrel.toml`` + ``models.toml`` pair naming a
    single openrouter route, and return the config path."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(
        """\
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


def test_hostile_escape_sequences_never_reach_stdout(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock backend replays a completion containing a
    screen-clear CSI sequence and an OSC window-title sequence, when
    `run_repl` renders it, then the raw escape bytes never appear in
    its output while the surrounding text survives."""
    base_url = mock_openai_server(_ANSI_CASSETTE)
    config_path = _write_system_config(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)
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
        input_fn=_scripted_input("hello", "/quit"),
        out=out,
    )

    stdout = out.getvalue()
    assert exit_code == 0
    assert "\x1b" not in stdout
    assert "\x9b" not in stdout
    assert "\x07" not in stdout
    assert "before" in stdout
    assert "after" in stdout
