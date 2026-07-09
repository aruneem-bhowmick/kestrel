"""Red-team system test: hostile terminal escape sequences in model output
must never reach the real terminal the REPL prints to.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p008, pytest.mark.system, pytest.mark.redteam]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_ANSI_CASSETTE = _CASSETTES / "openrouter_glm52_ansi.sse"
_TIMEOUT_S = 30.0


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
    kestrel_executable: str,
) -> None:
    """Given the mock backend replays a completion containing a
    screen-clear CSI sequence and an OSC window-title sequence, when the
    REPL renders it, then the raw escape bytes never appear in stdout
    while the surrounding text survives."""
    base_url = mock_openai_server(_ANSI_CASSETTE)
    config_path = _write_system_config(tmp_path)

    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["KESTREL_OPENROUTER_BASE_URL"] = base_url
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [kestrel_executable, "--config", str(config_path)],
        input="hello\n/quit\n",
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert "\x9b" not in result.stdout
    assert "\x07" not in result.stdout
    assert "before" in result.stdout
    assert "after" in result.stdout
