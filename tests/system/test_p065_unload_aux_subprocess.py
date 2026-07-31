"""System test: `kestrel unload-aux` driven as a real subprocess against
the packaged console script and a hermetic mock Ollama server -- proving
the CLI's own config/registry loading, `"embed"` tag resolution, and
`unload_aux_model` call wire together correctly end to end through the
exact command an operator would type, not merely each piece in
isolation (already covered separately by
`tests/unit/test_p065_unload_aux_cli.py` and
`tests/integration/test_p064_unload_aux.py`).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p065, pytest.mark.system]

_TIMEOUT_S = 30.0


def _write_unload_aux_config(config_dir: Path, *, ollama_base_url: str) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    Ollama-backed, `"local"`-tagged entry pointed at `ollama_base_url` as
    the registry's sole entry and the configured default model --
    everything `kestrel unload-aux` needs to resolve and unload it, and
    nothing else, since this subcommand never talks to a chat backend.
    Returns the `kestrel.toml` path.
    """
    models_toml = config_dir / "models.toml"
    models_toml.write_text(
        f"""\
[[models]]
id = "nomic-embed-text"
backend = "ollama"
provider_model = "nomic-embed-text"
endpoint = "{ollama_base_url}"
context_window = 8192
max_output = 1
usd_per_mtok_input = 0
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = false
supports_cache = false
tags = ["local"]
""",
        encoding="utf-8",
    )

    kestrel_toml = config_dir / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "nomic-embed-text"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def _run_env() -> dict[str, str]:
    """Build the subprocess environment for a `kestrel unload-aux` call.

    No provider credentials are needed: this subcommand's only network
    call is the unauthenticated Ollama unload request, and
    `KESTREL_CONFIG` is cleared so it never overrides the `--config`
    path each test passes explicitly.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


def test_unload_aux_cli_succeeds_against_a_real_mock_ollama_server(
    tmp_path: Path,
    mock_ollama_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a fixture registry naming an Ollama-backed, `"local"`-tagged
    entry pointed at a mock Ollama server that accepts the request, when
    `kestrel unload-aux --config <fixture>` runs as a real subprocess,
    then it exits 0 and prints exactly `"unloaded nomic-embed-text"`."""
    base_url = mock_ollama_server(embeddings=[])
    config_path = _write_unload_aux_config(tmp_path, ollama_base_url=base_url)

    result = subprocess.run(
        [kestrel_executable, "unload-aux", "--config", str(config_path)],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(),
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unloaded nomic-embed-text"


def test_unload_aux_cli_fails_against_a_server_returning_500(
    tmp_path: Path,
    mock_ollama_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given the same fixture registry shape pointed instead at a mock
    Ollama server that fails every request with a 500, when `kestrel
    unload-aux --config <fixture>` runs, then it exits 1, prints nothing
    to stdout, and reports the offending entry's own id on stderr."""
    base_url = mock_ollama_server(status_code=500)
    config_path = _write_unload_aux_config(tmp_path, ollama_base_url=base_url)

    result = subprocess.run(
        [kestrel_executable, "unload-aux", "--config", str(config_path)],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(),
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "nomic-embed-text" in result.stderr
