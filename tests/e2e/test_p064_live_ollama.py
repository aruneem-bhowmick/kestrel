"""Live smoke test: `kestrel doctor --live`'s `ollama` check against a
real, locally running Ollama server.

Unlike the OpenRouter and Z.ai live smoke tests elsewhere in this suite,
this one spends nothing -- Ollama is local, and the entry this test's
own config names for it prices at $0 by registry construction -- so
there is no budget ceiling to assert here the way there is in
``test_p011_dod_live.py``. The config's own default model is an
unrelated, deliberately unreachable stand-in entry, present only so the
``default-model``/``api-key`` checks upstream of ``ollama`` in doctor's
own dependency chain pass and unblock it; this test does not care what
the ``endpoint`` check itself reports.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p064, pytest.mark.e2e, pytest.mark.live]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_STAND_IN_KEY_ENV = "KESTREL_E2E_STAND_IN_KEY"
_TIMEOUT_S = 30.0

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1, with a real Ollama server running locally and "
    "reachable at http://localhost:11434 with the nomic-embed-text model "
    "pulled, to run the live Ollama doctor smoke test"
)

_MODELS_TOML = """\
[[models]]
id = "stand-in-default"
backend = "zai"
provider_model = "unused"
endpoint = "http://127.0.0.1:1"
api_key_env = "KESTREL_E2E_STAND_IN_KEY"
context_window = 8192
max_output = 1024
usd_per_mtok_input = 0
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = true
supports_cache = false

[[models]]
id = "nomic-embed-text"
backend = "ollama"
provider_model = "nomic-embed-text"
endpoint = "http://localhost:11434"
context_window = 8192
max_output = 1
usd_per_mtok_input = 0
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = false
supports_cache = false
tags = ["local"]
"""


def _write_config(tmp_path: Path) -> Path:
    """Write a temp `kestrel.toml` + `models.toml` pair naming a real,
    local Ollama embedding entry alongside the unreachable stand-in
    default model described in this module's own docstring."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(_MODELS_TOML, encoding="utf-8")
    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "stand-in-default"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


@pytest.mark.skipif(os.environ.get(_LIVE_TESTS_ENV) != "1", reason=_SKIP_REASON)
def test_live_ollama_check_is_ok_against_a_real_local_server(
    tmp_path: Path, kestrel_executable: str
) -> None:
    """Given a real Ollama server running locally with the
    `nomic-embed-text` model available, when `kestrel doctor --live` runs
    against a config naming it under the router's own `"local"` tag,
    then the `ollama` line reports OK."""
    config_path = _write_config(tmp_path)
    env = dict(os.environ)
    env[_STAND_IN_KEY_ENV] = "unused"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [kestrel_executable, "doctor", "--config", str(config_path), "--live"],
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=_TIMEOUT_S,
        check=False,
    )

    lines = {line.split(None, 2)[1]: line for line in result.stdout.splitlines()}
    assert "ollama" in lines, result.stdout
    assert lines["ollama"].startswith("OK"), result.stdout
