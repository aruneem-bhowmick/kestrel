"""Acceptance suite proving the Definition of Done's own "embeddings
computed locally" clause three ways: two structural facts that make it
true by construction, checked entirely in-process, and one opt-in live
probe against a real, locally running Ollama server -- the one
assertion in this whole suite that actually exercises real embedding
hardware, exactly the bridge between simulated and real-hardware
verification this codebase's own live-suite convention
(`tests/e2e/test_p064_live_ollama.py`, `tests/e2e/test_p011_dod_live.py`)
already establishes for a check no CI runner can perform on its own.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kestrel.doctor import CheckStatus, run_doctor
from kestrel.registry import loader as registry_loader
from kestrel.registry.loader import load_registry

pytestmark = [pytest.mark.p067, pytest.mark.dod_phase_5, pytest.mark.acceptance]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_STAND_IN_KEY_ENV = "KESTREL_E2E_STAND_IN_KEY"
_LIVE_TIMEOUT_S = 30.0

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1, with a real Ollama server running locally and "
    "reachable at http://localhost:11434 with the nomic-embed-text model "
    "pulled, to run the live local-embeddings acceptance case"
)

_LIVE_MODELS_TOML = """\
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


def _isolate_registry_and_config_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chdir into an empty directory, point the shared `platformdirs`
    user-config-dir lookup at an empty temp directory, and clear
    `$KESTREL_CONFIG` -- so `load_config`/`load_registry` resolve to
    their packaged defaults regardless of whatever config or registry
    files happen to exist on the machine actually running this suite.
    `kestrel.config` and `kestrel.registry.loader` both `import
    platformdirs` directly, so patching the attribute on either module's
    own reference patches the single shared module both read from.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        registry_loader.platformdirs,
        "user_config_dir",
        lambda appname: str(tmp_path / "empty-user-config-dir"),
    )
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)


def test_nomic_embed_text_registry_entry_is_ollama_backed_and_tagged_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the packaged default registry (no `models.toml` in any
    layer), when it is loaded, then its `nomic-embed-text` entry is
    `backend="ollama"` and carries the `"local"` tag -- the two
    structural facts that make "embeddings computed locally" true by
    construction, not merely by claim.
    """
    _isolate_registry_and_config_discovery(monkeypatch, tmp_path)

    registry = load_registry()
    entry = registry.get("nomic-embed-text")

    assert entry.backend == "ollama"
    assert "local" in entry.tags


def test_doctor_without_live_reports_ollama_as_skip_never_silently_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the packaged default configuration and registry, and a
    real credential for the default chat model so nothing upstream of
    `ollama` in doctor's own dependency chain blocks it, when `kestrel
    doctor` runs without `--live`, then the `ollama` check reports
    `SKIP "pass --live"` -- it never reports `OK` without a real probe
    having actually run.
    """
    _isolate_registry_and_config_discovery(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    results = run_doctor(None, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["ollama"].status is CheckStatus.SKIP
    assert by_name["ollama"].detail == "pass --live"


def _write_live_config(tmp_path: Path) -> Path:
    """Write a temp `kestrel.toml` + `models.toml` pair naming a real,
    local Ollama embedding entry alongside a deliberately unreachable
    stand-in default model, present only so the `default-model`/
    `api-key` checks upstream of `ollama` in doctor's own dependency
    chain resolve and unblock it -- this test does not care what the
    `endpoint` check itself reports."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(_LIVE_MODELS_TOML, encoding="utf-8")
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


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.skipif(os.environ.get(_LIVE_TESTS_ENV) != "1", reason=_SKIP_REASON)
def test_doctor_live_reports_ollama_ok_against_a_real_local_server(
    tmp_path: Path, kestrel_executable: str
) -> None:
    """Given a real Ollama server running locally with the
    `nomic-embed-text` model available, when `kestrel doctor --live`
    runs against a config naming it under the router's own `"local"`
    tag, then the `ollama` line reports `OK` -- the one assertion in
    this whole suite that actually exercises real embedding hardware,
    rather than a mock standing in for it.
    """
    config_path = _write_live_config(tmp_path)
    env = dict(os.environ)
    env[_STAND_IN_KEY_ENV] = "unused"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [kestrel_executable, "doctor", "--config", str(config_path), "--live"],
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=_LIVE_TIMEOUT_S,
        check=False,
    )

    lines = {line.split(None, 2)[1]: line for line in result.stdout.splitlines()}
    assert "ollama" in lines, result.stdout
    assert lines["ollama"].startswith("OK"), result.stdout
