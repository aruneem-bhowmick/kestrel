"""Integration tests: the `--live` endpoint probe against a hermetic mock
backend.

The zai backend is used throughout (rather than openrouter) because its
registry entry carries the endpoint to call directly, so pointing a test
at a mock server needs no environment-variable seam -- the same pattern
``tests/integration/test_p006_zai.py`` uses for the underlying client.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.doctor import CheckStatus, run_doctor

pytestmark = [pytest.mark.p009, pytest.mark.integration]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_HELLO_CASSETTE = _CASSETTES / "zai_glm52_hello.sse"


def _write_config(tmp_path: Path, *, endpoint: str) -> Path:
    """Write a temp ``kestrel.toml`` + ``models.toml`` pair naming a
    single zai route pointed at ``endpoint``, and return the config path.
    """
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(
        f"""\
[[models]]
id = "glm-5.2-zai"
backend = "zai"
provider_model = "glm-5.2"
endpoint = "{endpoint}"
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
default_model = "glm-5.2-zai"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def test_live_endpoint_check_is_ok_on_the_hello_cassette(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server replays a successful completion, when doctor
    runs with ``live=True``, then the endpoint check is OK and every
    earlier check also passes."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-zai")
    base_url = mock_openai_server(_HELLO_CASSETTE)
    config_path = _write_config(tmp_path, endpoint=base_url)

    results = run_doctor(config_path, live=True)

    by_name = {result.name: result for result in results}
    assert by_name["endpoint"].status is CheckStatus.OK
    assert "zai" in by_name["endpoint"].detail
    for name in ("python-version", "config", "registry", "default-model", "api-key"):
        assert by_name[name].status is CheckStatus.OK


def test_live_endpoint_check_fails_with_typed_auth_error_on_401(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the mock server rejects the request with 401, when doctor
    runs with ``live=True``, then the endpoint check FAILs naming
    AuthError rather than the run raising or reporting a generic failure."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-zai")
    base_url = mock_openai_server(status_code=401)
    config_path = _write_config(tmp_path, endpoint=base_url)

    results = run_doctor(config_path, live=True)

    by_name = {result.name: result for result in results}
    assert by_name["endpoint"].status is CheckStatus.FAIL
    assert "AuthError" in by_name["endpoint"].detail


def test_non_live_run_never_reaches_the_mock_server(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a server configured to always fail, when doctor runs without
    ``live=True``, then the endpoint check SKIPs and the server is never
    contacted -- non-live doctor must spend nothing and touch no network."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test-zai")
    base_url = mock_openai_server(status_code=500)
    config_path = _write_config(tmp_path, endpoint=base_url)

    results = run_doctor(config_path, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["endpoint"].status is CheckStatus.SKIP
    assert by_name["endpoint"].detail == "pass --live"
