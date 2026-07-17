"""Unit tests for the `kestrel doctor` flight checks: dependency chaining
between checks, credential redaction, and the rendered output contract.

Every check that touches the filesystem or the environment runs against
an isolated temp directory and a cleared/monkeypatched environment, so
none of these tests can read (or be polluted by) the developer's real
config, registry, or credentials.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import litellm
import pytest

import kestrel.doctor as doctor_module
from kestrel.doctor import (
    CheckResult,
    CheckStatus,
    all_checks_passed,
    format_check_line,
    render_report,
    run_doctor,
)
from kestrel.tools.sandbox import SandboxResult, SandboxUnavailableError

pytestmark = [pytest.mark.p009, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p009_doctor_output.golden"
)

_VALID_MODELS_TOML = """\
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
"""

_BROKEN_MODELS_TOML = """\
[[models]]
id = "broken"
backend = "zai"
provider_model = "glm-5.2"
api_key_env = "ZAI_API_KEY"
context_window = 1000
max_output = 100
usd_per_mtok_input = 1.0
usd_per_mtok_output = 2.0
usd_per_mtok_cached = 0.5
supports_tools = true
supports_cache = false
"""


def _names(results: list[CheckResult]) -> list[str]:
    """Extract just the ordered check names from a result list."""
    return [result.name for result in results]


def _statuses(results: list[CheckResult]) -> dict[str, CheckStatus]:
    """Index a result list by check name for status lookups."""
    return {result.name: result.status for result in results}


def _patch_sandbox_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ``sandbox`` check deterministically ``OK``, standing in
    for a real ``bwrap``-equipped runner without depending on one being
    available wherever this test suite happens to run."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: True)
    monkeypatch.setattr(
        doctor_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout="", stderr="", exit_code=0, timed_out=False
        ),
    )


def _patch_tui_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ``tui`` check deterministically ``OK``, standing in for a
    real interactive terminal -- pytest's own captured stdout is never
    one, so a test asserting an all-green run must pin this explicitly
    rather than depend on however this suite happens to be invoked."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


@pytest.mark.sanity
def test_all_green_non_live_run_passes_seven_and_skips_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given a valid config, a valid registry, the default model present,
    its credential set, a sandbox-capable environment, and a real
    (simulated) interactive terminal, when run without ``--live``, then
    checks 1-5, ``sandbox``, and ``tui`` are OK, the remaining two
    (``endpoint``, ``ollama``) are SKIP, and the run counts as passing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    _patch_sandbox_ok(monkeypatch)
    _patch_tui_ok(monkeypatch)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    results = run_doctor(config_path, live=False)

    assert _names(results) == [
        "python-version",
        "config",
        "registry",
        "default-model",
        "api-key",
        "endpoint",
        "sandbox",
        "tui",
        "ollama",
    ]
    statuses = _statuses(results)
    for name in (
        "python-version",
        "config",
        "registry",
        "default-model",
        "api-key",
        "sandbox",
        "tui",
    ):
        assert statuses[name] is CheckStatus.OK
    for name in ("endpoint", "ollama"):
        assert statuses[name] is CheckStatus.SKIP
    assert all_checks_passed(results) is True


def test_sandbox_check_is_unconditional_even_when_config_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a config that fails to load, when run, then the ``sandbox``
    check still runs and reports its own real outcome rather than
    joining the ``config``-rooted chain of ``SKIP``s -- it depends on
    nothing upstream."""
    _patch_sandbox_ok(monkeypatch)
    missing = tmp_path / "missing.toml"

    results = run_doctor(missing, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["config"].status is CheckStatus.FAIL
    assert by_name["sandbox"].status is CheckStatus.OK


# --- pure helper coverage: the sandbox check ---------------------------------


def test_check_sandbox_fails_naming_bwrap_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no ``bwrap`` on ``PATH``, when the sandbox check runs, then
    it FAILs naming that absence, without attempting to run anything."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: False)

    def _unexpected_call(*_args: object, **_kwargs: object) -> SandboxResult:
        """Stand in for `run_sandboxed`, failing the test if reached."""
        raise AssertionError("run_sandboxed should not be called when bwrap is absent")

    monkeypatch.setattr(doctor_module, "run_sandboxed", _unexpected_call)

    result = doctor_module._check_sandbox()

    assert result.status is CheckStatus.FAIL
    assert result.detail == "bwrap not found on PATH"


def test_check_sandbox_fails_naming_a_nonzero_smoke_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ``bwrap`` present but the smoke invocation exiting non-zero,
    when the sandbox check runs, then it FAILs naming the exit code and
    the invocation's stderr."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: True)
    monkeypatch.setattr(
        doctor_module,
        "run_sandboxed",
        lambda *_args, **_kwargs: SandboxResult(
            stdout="",
            stderr="bwrap: setting up uid map: Permission denied",
            exit_code=1,
            timed_out=False,
        ),
    )

    result = doctor_module._check_sandbox()

    assert result.status is CheckStatus.FAIL
    assert "1" in result.detail
    assert "Permission denied" in result.detail


def test_check_sandbox_fails_naming_a_sandbox_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ``bwrap`` reported present but the smoke invocation itself
    raising ``SandboxUnavailableError`` (e.g. a race between the presence
    check and the real invocation), when the sandbox check runs, then it
    FAILs naming that error rather than letting it escape."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: True)

    def _raise(*_args: object, **_kwargs: object) -> SandboxResult:
        """Stand in for `run_sandboxed`, raising as if `bwrap` vanished
        between the presence check and this call."""
        raise SandboxUnavailableError("bwrap disappeared mid-check")

    monkeypatch.setattr(doctor_module, "run_sandboxed", _raise)

    result = doctor_module._check_sandbox()

    assert result.status is CheckStatus.FAIL
    assert result.detail == "bwrap disappeared mid-check"


def test_check_sandbox_fails_naming_an_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ``bwrap`` reported present but the smoke invocation itself
    raising ``OSError``, when the sandbox check runs, then it
    FAILs naming that error rather than letting it escape."""
    monkeypatch.setattr(doctor_module, "bwrap_available", lambda: True)

    def _raise(*_args: object, **_kwargs: object) -> SandboxResult:
        raise OSError("Permission denied or missing binary")

    monkeypatch.setattr(doctor_module, "run_sandboxed", _raise)

    result = doctor_module._check_sandbox()

    assert result.status is CheckStatus.FAIL
    assert result.detail == "Permission denied or missing binary"


def test_check_sandbox_ok_when_bwrap_available_and_smoke_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given ``bwrap`` present and a zero-exit smoke invocation, when the
    sandbox check runs, then it reports OK naming ``bwrap``."""
    _patch_sandbox_ok(monkeypatch)

    result = doctor_module._check_sandbox()

    assert result == CheckResult("sandbox", CheckStatus.OK, "bwrap")


@pytest.mark.sanity
def test_broken_registry_fails_and_cascades_skips_naming_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given a config that resolves fine but points at a models.toml with
    a zai entry missing its required endpoint, when run, then the
    registry check FAILs with the registry loader's own message, and
    every check after it SKIPs naming "registry" as the blocker."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = write_config(tmp_path, _BROKEN_MODELS_TOML, default_model="glm-5.2")

    results = run_doctor(config_path, live=False)

    statuses = _statuses(results)
    by_name = {result.name: result for result in results}
    assert statuses["config"] is CheckStatus.OK
    assert statuses["registry"] is CheckStatus.FAIL
    assert "endpoint" in by_name["registry"].detail
    for name in ("default-model", "api-key", "endpoint"):
        assert statuses[name] is CheckStatus.SKIP
        assert by_name[name].detail == "blocked by: registry"


@pytest.mark.sanity
def test_missing_api_key_env_fails_without_leaking_a_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given every check up through default-model resolution succeeds but
    OPENROUTER_API_KEY is unset, when run, then the api-key check FAILs
    naming the variable, its detail carries no credential-shaped value,
    and the endpoint check SKIPs blocked by it."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    results = run_doctor(config_path, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["api-key"].status is CheckStatus.FAIL
    assert "OPENROUTER_API_KEY" in by_name["api-key"].detail
    assert "sk-" not in by_name["api-key"].detail
    assert by_name["endpoint"].status is CheckStatus.SKIP
    assert by_name["endpoint"].detail == "blocked by: api-key"


@pytest.mark.sanity
def test_unknown_default_model_fails_listing_available_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given a valid registry that simply does not contain the configured
    default model id, when run, then the default-model check FAILs
    listing every id that *is* available, and downstream checks SKIP
    naming it as the blocker."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="nope")

    results = run_doctor(config_path, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["default-model"].status is CheckStatus.FAIL
    assert "nope" in by_name["default-model"].detail
    assert "glm-5.2" in by_name["default-model"].detail
    assert by_name["api-key"].status is CheckStatus.SKIP
    assert by_name["api-key"].detail == "blocked by: default-model"


def test_endpoint_skip_reason_is_blocked_by_when_both_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given both a missing credential and ``--live``, when run, then the
    endpoint check reports the blocking check rather than defaulting to
    the "pass --live" reason -- a real cause always outranks the generic
    opt-in hint."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    results = run_doctor(config_path, live=True)

    by_name = {result.name: result for result in results}
    assert by_name["endpoint"].detail == "blocked by: api-key"


def test_live_probe_fails_cleanly_instead_of_hanging_on_a_stalled_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[..., Path],
) -> None:
    """Given a backend that never returns from the underlying completion
    call, when the live probe runs, then it fails with a bounded timeout
    detail instead of hanging `kestrel doctor --live` indefinitely."""
    monkeypatch.setattr(doctor_module, "_LIVE_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    config_path = write_config(tmp_path, _VALID_MODELS_TOML, default_model="glm-5.2")

    async def _hang(**_kwargs: Any) -> Any:
        """Stand in for litellm.acompletion, never returning on its own."""
        await asyncio.sleep(10)
        raise AssertionError("the wait_for wrapper should have cancelled this first")

    monkeypatch.setattr(litellm, "acompletion", _hang)

    results = run_doctor(config_path, live=True)

    by_name = {result.name: result for result in results}
    assert by_name["endpoint"].status is CheckStatus.FAIL
    assert "0s" in by_name["endpoint"].detail


def test_missing_config_file_fails_and_blocks_every_downstream_check(
    tmp_path: Path,
) -> None:
    """Given an explicit --config path that does not exist, when run,
    then the config check itself FAILs and every dependent check SKIPs
    naming "config" as the blocker -- not the check nearest to it."""
    missing = tmp_path / "missing.toml"

    results = run_doctor(missing, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["config"].status is CheckStatus.FAIL
    for name in ("registry", "default-model", "api-key", "endpoint"):
        assert by_name[name].status is CheckStatus.SKIP
        assert by_name[name].detail == "blocked by: config"


def test_no_config_anywhere_resolves_to_builtin_defaults(tmp_path: Path) -> None:
    """Given no explicit path, no $KESTREL_CONFIG, and no ./kestrel.toml,
    when run, then the config check still passes, naming built-in
    defaults as its source rather than failing."""
    del tmp_path  # isolation fixture already chdir'd into an empty directory

    results = run_doctor(None, live=False)

    by_name = {result.name: result for result in results}
    assert by_name["config"].status is CheckStatus.OK
    assert by_name["config"].detail == "built-in defaults"


@pytest.mark.regression
@pytest.mark.acceptance
def test_render_report_matches_golden_snapshot() -> None:
    """The exact nine-line block `kestrel doctor` prints for an all-green,
    non-live run must match a pinned snapshot byte-for-byte -- this is the
    stable alignment contract a provisioning walkthrough can screenshot.

    Built from hand-specified results rather than a real config/registry
    file pair run through ``run_doctor``: a real file's resolved path
    would embed OS-specific separators (and a fresh ``tmp_path`` on every
    run) into the very text this test pins, which would make the
    snapshot neither stable nor portable. ``test_all_green_non_live_run_
    passes_seven_and_skips_two`` above already covers the real,
    file-backed path end to end.
    """
    results = [
        CheckResult("python-version", CheckStatus.OK, "3.12"),
        CheckResult("config", CheckStatus.OK, "./kestrel.toml"),
        CheckResult("registry", CheckStatus.OK, "2 models"),
        CheckResult("default-model", CheckStatus.OK, "glm-5.2"),
        CheckResult("api-key", CheckStatus.OK, "OPENROUTER_API_KEY"),
        CheckResult("endpoint", CheckStatus.SKIP, "pass --live"),
        CheckResult("sandbox", CheckStatus.OK, "bwrap"),
        CheckResult("tui", CheckStatus.OK, "interactive"),
        CheckResult(
            "ollama", CheckStatus.SKIP, "the Ollama backend is not implemented"
        ),
    ]

    rendered = render_report(results)

    assert rendered == _GOLDEN_FILE.read_text(encoding="utf-8")


# --- pure helper coverage: python-version and rendering ----------------------


@pytest.mark.parametrize(
    ("version_info", "expected_status"),
    [((3, 12), CheckStatus.OK), ((3, 13), CheckStatus.OK), ((3, 11), CheckStatus.FAIL)],
)
def test_python_version_check_boundary(
    version_info: tuple[int, int], expected_status: CheckStatus
) -> None:
    """Given interpreter versions at, above, and below the 3.12 floor,
    when checked, then only the sub-floor version FAILs."""
    from kestrel.doctor import _check_python_version

    result = _check_python_version(version_info)

    assert result.status is expected_status


def test_format_check_line_aligns_detail_to_a_stable_column() -> None:
    """Given results with the shortest and longest status/name strings,
    when rendered, then both lines' detail text starts at the same
    column -- the whole point of a fixed-width, aligned report."""
    short = format_check_line(CheckResult("api-key", CheckStatus.OK, "X"))
    longest = format_check_line(CheckResult("python-version", CheckStatus.FAIL, "Y"))

    assert short.index("X") == longest.index("Y")


@pytest.mark.redteam
def test_hostile_detail_text_is_sanitized_when_rendered() -> None:
    """Given a check result whose detail echoes a user-controlled string
    containing a raw ANSI escape sequence (as a malformed --config path
    would produce), when rendered, then the escape sequence is stripped
    while the surrounding text survives -- doctor output is routinely
    captured verbatim into CI logs."""
    result = CheckResult(
        "config", CheckStatus.FAIL, "Config file not found: \x1b[31mevil\x1b[0m.toml"
    )

    rendered = format_check_line(result)

    assert "\x1b" not in rendered
    assert "evil" in rendered
    assert "Config file not found" in rendered


@pytest.mark.redteam
def test_hostile_config_path_is_sanitized_end_to_end(tmp_path: Path) -> None:
    """Given an explicit --config path whose filename contains a raw ANSI
    escape sequence, when run through the full doctor pipeline, then the
    escape sequence never survives into the rendered report."""
    hostile = tmp_path / "\x1b[31mmissing\x1b[0m.toml"

    results = run_doctor(hostile, live=False)
    rendered = render_report(results)

    assert "\x1b" not in rendered
    assert "missing" in rendered
