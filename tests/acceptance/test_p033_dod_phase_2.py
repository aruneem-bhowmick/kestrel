"""Acceptance suite encoding every machine-checkable clause the agent
loop's verification, caching, and budget machinery converge on once
`kestrel run` itself wires them all together: a multi-file task only
completes once a real `verify` call has actually passed against the
fixture repo's own files, a multi-turn session against a cache-capable
backend reports a cache-hit ratio at or above 50%, and a session budget
soft cap visibly degrades a running task to a cheaper model while a hard
cap halts it outright -- recoverable afterward via `kestrel run --resume`
against the very same journal the halted run left behind.

Every scenario here drives the packaged `kestrel` console script as a
real subprocess against the hermetic mock backend (see
``tests/fixtures/mock_openai.py``); none of it depends on a live model
or a live credential. The verification scenario also drives the actual
`bwrap` sandbox to run a real `pytest` invocation against real files on
disk -- skipped where `bwrap` is unavailable, exactly like
`tests/acceptance/test_p023_dod_phase_1.py`'s own end-to-end scenario.
The one clause that needs a real cache-capable provider call has its own
budget-capped, opt-in twin in ``tests/e2e/test_p033_dod_live.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.managers.session import load_session
from kestrel.tools.sandbox import bwrap_available

pytestmark = [
    pytest.mark.p033,
    pytest.mark.acceptance,
    pytest.mark.system,
    pytest.mark.dod_phase_2,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_TOOLCALL_EDIT_FAREWELL = _CASSETTES / "toolcall_edit_farewell.sse"
_TOOLCALL_VERIFY = _CASSETTES / "toolcall_verify.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_CACHE_HIT_TURN1 = _CASSETTES / "cache_hit_turn1_cold.sse"
_CACHE_HIT_TURN2 = _CASSETTES / "cache_hit_turn2_warm.sse"
_CACHE_HIT_TURN3 = _CASSETTES / "cache_hit_turn3_done.sse"
_BUDGET_TOOLCALL_BIG = _CASSETTES / "budget_toolcall_big.sse"
_BUDGET_DONE_SMALL = _CASSETTES / "budget_done_small.sse"

_TIMEOUT_S = 60.0
_TASK_ID_RE = re.compile(r"^task_id: (?P<task_id>\S+)$", re.MULTILINE)
_CACHE_HIT_RE = re.compile(r"^cache_hit: (?P<pct>\d+)%$", re.MULTILINE)
_RESUME_HINT_RE = re.compile(
    r"kestrel run --resume (?P<task_id>\S+) --repo (?P<repo>\S+)$", re.MULTILINE
)

# `budget_toolcall_big.sse` bills 500,000 prompt tokens a turn, deliberately
# sized to cross a small USD cap on a clean dollar amount (see
# `_write_budget_config`) -- comfortably past `LoopLimits`'s own default
# 200,000-token cap after a single turn, so every budget scenario below
# raises `--max-total-tokens` well past what it actually spends, keeping
# `TOKEN_CAP` from tripping before the budget check ever gets a chance to.
_BUDGET_MAX_TOTAL_TOKENS = "100000000"

_GREET_STUB = "# TODO: implement greet\n"
_FAREWELL_STUB = "# TODO: implement farewell\n"
_MULTI_TEST = (
    "from greet import greet\n"
    "from farewell import farewell\n"
    "\n\n"
    "def test_greet() -> None:\n"
    '    assert greet("World") == "Hello, World!"\n'
    "\n\n"
    "def test_farewell() -> None:\n"
    '    assert farewell("World") == "Goodbye, World!"\n'
)


def _write_run_config(config_dir: Path) -> Path:
    """Write a single-entry ``kestrel.toml`` + ``models.toml`` pair,
    identical in shape to
    ``tests/acceptance/test_p023_dod_phase_1.py``'s own helper -- every
    scenario here redirects the entry's base URL to the hermetic mock
    server via the ``KESTREL_OPENROUTER_BASE_URL`` seam (see
    ``_run_env``), never through this file."""
    models_toml = config_dir / "models.toml"
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

    kestrel_toml = config_dir / "kestrel.toml"
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


def _write_budget_config(config_dir: Path) -> Path:
    """Write a two-entry ``kestrel.toml`` + ``models.toml`` pair: the
    main ``glm-5.2`` entry plus a ``"cheap"``-tagged ``glm-5.2-cheap``
    entry, both priced at round per-token rates (`$1.00`/`$0.10` per
    Mtok input, `$0` output and cached) so a scripted turn's own huge
    `usage.prompt_tokens` (see `budget_toolcall_big.sse`) crosses a small
    USD cap on a clean, hand-verifiable dollar amount, mirroring
    ``tests/unit/test_p031_budget_wiring.py``'s own convention."""
    config_dir.mkdir(parents=True, exist_ok=True)
    models_toml = config_dir / "models.toml"
    models_toml.write_text(
        """\
[[models]]
id = "glm-5.2"
backend = "openrouter"
provider_model = "z-ai/glm-5.2"
api_key_env = "OPENROUTER_API_KEY"
context_window = 100000000
max_output = 16384
usd_per_mtok_input = 1.00
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = true
supports_cache = true
tags = ["planner", "executor"]

[[models]]
id = "glm-5.2-cheap"
backend = "openrouter"
provider_model = "z-ai/glm-5.2-air"
api_key_env = "OPENROUTER_API_KEY"
context_window = 100000000
max_output = 16384
usd_per_mtok_input = 0.10
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = true
supports_cache = true
tags = ["cheap"]
""",
        encoding="utf-8",
    )

    kestrel_toml = config_dir / "kestrel.toml"
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


def _run_env(openrouter_base: str) -> dict[str, str]:
    """Build the subprocess environment for a `kestrel run` call against
    the hermetic mock backend -- identical in shape to
    ``tests/acceptance/test_p023_dod_phase_1.py``'s own helper."""
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


def _render_verification_reports(repo_dir: Path) -> str:
    """Render every persisted `.kestrel/artifacts/*.md` verification
    report under `repo_dir`, for a failure message: a real `verify`
    call's own sandboxed command can fail for reasons entirely outside
    this test's own logic (a broken host/CI sandbox environment, most
    notably), and a bare `returncode != 0` assertion gives no way to
    tell that apart from a genuine regression in the verification-gate
    behavior this test exists to check. Returns a placeholder string
    when no artifacts directory exists at all."""
    artifacts_dir = repo_dir / ".kestrel" / "artifacts"
    if not artifacts_dir.exists():
        return "(no .kestrel/artifacts directory)"
    return "\n".join(
        f"--- {path} ---\n{path.read_text(encoding='utf-8')}"
        for path in sorted(artifacts_dir.glob("*.md"))
    )


@pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH")
def test_dod_verification_is_the_exit_criterion(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a fixture repo with two stub modules, a test file exercising
    both, and a KESTREL.md configuring `test = "pytest -q"`, and a
    scripted task that edits both modules, declares done WITHOUT calling
    `verify` first, then calls `verify` (which passes against the real
    files on disk), then declares done again, when `kestrel run` executes
    with verification required by default (no `--require-verification`
    override), then all five scripted turns are actually consumed by
    real HTTP calls -- proving the first "done" attempt did not end the
    task and the loop instead re-entered Think -- the run exits 0,
    prints `TASK_COMPLETE`, both modules carry their real implementation
    on disk, and a passing verification report was persisted under
    `.kestrel/artifacts/`.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")
    (repo_dir / "farewell.py").write_text(_FAREWELL_STUB, encoding="utf-8")
    (repo_dir / "test_multi.py").write_text(_MULTI_TEST, encoding="utf-8")
    (repo_dir / "KESTREL.md").write_text(
        '```kestrel-verify\ntest = "pytest -q"\n```\n', encoding="utf-8"
    )

    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _TOOLCALL_EDIT_FAREWELL,
            _DONE_CASSETTE,
            _TOOLCALL_VERIFY,
            _DONE_CASSETTE,
        ],
        capture=captured,
    )
    config_path = _write_run_config(tmp_path)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "implement greet and farewell, run the tests via verify, "
            "and only declare done once verify has passed",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}\n"
        f"{_render_verification_reports(repo_dir)}"
    )
    assert "TASK_COMPLETE" in result.stdout
    assert len(captured) == 5, (
        "the premature 'done' attempt must not have ended the task"
    )
    assert "def greet" in (repo_dir / "greet.py").read_text(encoding="utf-8")
    assert "def farewell" in (repo_dir / "farewell.py").read_text(encoding="utf-8")

    artifacts_dir = repo_dir / ".kestrel" / "artifacts"
    reports = list(artifacts_dir.glob("verification-*.md"))
    assert len(reports) == 1
    assert "# Verification: PASSED" in reports[0].read_text(encoding="utf-8")


def test_dod_cache_hit_ratio_meets_threshold(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a fixture repo with two modules and a scripted three-turn
    task whose cassettes carry realistic `cached_tokens` values -- a cold
    first turn, then two turns at an 80% hit rate simulating a real
    cache-capable backend after the first turn establishes the stable
    prefix -- when `kestrel run` executes, then the printed summary's
    `cache_hit` line reports an aggregate ratio at or above 50%, with no
    trailing low-cache-hit alert text on that line.
    """
    repo_dir = tmp_path / "repo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "greet.py").write_text("def greet(): ...\n", encoding="utf-8")
    (repo_dir / "src" / "farewell.py").write_text(
        "def farewell(): ...\n", encoding="utf-8"
    )

    base_url = mock_openai_server(
        cassette_sequence=[_CACHE_HIT_TURN1, _CACHE_HIT_TURN2, _CACHE_HIT_TURN3]
    )
    config_path = _write_run_config(tmp_path)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "read both modules under src/ and summarize them",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "TASK_COMPLETE" in result.stdout

    match = _CACHE_HIT_RE.search(result.stdout)
    assert match is not None, result.stdout
    assert int(match["pct"]) >= 50


def test_dod_budget_soft_degrades_and_hard_halts_then_resumes(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Two scenarios against the same two-entry registry (`glm-5.2` plus
    a `"cheap"`-tagged `glm-5.2-cheap`), both scripted with the same
    deliberately huge-`prompt_tokens` tool-call cassette so two turns
    cross a small USD cap on a clean dollar amount:

    Given a session cap crossed into SOFT (but not HARD) on turn 2, when
    `kestrel run` executes, then it completes `TASK_COMPLETE`, a degrade
    warning naming the soft cap fires on stderr, and the session journal
    shows turn 3 priced at the cheap entry's own rate while turns 1-2
    stayed on the original entry.

    Given a session cap crossed into HARD on turn 2 instead, when
    `kestrel run` executes, then it exits 1, prints an abbreviated
    summary naming `BUDGET_HALT` and the exact `kestrel run --resume`
    invocation to continue, and running that exact invocation (with the
    cap raised) against the remaining scripted cassette completes
    `TASK_COMPLETE`.
    """
    # --- soft cap: degrades on turn 3, still completes ---
    soft_repo = tmp_path / "repo-soft"
    (soft_repo / "src").mkdir(parents=True)
    (soft_repo / "src" / "greet.py").write_text("# marker\n", encoding="utf-8")

    soft_base_url = mock_openai_server(
        cassette_sequence=[
            _BUDGET_TOOLCALL_BIG,
            _BUDGET_TOOLCALL_BIG,
            _BUDGET_DONE_SMALL,
        ]
    )
    soft_config = _write_budget_config(tmp_path / "cfg-soft")

    soft_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "read src/greet.py twice then stop",
            "--repo",
            str(soft_repo),
            "--config",
            str(soft_config),
            "--no-require-verification",
            "--session-budget-usd",
            "1.20",
            "--max-total-tokens",
            _BUDGET_MAX_TOTAL_TOKENS,
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(soft_base_url),
        cwd=soft_repo,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert soft_result.returncode == 0, soft_result.stderr
    assert "TASK_COMPLETE" in soft_result.stdout
    assert "budget soft cap reached" in soft_result.stderr
    assert "degrading to" in soft_result.stderr

    soft_match = _TASK_ID_RE.search(soft_result.stdout)
    assert soft_match is not None, soft_result.stdout
    soft_state = load_session(soft_repo, soft_match["task_id"])
    assert soft_state.turns_used == 3
    assert [turn.model_id for turn in soft_state.turns] == [
        "glm-5.2",
        "glm-5.2",
        "glm-5.2-cheap",
    ]
    assert soft_state.degraded is True

    # --- hard cap: halts on turn 2, then resumes to completion ---
    hard_repo = tmp_path / "repo-hard"
    (hard_repo / "src").mkdir(parents=True)
    (hard_repo / "src" / "greet.py").write_text("# marker\n", encoding="utf-8")

    hard_base_url = mock_openai_server(
        cassette_sequence=[_BUDGET_TOOLCALL_BIG, _BUDGET_TOOLCALL_BIG]
    )
    hard_config = _write_budget_config(tmp_path / "cfg-hard")

    hard_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "read src/greet.py twice then stop",
            "--repo",
            str(hard_repo),
            "--config",
            str(hard_config),
            "--no-require-verification",
            "--session-budget-usd",
            "0.90",
            "--max-total-tokens",
            _BUDGET_MAX_TOTAL_TOKENS,
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(hard_base_url),
        cwd=hard_repo,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert hard_result.returncode == 1, hard_result.stderr
    assert "reason: BUDGET_HALT" in hard_result.stdout
    resume_match = _RESUME_HINT_RE.search(hard_result.stdout)
    assert resume_match is not None, hard_result.stdout
    assert resume_match["task_id"] not in (None, "")

    resumed_base_url = mock_openai_server(cassette_sequence=[_BUDGET_DONE_SMALL])
    resume_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "--resume",
            resume_match["task_id"],
            "--repo",
            str(hard_repo),
            "--config",
            str(hard_config),
            "--no-require-verification",
            "--session-budget-usd",
            "100",
            "--max-total-tokens",
            _BUDGET_MAX_TOTAL_TOKENS,
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(resumed_base_url),
        cwd=hard_repo,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert resume_result.returncode == 0, resume_result.stderr
    assert "TASK_COMPLETE" in resume_result.stdout
