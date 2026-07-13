"""Budget-capped live smoke test proving `kestrel run` completes a real
small task end to end against a real OpenRouter-backed model -- the
live counterpart to the mock-backed task-completion scenario in
``tests/acceptance/test_p023_dod_phase_1.py``.

This is the one place in the suite that drives the tool-calling agent
loop (rather than the plain chat REPL) against a real endpoint, so it
runs with no ``--config`` override at all: the packaged default
configuration and registry are exactly what a fresh install resolves
to, and are what this test exercises. The task is worded to keep the
real model's own tool-call count low (add one docstring to one already-
correct function), and spending discipline mirrors the rest of the live
suite (``tests/e2e/test_p005_live_openrouter.py``,
``tests/e2e/test_p011_dod_live.py``): the hard budget assertion below
keeps every run well inside the project's $0.50-per-run policy ceiling
documented in ``docs-kestrel``'s own acceptance criteria for this
milestone. The invocation passes ``--no-require-verification`` since the
fixture repo carries no KESTREL.md to verify against and this scenario's
own job is proving task completion, not the verification gate.
"""

from __future__ import annotations

import os
import re
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.p023,
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.dod_phase_1,
]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_TIMEOUT_S = 120.0
# Well under the $0.50/run policy ceiling described in the module docstring.
_BUDGET_CEILING_USD = Decimal("0.10")

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1 and {_API_KEY_ENV} to run the live `kestrel run` "
    "smoke test"
)

_GREET_MODULE = 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'
_TOTAL_USD_RE = re.compile(r"^total_usd: \$(?P<usd>\d+\.\d{4})$", re.MULTILINE)


@pytest.mark.skipif(
    os.environ.get(_LIVE_TESTS_ENV) != "1" or not os.environ.get(_API_KEY_ENV),
    reason=_SKIP_REASON,
)
def test_dod_live_run_completes_a_real_small_task(
    tmp_path: Path, kestrel_executable: str
) -> None:
    """Given the packaged default configuration and registry -- no
    ``--config`` override, the same defaults a fresh install resolves to
    -- and a real OpenRouter credential, when `kestrel run` is asked to
    add a docstring to the one function in a tiny fixture module, then
    the process exits 0, reports `TASK_COMPLETE`, and the metered total
    cost of every turn the real model actually took stays under the
    budget ceiling.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_MODULE, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "add a docstring to the one function in greet.py",
            "--repo",
            str(repo_dir),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "TASK_COMPLETE" in result.stdout

    match = _TOTAL_USD_RE.search(result.stdout)
    assert match is not None, result.stdout
    assert Decimal(match["usd"]) < _BUDGET_CEILING_USD
