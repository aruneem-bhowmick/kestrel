"""Budget-capped live smoke test proving the packaged REPL streams a real
completion, prints a priced cost line, and exits cleanly against a real
backend end to end -- the live counterpart to the mock-backed streaming
and cost-line scenarios in ``tests/acceptance/test_p011_dod_phase_0.py``.

This is the one place in the suite that drives the REPL itself (rather
than the provider client directly) against a real endpoint, so it runs
with no ``--config`` override at all: the packaged default configuration
and registry are exactly what a fresh install resolves to, and are what
this test exercises. Spending discipline mirrors the rest of the live
suite (``tests/e2e/test_p005_live_openrouter.py``,
``tests/e2e/test_p006_live_zai.py``): the scripted prompt asks for the
shortest possible reply, and the metered turn cost is asserted well under
the project's $0.50-per-run policy ceiling. Neither the registry nor the
provider layer enforces a token cap of its own yet, so the budget here is
kept by the prompt plus the hard assertion below, not by a request
parameter.
"""

from __future__ import annotations

import os
import re
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.p011,
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.dod_phase_0,
]

_LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_TIMEOUT_S = 30.0
# Well under the $0.50/run policy ceiling described in the module docstring.
_BUDGET_CEILING_USD = Decimal("0.02")

_SKIP_REASON = (
    f"set {_LIVE_TESTS_ENV}=1 and {_API_KEY_ENV} to run the live REPL smoke test"
)

_TURN_RE = re.compile(
    r"kestrel> (?P<reply>.*?)\n"
    r"in:(?P<input>\d+)(?: \(cached:(?P<cached>\d+)\))? out:(?P<output>\d+) "
    r"· \$(?P<turn_usd>\d+\.\d{4}) turn · \$(?P<session_usd>\d+\.\d{4}) session",
    re.DOTALL,
)


@pytest.mark.skipif(
    os.environ.get(_LIVE_TESTS_ENV) != "1" or not os.environ.get(_API_KEY_ENV),
    reason=_SKIP_REASON,
)
def test_dod_live_repl_streams_and_prices_a_real_completion(
    tmp_path: Path, kestrel_executable: str
) -> None:
    """Given the packaged default configuration and registry -- no
    ``--config`` override, the same defaults a fresh install resolves to
    -- and a real OpenRouter credential, when a REPL script sends one
    short prompt and quits, then the real completion streams non-empty
    text immediately followed by a cost line reporting nonzero usage on
    both sides of the exchange, the metered turn cost stays under the
    budget ceiling, and the process exits cleanly.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)

    script = "Reply with exactly: kestrel\n/quit\n"
    result = subprocess.run(
        [kestrel_executable],
        input=script,
        capture_output=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    match = _TURN_RE.search(result.stdout)
    assert match is not None, result.stdout
    assert match["reply"].strip() != ""
    assert int(match["input"]) > 0
    assert int(match["output"]) > 0

    turn_usd = Decimal(match["turn_usd"])
    assert turn_usd < _BUDGET_CEILING_USD
