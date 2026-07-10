"""Unit test for `search`'s timeout handling: a monkeypatched `rg`
invocation that never returns must not let a raw
`subprocess.TimeoutExpired` escape to the caller.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from kestrel.tools.search import SearchArgs, SearchError, search

pytestmark = [pytest.mark.p015, pytest.mark.unit]

# `kestrel.tools.__init__` does `from kestrel.tools.search import search`,
# which rebinds the `search` *attribute* on the `kestrel.tools` package to
# that function -- so `import kestrel.tools.search as search_module` would
# resolve to the function, not the module, once `kestrel.tools` has been
# imported anywhere. `importlib.import_module` reads `sys.modules`
# directly and is immune to that shadowing.
_search_module = importlib.import_module("kestrel.tools.search")


def test_rg_timeout_raises_search_error_naming_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given an `rg` invocation that does not finish within the
    configured bound, when searched, then `SearchError` is raised naming
    that bound and chained from the original `TimeoutExpired` -- never a
    raw `subprocess.TimeoutExpired` escaping to the caller."""
    monkeypatch.setattr(_search_module, "_RG_TIMEOUT_S", 5.0)

    def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["rg"], timeout=5.0)

    monkeypatch.setattr(subprocess, "run", _timeout)

    with pytest.raises(SearchError, match="did not finish within 5s") as exc_info:
        search(SearchArgs(pattern="needle"), repo_root=tmp_path)

    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)
