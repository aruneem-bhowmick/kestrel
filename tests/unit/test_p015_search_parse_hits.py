"""Unit test for `search`'s defense against `rg` output that does not
match the `path:line:text` shape this tool relies on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kestrel.tools.search import SearchArgs, SearchError, search

pytestmark = [pytest.mark.p015, pytest.mark.unit]


def test_unparseable_line_number_raises_search_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `rg` output whose line-number segment is not a plain
    integer, when parsed, then `SearchError` names the offending line
    and is chained from the original `ValueError` -- never a raw
    `ValueError` escaping to the caller."""

    def _fake_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["rg"],
            returncode=0,
            stdout="a.py:not-a-number:some text\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(SearchError, match="could not parse") as exc_info:
        search(SearchArgs(pattern="needle"), repo_root=tmp_path)

    assert isinstance(exc_info.value.__cause__, ValueError)
