"""Integration tests for `search`'s real `rg` invocation: hit ordering,
the scope containment guard, invalid-regex handling, the zero-match
case, and `max_results` truncation.

Skipped locally when `rg` is not on `PATH` -- a real local seam (the
binary genuinely may not be installed), not a network one. CI installs
`ripgrep` on every runner, so this suite always actually runs there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kestrel.tools.search import SearchArgs, SearchError, search

pytestmark = [
    pytest.mark.p015,
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("rg") is None, reason="rg not found on PATH"),
]


def _write(root: Path, relative: str, content: str) -> None:
    """Write `content` as UTF-8 bytes to `relative` under `root`, creating
    parent directories as needed."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


@pytest.mark.sanity
def test_known_pattern_returns_expected_hits_in_file_order(tmp_path: Path) -> None:
    """Given a small fixture tree with a pattern matching lines in two
    files, when searched, then both matches are present and the file
    that sorts first by path appears before the one that sorts second."""
    _write(tmp_path, "a_first.py", "alpha\nneedle in a_first\n")
    _write(tmp_path, "b_second.py", "needle in b_second\nbeta\n")

    framed = search(SearchArgs(pattern="needle"), repo_root=tmp_path)

    first_index = framed.index("a_first.py")
    second_index = framed.index("b_second.py")
    assert first_index < second_index
    assert "needle in a_first" in framed
    assert "needle in b_second" in framed


def test_scope_escaping_repo_root_raises(tmp_path: Path) -> None:
    """Given a `scope` that climbs above `repo_root` with `..`, when
    searched, then `SearchError` is raised rather than a location outside
    the repo being searched."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write(tmp_path, "outside.txt", "needle\n")

    with pytest.raises(SearchError, match="escapes the repository root"):
        search(
            SearchArgs(pattern="needle", scope="../outside.txt"),
            repo_root=repo_root,
        )


def test_invalid_regex_raises_naming_rgs_own_message(tmp_path: Path) -> None:
    """Given a pattern that is not a well-formed regex, when searched,
    then `SearchError` is raised carrying `rg`'s own regex-parse
    diagnostic rather than the generic "exited with status N" fallback."""
    _write(tmp_path, "a.py", "content\n")

    with pytest.raises(SearchError) as exc_info:
        search(SearchArgs(pattern="("), repo_root=tmp_path)

    message = str(exc_info.value)
    assert "regex parse error" in message
    assert "exited with status" not in message


def test_zero_matches_returns_a_framed_no_matches_result(tmp_path: Path) -> None:
    """Given a pattern matching nothing in the fixture tree, when
    searched, then the result is a normal framed message stating no
    matches were found, not a raised `SearchError`."""
    _write(tmp_path, "a.py", "content\n")

    framed = search(SearchArgs(pattern="not_present_anywhere"), repo_root=tmp_path)

    assert framed.startswith("<<<UNTRUSTED:search_result:")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert "no matches" in framed.lower()


def test_max_results_truncates_a_fixture_tree_with_more_hits_than_the_cap(
    tmp_path: Path,
) -> None:
    """Given a fixture file with ten matching lines and `max_results`
    set to 3, when searched, then only the first three hits (in file
    order) appear in the result."""
    lines = "\n".join(f"needle {i}" for i in range(10))
    _write(tmp_path, "many.txt", lines + "\n")

    framed = search(SearchArgs(pattern="needle", max_results=3), repo_root=tmp_path)

    assert "needle 0" in framed
    assert "needle 1" in framed
    assert "needle 2" in framed
    assert "needle 3" not in framed


def test_scope_restricts_the_search_to_a_subdirectory(tmp_path: Path) -> None:
    """Given a `scope` naming a subdirectory, when searched, then only
    matches under that subdirectory are returned, even when a match for
    the same pattern exists elsewhere in the repo."""
    _write(tmp_path, "included/hit.py", "needle inside scope\n")
    _write(tmp_path, "excluded/hit.py", "needle outside scope\n")

    framed = search(SearchArgs(pattern="needle", scope="included"), repo_root=tmp_path)

    assert "needle inside scope" in framed
    assert "needle outside scope" not in framed
