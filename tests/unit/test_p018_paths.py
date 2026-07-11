"""Tests for `kestrel.tools._paths.resolve_repo_path`: the containment
guard shared by `read_file` and `edit_file`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.tools._paths import resolve_repo_path

pytestmark = [pytest.mark.p018, pytest.mark.unit]


@pytest.mark.sanity
def test_resolves_an_in_repo_relative_path(tmp_path: Path) -> None:
    """Given a plain relative path to a file that exists under
    `repo_root`, when resolved, then the exact absolute location of
    that file is returned."""
    (tmp_path / "a.txt").write_text("content", encoding="utf-8")

    resolved = resolve_repo_path("a.txt", repo_root=tmp_path)

    assert resolved == (tmp_path / "a.txt").resolve()


def test_resolves_a_path_with_no_file_on_disk_yet(tmp_path: Path) -> None:
    """Given a relative path naming a file that does not exist, when
    resolved, then the candidate location is still returned rather than
    raising -- this helper only checks containment, never existence."""
    resolved = resolve_repo_path("does-not-exist.txt", repo_root=tmp_path)

    assert resolved == (tmp_path / "does-not-exist.txt").resolve()


@pytest.mark.sanity
def test_dotdot_climb_outside_repo_root_raises_value_error(tmp_path: Path) -> None:
    """Given a path that climbs above `repo_root` with `..`, when
    resolved, then `ValueError` names the escape rather than returning
    a location outside the root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="escapes the repository root"):
        resolve_repo_path("../secret.txt", repo_root=repo_root)


def test_symlink_escaping_repo_root_raises_value_error(tmp_path: Path) -> None:
    """Given a symlink inside `repo_root` pointing at a location
    outside it, when resolved, then `ValueError` is raised -- following
    the symlink before the containment check must not let a request
    escape the root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside the repo", encoding="utf-8")
    link = repo_root / "link.txt"

    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable in this environment: {exc}")

    with pytest.raises(ValueError, match="escapes the repository root"):
        resolve_repo_path("link.txt", repo_root=repo_root)


def test_error_message_names_the_offending_path(tmp_path: Path) -> None:
    """Given an escaping path, when resolved, then the raised message
    includes the exact path string the caller passed in, not just a
    generic refusal."""
    with pytest.raises(ValueError, match=r"\.\./elsewhere\.txt"):
        resolve_repo_path("../elsewhere.txt", repo_root=tmp_path)
