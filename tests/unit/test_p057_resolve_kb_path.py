"""Tests for `resolve_kb_path`'s per-repo and global path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.kb import store as kb_store
from kestrel.kb.store import resolve_kb_path

pytestmark = [pytest.mark.p057, pytest.mark.unit, pytest.mark.sanity]


def test_per_repo_path_lands_under_dot_kestrel(tmp_path: Path) -> None:
    """Given a repo root and `global_=False`, when resolved, then the
    path is `repo_root/.kestrel/kb.sqlite3`."""
    assert (
        resolve_kb_path(tmp_path, global_=False) == tmp_path / ".kestrel" / "kb.sqlite3"
    )


def test_global_path_lands_under_platformdirs_user_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given `global_=True`, when resolved, then the path lands under
    `platformdirs.user_data_dir("kestrel")` -- monkeypatched to a
    `tmp_path` so this test never touches a real per-user directory."""
    user_data_dir = tmp_path / "userdata"
    monkeypatch.setattr(
        kb_store.platformdirs,
        "user_data_dir",
        lambda appname: str(user_data_dir),  # noqa: ARG005
    )

    assert resolve_kb_path(tmp_path, global_=True) == user_data_dir / "kb.sqlite3"
