"""Tests for `kestrel.agent._artifact_paths`: atomic collision-avoiding
allocation of a markdown artifact path and the stem-validation guard
that keeps a caller-supplied stem from ever escaping `artifacts_dir`.

`persist_plan`'s own tests (`tests/unit/test_p048_plan.py`) already
exercise this module's happy path and its directory-level symlink
refusal indirectly; this suite targets `allocate_artifact_path` itself,
including cases that don't route through any one artifact type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.agent._artifact_paths import allocate_artifact_path

pytestmark = [pytest.mark.p048, pytest.mark.unit]


@pytest.mark.sanity
def test_first_call_reserves_the_unsuffixed_path(tmp_path: Path) -> None:
    """Given an empty artifacts directory, when a path is allocated, then
    it is the plain `{stem}.md` name, and that name is already claimed
    (an empty file exists there) by the time the call returns."""
    allocated = allocate_artifact_path(tmp_path, stem="plan-t1")

    assert allocated == tmp_path / "plan-t1.md"
    assert allocated.is_file()


@pytest.mark.sanity
def test_a_second_call_for_the_same_stem_gets_the_numeric_suffix(
    tmp_path: Path,
) -> None:
    """Given a stem already claimed by an earlier call, when allocated
    again, then the `-1`-suffixed name is reserved instead, leaving the
    first file untouched."""
    first = allocate_artifact_path(tmp_path, stem="plan-t1")

    second = allocate_artifact_path(tmp_path, stem="plan-t1")

    assert second == tmp_path / "plan-t1-1.md"
    assert first.is_file()
    assert second.is_file()


def test_a_pre_existing_file_at_the_unsuffixed_name_is_skipped(tmp_path: Path) -> None:
    """Given a file already sitting at `{stem}.md` before this module
    ever ran (not created via `allocate_artifact_path` itself), when
    allocated, then the `-1`-suffixed name is claimed instead -- the
    same outcome as the two-call case, proving the check is against the
    filesystem, not an internal call count."""
    (tmp_path / "plan-t1.md").write_text("pre-existing", encoding="utf-8")

    allocated = allocate_artifact_path(tmp_path, stem="plan-t1")

    assert allocated == tmp_path / "plan-t1-1.md"


def test_a_dangling_symlink_at_the_candidate_name_is_refused_not_followed(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Given a dangling symlink (pointing at a target that does not
    exist) already sitting at the unsuffixed candidate name -- exactly
    what an `exists()`-based check would treat as free, since `exists()`
    reports `False` for a symlink whose target is missing -- when
    allocated, then the symlink is skipped in favor of the `-1`-suffixed
    name rather than followed and written through, which would escape
    `tmp_path` to wherever the dangling target resolves."""
    outside = tmp_path_factory.mktemp("artifact-paths-symlink-target")
    dangling_target = outside / "nonexistent.md"
    (tmp_path / "plan-t1.md").symlink_to(dangling_target)

    allocated = allocate_artifact_path(tmp_path, stem="plan-t1")

    assert allocated == tmp_path / "plan-t1-1.md"
    assert not dangling_target.exists()


def test_allocated_path_is_reserved_empty_not_yet_holding_content(
    tmp_path: Path,
) -> None:
    """Given a fresh stem, when allocated, then the reserved file is
    empty -- `allocate_artifact_path` only claims the name; writing the
    real content is a separate step its caller performs afterward."""
    allocated = allocate_artifact_path(tmp_path, stem="plan-t1")

    assert allocated.read_text(encoding="utf-8") == ""


@pytest.mark.sanity
@pytest.mark.parametrize("bad_stem", ["", "a/b", "a\\b", ".", ".."])
def test_an_unsafe_stem_is_rejected_before_any_file_is_touched(
    tmp_path: Path, bad_stem: str
) -> None:
    """Given a stem that is empty, carries a path separator, or is a
    `.`/`..` traversal segment, when allocated, then `ValueError` names
    the stem and nothing is written under `tmp_path`."""
    with pytest.raises(ValueError, match="not a valid artifact filename stem"):
        allocate_artifact_path(tmp_path, stem=bad_stem)

    assert list(tmp_path.iterdir()) == []


def test_a_multi_segment_traversal_stem_is_rejected_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    """Given a stem crafted to climb several directories above
    `artifacts_dir` via repeated `../` segments, when allocated, then
    `ValueError` is raised before any file is written -- a stem carrying
    a single `/` is already enough to reject, so a longer climb is
    refused the same way, not partially followed."""
    artifacts_dir = tmp_path / ".kestrel" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="not a valid artifact filename stem"):
        allocate_artifact_path(artifacts_dir, stem="../../../escaped")

    assert list(artifacts_dir.iterdir()) == []
