"""Red-team proof that reading a hostile file through `read_file` still
leaves the returned content wrapped by the real frame markers -- a
hostile file's content cannot smuggle itself out of its own frame.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.read_file import ReadFileArgs, read_file

pytestmark = [pytest.mark.p014, pytest.mark.unit, pytest.mark.redteam]

_HOSTILE_CASE_ID = "readme_ignore_previous_instructions"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


@pytest.fixture(scope="session")
def hostile_file_case() -> InjectionCase:
    """The corpus case used to prove `read_file`'s framing survives a
    real prompt-injection payload. Looked up lazily (rather than at
    module import time) so a lookup failure surfaces as a normal test
    error tied to whichever test requested it."""
    return _find_case(_HOSTILE_CASE_ID)


def test_find_case_raises_a_clear_error_for_an_unknown_case_id() -> None:
    """Given a case id absent from the corpus, when looked up, then a
    clear `AssertionError` names it instead of an opaque
    `StopIteration`."""
    with pytest.raises(AssertionError, match="not-a-real-case-id"):
        _find_case("not-a-real-case-id")


def test_hostile_file_content_still_carries_the_real_frame_markers(
    tmp_path: Path, hostile_file_case: InjectionCase
) -> None:
    """Given a file whose content is one of the injection corpus's
    hostile payloads, when read through `read_file`, then the result
    still starts with the real opening header naming this file, still
    ends with the real closing delimiter, and the hostile payload itself
    survives unchanged between them."""
    (tmp_path / "README.md").write_bytes(hostile_file_case.payload.encode("utf-8"))

    framed = read_file(ReadFileArgs(path="README.md"), repo_root=tmp_path)

    assert framed.startswith("<<<UNTRUSTED:file:README.md>>>\n")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert hostile_file_case.payload in framed
    for marker in hostile_file_case.forbidden_markers:
        assert framed.count(marker) == 1
