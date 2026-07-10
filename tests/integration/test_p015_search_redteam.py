"""Red-team proof that searching a hostile file through `search` still
leaves the returned content wrapped by the real frame markers -- a
matched line's hostile content cannot smuggle itself out of its own
frame.

Skipped locally when `rg` is not on `PATH`, for the same reason as
`test_p015_search_rg.py`: `search` always shells out to a real `rg`
process, so this proof needs the real binary, not a mock.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.search import SearchArgs, search

pytestmark = [
    pytest.mark.p015,
    pytest.mark.integration,
    pytest.mark.redteam,
    pytest.mark.skipif(shutil.which("rg") is None, reason="rg not found on PATH"),
]

_HOSTILE_CASE_ID = "readme_ignore_previous_instructions"
_MATCH_PATTERN = "ignore previous instructions"


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
def hostile_search_case() -> InjectionCase:
    """The corpus case used to prove `search`'s framing survives a real
    prompt-injection payload matched by a search pattern. Looked up
    lazily (rather than at module import time) so a lookup failure
    surfaces as a normal test error tied to whichever test requested it.
    """
    return _find_case(_HOSTILE_CASE_ID)


def test_hostile_matched_line_still_carries_the_real_frame_markers(
    tmp_path: Path, hostile_search_case: InjectionCase
) -> None:
    """Given a fixture file whose content is one of the injection
    corpus's hostile payloads, when searched for a pattern that matches
    a line inside that payload, then the result still starts with the
    real opening header, still ends with the real closing delimiter, and
    none of the case's forbidden markers appear more than the one time
    `frame_untrusted` itself emits them."""
    (tmp_path / "README.md").write_bytes(hostile_search_case.payload.encode("utf-8"))

    framed = search(SearchArgs(pattern=_MATCH_PATTERN), repo_root=tmp_path)

    assert framed.startswith("<<<UNTRUSTED:search_result:")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert _MATCH_PATTERN in framed
    for marker in hostile_search_case.forbidden_markers:
        assert framed.count(marker) == 1
