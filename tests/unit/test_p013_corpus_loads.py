"""Structural proof that the checked-in injection corpus and
`frame_untrusted` together neutralize every case's delimiter-forgery
attempt.

This does not yet prove a live model refuses any of these payloads --
only that framing leaves no unescaped, forged occurrence of either
marker in the rendered output. The end-to-end, live-model proof belongs
to later suites that drive real tool output through this same corpus.
"""

from __future__ import annotations

import pytest

from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.security.framing import frame_untrusted

pytestmark = [pytest.mark.p013, pytest.mark.unit, pytest.mark.redteam]

_EXPECTED_CASE_COUNT = 6
_ORIGIN = "corpus-fixture"


def test_load_corpus_returns_exactly_six_cases() -> None:
    """Given the checked-in corpus directory, when loaded, then it
    yields exactly six cases."""
    assert len(load_corpus()) == _EXPECTED_CASE_COUNT


def test_every_case_id_is_unique() -> None:
    """Given the loaded corpus, when every case's `id` is collected,
    then no two cases share an id."""
    ids = [case.id for case in load_corpus()]

    assert len(ids) == len(set(ids))


def test_load_corpus_is_sorted_by_id() -> None:
    """Given the loaded corpus, when its order is inspected, then cases
    appear in ascending `id` order, making iteration deterministic
    regardless of filesystem directory order."""
    cases = load_corpus()

    assert [case.id for case in cases] == sorted(case.id for case in cases)


@pytest.mark.parametrize("case", load_corpus(), ids=[case.id for case in load_corpus()])
def test_case_payload_forbidden_markers_escape_to_one_occurrence(
    case: InjectionCase,
) -> None:
    """Given one corpus case's payload, when framed, then every one of
    its `forbidden_markers` appears exactly once in the rendered
    output -- the real marker `frame_untrusted` itself emits, and never
    a second, forged occurrence smuggled in through the payload."""
    framed = frame_untrusted(case.payload, source=case.source, origin=_ORIGIN)

    for marker in case.forbidden_markers:
        assert framed.count(marker) == 1
