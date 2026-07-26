"""Unit tests for `kestrel.kb.writeback._parse_proposal_line`: turning
one raw reply line into a `ProposedLearning`, or `None` for anything
that does not conform to the documented `LEARNING: ... | TAGS: ...`
shape.

Every case here is a pure, synchronous parse -- no model call, no
`KbService`, nothing async.
"""

from __future__ import annotations

import pytest

from kestrel.kb.writeback import ProposedLearning, _parse_proposal_line

pytestmark = [pytest.mark.p062, pytest.mark.unit, pytest.mark.sanity]


def test_a_well_formed_line_parses_text_and_tags() -> None:
    """Given a line naming two comma-separated tags, when parsed, then
    the text and both tags come back exactly."""
    result = _parse_proposal_line(
        "LEARNING: prefer tabs in this repo | TAGS: style, formatting"
    )

    assert result == ProposedLearning(
        text="prefer tabs in this repo", tags=("style", "formatting")
    )


def test_a_single_tag_line_parses_one_tag() -> None:
    """Given a line naming exactly one tag, when parsed, then `tags` is
    a one-element tuple."""
    result = _parse_proposal_line("LEARNING: run tests via uv | TAGS: testing")

    assert result == ProposedLearning(text="run tests via uv", tags=("testing",))


def test_a_trailing_empty_tags_field_parses_to_zero_tags() -> None:
    """Given a line whose `TAGS:` field is present but trails off empty,
    when parsed, then `tags` is an empty tuple rather than a tuple
    holding one blank string."""
    result = _parse_proposal_line("LEARNING: no tags here | TAGS: ")

    assert result == ProposedLearning(text="no tags here", tags=())


def test_the_literal_none_reply_parses_to_none() -> None:
    """Given the model's own documented "nothing durable happened" reply,
    when parsed, then it comes back as `None`, not a learning naming
    "NONE" as its text."""
    assert _parse_proposal_line("NONE") is None


def test_a_blank_line_parses_to_none() -> None:
    """Given an empty line, when parsed, then it comes back as `None`."""
    assert _parse_proposal_line("") is None
    assert _parse_proposal_line("   ") is None


def test_a_line_missing_the_tags_separator_parses_to_none_not_an_exception() -> None:
    """Given a line that starts with the `LEARNING:` prefix but never
    supplies the `| TAGS:` separator, when parsed, then it degrades to
    `None` -- one malformed line never raises out of this function."""
    assert _parse_proposal_line("LEARNING: something happened, no tags field") is None


def test_a_line_not_starting_with_the_learning_prefix_parses_to_none() -> None:
    """Given an ordinary sentence that happens to mention "LEARNING"
    without leading with it, when parsed, then it comes back as `None`."""
    assert _parse_proposal_line("we learned something | TAGS: x") is None
