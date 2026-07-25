"""Unit tests for `search`'s new `semantic` argument: `parse_search_args`
accepts a real boolean or its absence (defaulting to `False`), and
rejects any other JSON type naming the field -- mirroring
`max_results`'s own validation shape.
"""

from __future__ import annotations

import pytest

from kestrel.tools.search import SearchArgs, SearchError, parse_search_args

pytestmark = [pytest.mark.p060, pytest.mark.unit]


@pytest.mark.sanity
def test_semantic_true_is_accepted() -> None:
    """Given `"semantic": true`, when parsed, then the resulting
    `SearchArgs` carries `semantic=True`."""
    args = parse_search_args('{"pattern": "foo", "semantic": true}')

    assert args == SearchArgs(pattern="foo", semantic=True)


def test_semantic_false_is_accepted() -> None:
    """Given `"semantic": false`, when parsed, then the resulting
    `SearchArgs` carries `semantic=False`."""
    args = parse_search_args('{"pattern": "foo", "semantic": false}')

    assert args.semantic is False


@pytest.mark.sanity
def test_semantic_absent_defaults_to_false() -> None:
    """Given arguments with no `semantic` field, when parsed, then the
    resulting `SearchArgs` defaults `semantic` to `False`."""
    args = parse_search_args('{"pattern": "foo"}')

    assert args.semantic is False


@pytest.mark.parametrize("bad_value", ['"yes"', "1", "0", "1.5", "[]"])
def test_semantic_non_boolean_raises(bad_value: str) -> None:
    """Given a `semantic` field that is not a JSON boolean -- including a
    JSON `1`/`0`, which are `int`s, never `bool`s, in Python -- when
    parsed, then `SearchError` names the offending field."""
    with pytest.raises(SearchError, match="'semantic' must be a boolean"):
        parse_search_args(f'{{"pattern": "foo", "semantic": {bad_value}}}')
