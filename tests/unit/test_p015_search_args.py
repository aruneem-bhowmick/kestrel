"""Tests for `search`'s argument parsing and validation: malformed JSON,
a missing `pattern`, out-of-range `max_results`, and the well-formed
case -- mirroring `read_file`'s `parse_*_args` shape.
"""

from __future__ import annotations

import pytest

from kestrel.tools.search import SearchArgs, SearchError, parse_search_args

pytestmark = [pytest.mark.p015, pytest.mark.unit]


@pytest.mark.sanity
def test_malformed_json_raises_via_parse_search_args() -> None:
    """Given a syntactically invalid JSON string, when parsed, then
    `SearchError` is raised instead of a raw `json.JSONDecodeError`
    escaping to the caller."""
    with pytest.raises(SearchError, match="invalid JSON"):
        parse_search_args("{not json")


def test_arguments_json_not_an_object_raises() -> None:
    """Given valid JSON that is not an object, when parsed, then
    `SearchError` names the shape mismatch."""
    with pytest.raises(SearchError, match="expected a JSON object"):
        parse_search_args("[1, 2, 3]")


@pytest.mark.sanity
def test_missing_pattern_field_raises_via_parse_search_args() -> None:
    """Given arguments with no `pattern` field, when parsed, then
    `SearchError` names the missing field."""
    with pytest.raises(SearchError, match="missing required field 'pattern'"):
        parse_search_args("{}")


def test_unexpected_extra_field_raises_via_parse_search_args() -> None:
    """Given arguments carrying a field the schema does not declare, when
    parsed, then `SearchError` names the offending field."""
    with pytest.raises(SearchError, match="unexpected field"):
        parse_search_args('{"pattern": "foo", "recursive": true}')


def test_pattern_field_wrong_type_raises_via_parse_search_args() -> None:
    """Given a `pattern` field that is not a string, when parsed, then
    `SearchError` is raised naming the expected type."""
    with pytest.raises(SearchError, match="'pattern' must be a string"):
        parse_search_args('{"pattern": 123}')


def test_scope_field_wrong_type_raises_via_parse_search_args() -> None:
    """Given a `scope` field that is not a string, when parsed, then
    `SearchError` is raised naming the expected type."""
    with pytest.raises(SearchError, match="'scope' must be a string"):
        parse_search_args('{"pattern": "foo", "scope": 123}')


@pytest.mark.parametrize("bad_value", ['"two"', "0", "-1", "201", "true"])
def test_max_results_out_of_range_or_wrong_type_raises(bad_value: str) -> None:
    """Given a `max_results` that is neither an integer nor within
    `[1, 200]` -- including a JSON boolean, which is an `int` subclass in
    Python but never a valid result cap -- when parsed, then `SearchError`
    names the offending field."""
    with pytest.raises(SearchError, match="'max_results' must be"):
        parse_search_args(f'{{"pattern": "foo", "max_results": {bad_value}}}')


def test_max_results_absent_defaults_to_fifty() -> None:
    """Given arguments with no `max_results` field, when parsed, then
    the resulting `SearchArgs` defaults `max_results` to 50."""
    args = parse_search_args('{"pattern": "foo"}')

    assert args.max_results == 50


@pytest.mark.parametrize("boundary_value", [1, 200])
def test_max_results_boundary_values_are_accepted(boundary_value: int) -> None:
    """Given `max_results` set to exactly 1 or exactly 200, when parsed,
    then no error is raised and the value is carried through unchanged."""
    args = parse_search_args(f'{{"pattern": "foo", "max_results": {boundary_value}}}')

    assert args.max_results == boundary_value


@pytest.mark.sanity
def test_parse_search_args_builds_the_expected_dataclass() -> None:
    """Given well-formed arguments carrying every field, when parsed,
    then the resulting `SearchArgs` carries each value exactly."""
    args = parse_search_args('{"pattern": "TODO", "scope": "src", "max_results": 10}')

    assert args == SearchArgs(pattern="TODO", scope="src", max_results=10)


def test_parse_search_args_defaults_scope_to_none() -> None:
    """Given arguments with no `scope` field, when parsed, then the
    resulting `SearchArgs` leaves `scope` as `None`."""
    args = parse_search_args('{"pattern": "TODO"}')

    assert args.scope is None
