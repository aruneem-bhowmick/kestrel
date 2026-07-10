"""Tests for `execute`'s argument parsing and validation: malformed
JSON, an empty or non-list `cmd`, out-of-range `timeout_s`, and the
well-formed case -- mirroring `read_file`/`search`'s `parse_*_args`
shape.
"""

from __future__ import annotations

import pytest

from kestrel.tools.execute import ExecuteArgs, ExecuteError, parse_execute_args

pytestmark = [pytest.mark.p016, pytest.mark.unit]


@pytest.mark.sanity
def test_malformed_json_raises_via_parse_execute_args() -> None:
    """Given a syntactically invalid JSON string, when parsed, then
    `ExecuteError` is raised instead of a raw `json.JSONDecodeError`
    escaping to the caller."""
    with pytest.raises(ExecuteError, match="invalid JSON"):
        parse_execute_args("{not json")


def test_arguments_json_not_an_object_raises() -> None:
    """Given valid JSON that is not an object, when parsed, then
    `ExecuteError` names the shape mismatch."""
    with pytest.raises(ExecuteError, match="expected a JSON object"):
        parse_execute_args("[1, 2, 3]")


@pytest.mark.sanity
def test_missing_cmd_field_raises_via_parse_execute_args() -> None:
    """Given arguments with no `cmd` field, when parsed, then
    `ExecuteError` names the missing field."""
    with pytest.raises(ExecuteError, match="missing required field 'cmd'"):
        parse_execute_args("{}")


def test_unexpected_extra_field_raises_via_parse_execute_args() -> None:
    """Given arguments carrying a field the schema does not declare, when
    parsed, then `ExecuteError` names the offending field."""
    with pytest.raises(ExecuteError, match="unexpected field"):
        parse_execute_args('{"cmd": ["echo"], "shell": true}')


@pytest.mark.sanity
@pytest.mark.parametrize(
    "bad_cmd_json", ["[]", '"echo hello"', "123", "null", "[1, 2]", '["ok", 3]']
)
def test_cmd_empty_or_non_list_of_strings_raises(bad_cmd_json: str) -> None:
    """Given a `cmd` that is empty, not a list, or a list containing a
    non-string element, when parsed, then `ExecuteError` names the
    expected shape -- `cmd` must stay an argv list of strings all the
    way down to the sandboxed subprocess, never a shell string."""
    with pytest.raises(
        ExecuteError, match="'cmd' must be a non-empty array of strings"
    ):
        parse_execute_args(f'{{"cmd": {bad_cmd_json}}}')


@pytest.mark.parametrize("bad_value", ['"60"', "0", "0.5", "301", "true"])
def test_timeout_s_out_of_range_or_wrong_type_raises(bad_value: str) -> None:
    """Given a `timeout_s` that is neither a number nor within
    `[1, 300]` -- including a JSON boolean, which is an `int` subclass
    in Python but never a valid timeout -- when parsed, then
    `ExecuteError` names the offending field."""
    with pytest.raises(ExecuteError, match="'timeout_s' must be"):
        parse_execute_args(f'{{"cmd": ["true"], "timeout_s": {bad_value}}}')


def test_timeout_s_absent_defaults_to_sixty() -> None:
    """Given arguments with no `timeout_s` field, when parsed, then the
    resulting `ExecuteArgs` defaults `timeout_s` to 60.0."""
    args = parse_execute_args('{"cmd": ["true"]}')

    assert args.timeout_s == 60.0


@pytest.mark.parametrize("boundary_value", [1, 300])
def test_timeout_s_boundary_values_are_accepted(boundary_value: int) -> None:
    """Given `timeout_s` set to exactly 1 or exactly 300, when parsed,
    then no error is raised and the value is carried through as a
    `float`."""
    args = parse_execute_args(f'{{"cmd": ["true"], "timeout_s": {boundary_value}}}')

    assert args.timeout_s == float(boundary_value)


@pytest.mark.sanity
def test_parse_execute_args_builds_the_expected_dataclass() -> None:
    """Given well-formed arguments carrying every field, when parsed,
    then the resulting `ExecuteArgs` carries each value exactly, with
    `cmd` as a tuple in its original order."""
    args = parse_execute_args('{"cmd": ["pytest", "-q"], "timeout_s": 30}')

    assert args == ExecuteArgs(cmd=("pytest", "-q"), timeout_s=30.0)


def test_parse_execute_args_preserves_cmd_element_order() -> None:
    """Given a multi-element `cmd`, when parsed, then the resulting
    tuple preserves the exact original order -- `execute` hands this
    straight to the sandboxed subprocess as its argv."""
    args = parse_execute_args('{"cmd": ["git", "status", "--short"]}')

    assert args.cmd == ("git", "status", "--short")
