"""Contract test: `EXECUTE_SCHEMA` is valid JSON Schema, stays in sync
with `ExecuteArgs`'s fields, and enforces the arguments shape
`parse_execute_args` itself expects.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.execute import EXECUTE_SCHEMA, ExecuteArgs

pytestmark = [pytest.mark.p016, pytest.mark.unit, pytest.mark.api]


def test_schema_parameters_are_valid_json_schema() -> None:
    """Given `EXECUTE_SCHEMA.parameters`, when checked against the
    Draft 7 meta-schema, then it validates without raising."""
    Draft7Validator.check_schema(EXECUTE_SCHEMA.parameters)


def test_every_execute_args_field_has_a_schema_property() -> None:
    """Given every field `ExecuteArgs` can hold, when compared against
    `EXECUTE_SCHEMA.parameters`'s declared properties, then each has a
    corresponding entry -- the wire schema can never silently drift out
    of sync with the dataclass callers actually receive."""
    dataclass_fields = set(ExecuteArgs.__dataclass_fields__)

    assert dataclass_fields <= set(EXECUTE_SCHEMA.parameters["properties"])


def test_schema_declares_cmd_required_and_forbids_extra_fields() -> None:
    """Given `EXECUTE_SCHEMA.parameters`, when its `required` and
    `additionalProperties` keys are inspected, then `cmd` is the only
    required field and no field beyond the declared two is accepted --
    matching `parse_execute_args`'s own unexpected-field rejection."""
    parameters = EXECUTE_SCHEMA.parameters

    assert parameters["required"] == ["cmd"]
    assert parameters["additionalProperties"] is False


def test_schema_requires_cmd_to_be_a_non_empty_array_of_strings() -> None:
    """Given `EXECUTE_SCHEMA.parameters`'s `cmd` property, when its
    shape is inspected, then it declares an array of `string` items
    with a minimum of one -- matching `parse_execute_args`'s own
    empty/non-string rejection."""
    cmd_property = EXECUTE_SCHEMA.parameters["properties"]["cmd"]

    assert cmd_property["type"] == "array"
    assert cmd_property["items"] == {"type": "string"}
    assert cmd_property["minItems"] == 1


def test_schema_bounds_timeout_s_between_one_and_three_hundred() -> None:
    """Given `EXECUTE_SCHEMA.parameters`'s `timeout_s` property, when
    its bounds are inspected, then they match `parse_execute_args`'s
    own `[1, 300]` validation range."""
    timeout_property = EXECUTE_SCHEMA.parameters["properties"]["timeout_s"]

    assert timeout_property["minimum"] == 1
    assert timeout_property["maximum"] == 300
