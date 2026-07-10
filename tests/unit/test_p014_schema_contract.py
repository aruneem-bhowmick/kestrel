"""Contract test: `READ_FILE_SCHEMA` is valid JSON Schema, stays in sync
with `ReadFileArgs`'s fields, and enforces the arguments shape
`parse_read_file_args` itself expects.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.read_file import READ_FILE_SCHEMA, ReadFileArgs

pytestmark = [pytest.mark.p014, pytest.mark.unit, pytest.mark.api]


def test_schema_parameters_are_valid_json_schema() -> None:
    """Given `READ_FILE_SCHEMA.parameters`, when checked against the
    Draft 7 meta-schema, then it validates without raising."""
    Draft7Validator.check_schema(READ_FILE_SCHEMA.parameters)


def test_every_read_file_args_field_has_a_schema_property() -> None:
    """Given every field `ReadFileArgs` can hold, when compared against
    `READ_FILE_SCHEMA.parameters`'s declared properties, then each has a
    corresponding entry -- the wire schema can never silently drift out
    of sync with the dataclass callers actually receive."""
    dataclass_fields = set(ReadFileArgs.__dataclass_fields__)

    assert dataclass_fields <= set(READ_FILE_SCHEMA.parameters["properties"])


def test_schema_declares_path_required_and_forbids_extra_fields() -> None:
    """Given `READ_FILE_SCHEMA.parameters`, when its `required` and
    `additionalProperties` keys are inspected, then `path` is the only
    required field and no field beyond the declared three is accepted --
    matching `parse_read_file_args`'s own unexpected-field rejection."""
    parameters = READ_FILE_SCHEMA.parameters

    assert parameters["required"] == ["path"]
    assert parameters["additionalProperties"] is False
