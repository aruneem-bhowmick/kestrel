"""Contract test: `EDIT_FILE_SCHEMA` is valid JSON Schema, stays in sync
with `EditFileArgs`'s fields, and enforces the arguments shape
`parse_edit_file_args` itself expects.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.edit_file import EDIT_FILE_SCHEMA, EditFileArgs

pytestmark = [pytest.mark.p018, pytest.mark.unit, pytest.mark.api]


def test_schema_parameters_are_valid_json_schema() -> None:
    """Given `EDIT_FILE_SCHEMA.parameters`, when checked against the
    Draft 7 meta-schema, then it validates without raising."""
    Draft7Validator.check_schema(EDIT_FILE_SCHEMA.parameters)


def test_every_edit_file_args_field_has_a_schema_property() -> None:
    """Given every field `EditFileArgs` can hold, when compared against
    `EDIT_FILE_SCHEMA.parameters`'s declared properties, then each has
    a corresponding entry -- the wire schema can never silently drift
    out of sync with the dataclass callers actually receive."""
    dataclass_fields = set(EditFileArgs.__dataclass_fields__)

    assert dataclass_fields <= set(EDIT_FILE_SCHEMA.parameters["properties"])


def test_schema_declares_path_old_new_required_and_forbids_extra_fields() -> None:
    """Given `EDIT_FILE_SCHEMA.parameters`, when its `required` and
    `additionalProperties` keys are inspected, then `path`, `old`, and
    `new` are exactly the required fields and no field beyond the
    declared four is accepted -- matching `parse_edit_file_args`'s own
    missing/unexpected-field rejection."""
    parameters = EDIT_FILE_SCHEMA.parameters

    assert set(parameters["required"]) == {"path", "old", "new"}
    assert parameters["additionalProperties"] is False


def test_schema_declares_dry_run_as_an_optional_boolean() -> None:
    """Given `EDIT_FILE_SCHEMA.parameters`'s `dry_run` property, when
    inspected, then it declares a `boolean` type and is not among the
    schema's required fields -- matching `parse_edit_file_args`'s own
    default of `False` when it is omitted."""
    parameters = EDIT_FILE_SCHEMA.parameters

    assert parameters["properties"]["dry_run"]["type"] == "boolean"
    assert "dry_run" not in parameters["required"]
