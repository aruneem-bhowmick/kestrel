"""Contract test: `VERIFY_SCHEMA` is valid JSON Schema and enforces the
arguments shape `parse_verify_args` itself expects, mirroring every
other tool's own contract test.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.verify import VERIFY_SCHEMA, VerifyArgs

pytestmark = [pytest.mark.p025, pytest.mark.unit, pytest.mark.api]


def test_schema_parameters_are_valid_json_schema() -> None:
    """Given `VERIFY_SCHEMA.parameters`, when checked against the
    Draft 7 meta-schema, then it validates without raising."""
    Draft7Validator.check_schema(VERIFY_SCHEMA.parameters)


def test_every_verify_args_field_has_a_schema_property() -> None:
    """Given every field `VerifyArgs` can hold, when compared against
    `VERIFY_SCHEMA.parameters`'s declared properties, then each has a
    corresponding entry -- the wire schema can never silently drift out
    of sync with the dataclass callers actually receive."""
    dataclass_fields = set(VerifyArgs.__dataclass_fields__)

    assert dataclass_fields <= set(VERIFY_SCHEMA.parameters["properties"])


def test_schema_declares_no_required_fields_and_forbids_extras() -> None:
    """Given `VERIFY_SCHEMA.parameters`, when its `required` and
    `additionalProperties` keys are inspected, then nothing is required
    (every configured command runs when `only` is omitted) and no field
    beyond the declared one is accepted -- matching `parse_verify_args`'s
    own unexpected-field rejection."""
    parameters = VERIFY_SCHEMA.parameters

    assert parameters["required"] == []
    assert parameters["additionalProperties"] is False


def test_schema_restricts_only_items_to_lint_build_test() -> None:
    """Given `VERIFY_SCHEMA.parameters`'s `only` property, when its item
    shape is inspected, then it declares a string enum of exactly
    lint/build/test -- matching `parse_verify_args`'s own unknown-name
    rejection."""
    only_property = VERIFY_SCHEMA.parameters["properties"]["only"]

    assert only_property["type"] == "array"
    assert only_property["items"]["type"] == "string"
    assert set(only_property["items"]["enum"]) == {"lint", "build", "test"}
