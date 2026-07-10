"""Contract test: `SEARCH_SCHEMA` is valid JSON Schema, stays in sync
with `SearchArgs`'s fields, and enforces the arguments shape
`parse_search_args` itself expects.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.search import SEARCH_SCHEMA, SearchArgs

pytestmark = [pytest.mark.p015, pytest.mark.unit, pytest.mark.api]


def test_schema_parameters_are_valid_json_schema() -> None:
    """Given `SEARCH_SCHEMA.parameters`, when checked against the
    Draft 7 meta-schema, then it validates without raising."""
    Draft7Validator.check_schema(SEARCH_SCHEMA.parameters)


def test_every_search_args_field_has_a_schema_property() -> None:
    """Given every field `SearchArgs` can hold, when compared against
    `SEARCH_SCHEMA.parameters`'s declared properties, then each has a
    corresponding entry -- the wire schema can never silently drift out
    of sync with the dataclass callers actually receive."""
    dataclass_fields = set(SearchArgs.__dataclass_fields__)

    assert dataclass_fields <= set(SEARCH_SCHEMA.parameters["properties"])


def test_schema_declares_pattern_required_and_forbids_extra_fields() -> None:
    """Given `SEARCH_SCHEMA.parameters`, when its `required` and
    `additionalProperties` keys are inspected, then `pattern` is the only
    required field and no field beyond the declared three is accepted --
    matching `parse_search_args`'s own unexpected-field rejection."""
    parameters = SEARCH_SCHEMA.parameters

    assert parameters["required"] == ["pattern"]
    assert parameters["additionalProperties"] is False


def test_schema_bounds_max_results_between_one_and_two_hundred() -> None:
    """Given `SEARCH_SCHEMA.parameters`'s `max_results` property, when
    its bounds are inspected, then they match `parse_search_args`'s own
    `[1, 200]` validation range."""
    max_results_property = SEARCH_SCHEMA.parameters["properties"]["max_results"]

    assert max_results_property["minimum"] == 1
    assert max_results_property["maximum"] == 200
