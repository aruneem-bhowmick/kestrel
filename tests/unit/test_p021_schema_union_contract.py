"""Contract test: every schema `all_schemas()` offers a provider call
is itself valid JSON Schema, and no two of them collide on `name`.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from kestrel.tools.registry import all_schemas

pytestmark = [pytest.mark.p021, pytest.mark.unit, pytest.mark.api]


def test_every_schemas_parameters_are_valid_json_schema() -> None:
    """Given every schema `all_schemas()` returns, when each one's
    `parameters` is checked against the Draft 7 meta-schema, then it
    validates without raising."""
    for schema in all_schemas():
        Draft7Validator.check_schema(schema.parameters)


def test_no_two_schemas_share_a_name() -> None:
    """Given `all_schemas()`, when every schema's `name` is collected,
    then no two of them collide -- a model is never offered two tools
    it cannot tell apart by name."""
    names = [schema.name for schema in all_schemas()]

    assert len(names) == len(set(names))
