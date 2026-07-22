"""Tests for the L2-distance-to-cosine-similarity identity `KnowledgeStore.
search` relies on: `1 - d**2 / 2`, exact for unit-normalized vectors.

Each case below is worked by hand against a pair of already-unit-length
vectors at a known angle, independent of `KnowledgeStore` itself, so a
failure here always points at the identity (or its arithmetic), never at
the store's own SQL or schema.
"""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.p057, pytest.mark.unit, pytest.mark.sanity]


def _l2_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """The Euclidean distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _cosine_from_l2_distance(distance: float) -> float:
    """The identity `KnowledgeStore.search` itself applies to every
    `vec0` KNN result's own distance column."""
    return 1 - distance**2 / 2


def test_parallel_unit_vectors_score_approximately_one() -> None:
    """Given two identical unit vectors (angle 0), when the identity is
    applied to their L2 distance, then the resulting score is
    approximately 1.0 -- maximally similar."""
    a = (1.0, 0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0, 0.0)

    score = _cosine_from_l2_distance(_l2_distance(a, b))

    assert score == pytest.approx(1.0)


def test_orthogonal_unit_vectors_score_approximately_zero() -> None:
    """Given two orthogonal unit vectors (angle 90 degrees), when the
    identity is applied to their L2 distance, then the resulting score
    is approximately 0.0."""
    a = (1.0, 0.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0, 0.0)

    score = _cosine_from_l2_distance(_l2_distance(a, b))

    assert score == pytest.approx(0.0)


def test_opposite_unit_vectors_score_approximately_negative_one() -> None:
    """Given two diametrically opposed unit vectors (angle 180 degrees),
    when the identity is applied to their L2 distance, then the
    resulting score is approximately -1.0 -- maximally dissimilar."""
    a = (1.0, 0.0, 0.0, 0.0)
    b = (-1.0, 0.0, 0.0, 0.0)

    score = _cosine_from_l2_distance(_l2_distance(a, b))

    assert score == pytest.approx(-1.0)


def test_45_degree_unit_vectors_score_matches_cosine_of_the_angle() -> None:
    """Given two unit vectors 45 degrees apart, when the identity is
    applied to their L2 distance, then the resulting score matches
    `cos(45deg)` exactly -- a non-axis-aligned angle, not just the three
    canonical cases above."""
    angle = math.pi / 4
    a = (1.0, 0.0)
    b = (math.cos(angle), math.sin(angle))

    score = _cosine_from_l2_distance(_l2_distance(a, b))

    assert score == pytest.approx(math.cos(angle))
