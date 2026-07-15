"""Guards `tests/golden/`'s byte-exact fixtures against line-ending
drift.

`.gitattributes` already stops git itself from rewriting these files'
line endings on checkout, but that only protects against one path to
corruption. This test is a second, independent check: it inspects the
fixtures' actual on-disk bytes directly, so a fixture that gets
hand-edited or re-added with the wrong line ending later -- rather than
mangled by a checkout -- is caught here too, instead of surfacing as a
confusing failure in whichever test happens to compare against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def test_no_golden_fixture_contains_a_carriage_return() -> None:
    """Given every `*.golden` fixture on disk, when its raw bytes are
    inspected, then none contains a `\\r` -- the byte-exact snapshots
    several tests compare against with `read_bytes()` are meaningless if
    a platform-specific line-ending conversion has silently changed them
    from what was actually committed."""
    golden_files = sorted(_GOLDEN_DIR.glob("*.golden"))
    assert _GOLDEN_DIR.is_dir(), f"golden fixture directory missing: {_GOLDEN_DIR}"
    assert golden_files, f"no *.golden fixtures found under {_GOLDEN_DIR}"

    offenders = [path.name for path in golden_files if b"\r" in path.read_bytes()]

    assert offenders == []
