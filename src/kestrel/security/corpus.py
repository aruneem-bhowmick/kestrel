"""Loader for the checked-in adversarial prompt-injection corpus.

The corpus itself lives as data, not code: one JSON file per case under
``tests/fixtures/injection_corpus/``, so a new adversarial shape can be
added by dropping in a file rather than editing a source module. This
module only knows how to find and parse that directory; it carries no
opinion about what makes a payload hostile, and nothing here asserts
that :func:`~kestrel.security.framing.frame_untrusted` actually defeats
a given case -- that proof belongs to the tests that consume
:func:`load_corpus`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kestrel.security.framing import SourceKind

# tests/fixtures/injection_corpus, relative to this file's location in a
# repo checkout (src/kestrel/security/corpus.py -> repo root -> tests/...).
# The corpus is a checked-in fixture tree, not packaged data, so this
# resolves relative to the source tree rather than importlib.resources.
_CORPUS_DIR: Path = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "injection_corpus"
)


@dataclass(frozen=True, slots=True)
class InjectionCase:
    """One adversarial fixture from the injection corpus.

    Attributes:
        id: Stable slug identifying this case; matches its JSON filename
            (minus the ``.json`` extension).
        source: The :class:`~kestrel.security.framing.SourceKind` this
            case pretends to originate from.
        payload: The hostile text itself, verbatim -- the exact string a
            test passes to :func:`~kestrel.security.framing.frame_untrusted`.
        forbidden_markers: Strings that must never appear unescaped in a
            rendered frame built from ``payload``.
    """

    id: str
    source: SourceKind
    payload: str
    forbidden_markers: tuple[str, ...]


def _case_from_json(data: dict[str, Any]) -> InjectionCase:
    """Build one :class:`InjectionCase` from a parsed JSON object."""
    return InjectionCase(
        id=data["id"],
        source=data["source"],
        payload=data["payload"],
        forbidden_markers=tuple(data["forbidden_markers"]),
    )


def load_corpus() -> tuple[InjectionCase, ...]:
    """Load every case from tests/fixtures/injection_corpus/*.json,
    sorted by id for deterministic iteration order.
    """
    cases = [
        _case_from_json(json.loads(path.read_text(encoding="utf-8")))
        for path in _CORPUS_DIR.glob("*.json")
    ]
    return tuple(sorted(cases, key=lambda case: case.id))
