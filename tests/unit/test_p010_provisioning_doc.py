"""Structural and hygiene tests for the Jetson provisioning guide.

These tests never touch the network or a real Jetson board -- they only
read the guide as committed text, so its contract cannot silently drift
out from under a future edit: which sections exist and in what order,
that every hardware-only caveat stays attached to the section it
caveats, and that no example ever carries something that looks like a
real credential.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.p010,
    pytest.mark.unit,
    pytest.mark.sanity,
    pytest.mark.regression,
]

_DOC_PATH = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "provisioning-jetson.md"
)

_EXPECTED_SECTIONS = [
    "Prerequisites",
    "Flash JetPack 6.2",
    "NVMe setup",
    "Power mode",
    "Python & uv",
    "Kestrel install",
    "Flight check",
    "Ollama (deferred)",
]

_SECTION_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)

# Mirrors the credential-shaped key pattern kestrel.config rejects in
# kestrel.toml, applied here to assignment-style text in the guide itself
# rather than a parsed TOML tree.
_SECRET_ASSIGNMENT = re.compile(
    r"[\w.-]*(?:api[_-]?key|token|secret|password)[\w.-]*\s*[:=]\s*(\S+)",
    re.IGNORECASE,
)
_ALLOWED_SECRET_VALUE = "sk-...redacted"
_UNVERIFIED_TAG = "[UNVERIFIED]"


def _sections(text: str) -> dict[str, str]:
    """Split the guide into a mapping of H2 heading text to its body."""
    headings = list(_SECTION_HEADING.finditer(text))
    body_by_heading: dict[str, str] = {}
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body_by_heading[match.group(1)] = text[start:end]
    return body_by_heading


def test_provisioning_guide_exists() -> None:
    """The guide is committed at the fixed path the README links to."""
    assert _DOC_PATH.is_file()


@pytest.mark.acceptance
def test_provisioning_guide_has_the_expected_sections_in_order() -> None:
    """Given the guide's committed text, when its top-level headings are
    extracted, then they match the required section list exactly and in
    order -- a future edit cannot silently drop or reorder one."""
    text = _DOC_PATH.read_text(encoding="utf-8")
    assert _SECTION_HEADING.findall(text) == _EXPECTED_SECTIONS


def test_provisioning_guide_has_no_secret_shaped_assignments() -> None:
    """Given the guide's committed text, when scanned for credential-shaped
    assignments, then every one of them uses the documented placeholder
    value rather than something that could be mistaken for a real key."""
    text = _DOC_PATH.read_text(encoding="utf-8")
    offending = [
        match.group(0)
        for match in _SECRET_ASSIGNMENT.finditer(text)
        if match.group(1) != _ALLOWED_SECRET_VALUE
    ]
    assert offending == []


def test_unverified_tags_are_confined_to_hardware_dependent_sections() -> None:
    """Given the guide's committed text, when split into sections, then
    the hardware-validation tag appears only in the two sections whose
    steps cannot be confirmed without a physical board, and never in the
    guide's introduction before its first section."""
    text = _DOC_PATH.read_text(encoding="utf-8")
    sections = _sections(text)

    tagged = {name for name, body in sections.items() if _UNVERIFIED_TAG in body}
    assert tagged == {"Flash JetPack 6.2", "NVMe setup"}

    first_heading = _SECTION_HEADING.search(text)
    assert first_heading is not None
    preamble = text[: first_heading.start()]
    assert _UNVERIFIED_TAG not in preamble
