"""Tests for stripping terminal control sequences from untrusted model output."""

from __future__ import annotations

import pytest

from kestrel.repl import sanitize_terminal

pytestmark = [pytest.mark.p008, pytest.mark.unit, pytest.mark.sanity]


@pytest.mark.regression
def test_strips_csi_clear_screen_sequence() -> None:
    """Given text containing a CSI "clear screen" sequence, when
    sanitized, then the escape bytes are removed and surrounding text
    survives."""
    assert sanitize_terminal("before\x1b[2Jafter") == "beforeafter"


@pytest.mark.regression
def test_strips_osc_window_title_sequence() -> None:
    """Given text containing an OSC "set window title" sequence
    terminated by BEL, when sanitized, then the whole sequence is
    removed and surrounding text survives."""
    assert sanitize_terminal("before\x1b]0;evil title\x07after") == "beforeafter"


@pytest.mark.regression
def test_strips_eight_bit_csi_introducer() -> None:
    """Given text using the single-byte (8-bit) CSI introducer 0x9b
    instead of ESC-[, when sanitized, then it is removed exactly like
    the 7-bit form."""
    assert sanitize_terminal("before\x9b2Jafter") == "beforeafter"


@pytest.mark.regression
def test_strips_osc_terminated_by_eight_bit_string_terminator() -> None:
    """Given an OSC sequence terminated by the 8-bit string terminator
    (0x9c) rather than BEL, when sanitized, then it is removed in full."""
    assert sanitize_terminal("before\x9d0;title\x9cafter") == "beforeafter"


@pytest.mark.regression
def test_strips_bare_carriage_return() -> None:
    """Given a bare carriage return with no matching newline, when
    sanitized, then it is removed like any other stray C0 control byte."""
    assert sanitize_terminal("before\rafter") == "beforeafter"


def test_strips_lone_escape_byte_not_part_of_a_full_sequence() -> None:
    """Given a lone ESC byte that does not form a complete CSI/OSC
    sequence, when sanitized, then it is still removed as a stray C0
    control byte."""
    assert sanitize_terminal("before\x1bafter") == "beforeafter"


def test_preserves_newline_and_tab() -> None:
    """Given text containing newlines and tabs, when sanitized, then
    neither is treated as control-sequence noise."""
    assert sanitize_terminal("line one\n\tindented") == "line one\n\tindented"


def test_preserves_multibyte_utf8_text() -> None:
    """Given ordinary multibyte Unicode text, when sanitized, then it
    passes through completely unchanged."""
    text = "café 日本語 \U0001f985"
    assert sanitize_terminal(text) == text
