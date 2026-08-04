"""Red-team unit tests for `kestrel.tui.writeback_modal.WritebackModal`:
a proposed learning's own text is ultimately model-authored, so every
payload in the checked-in injection corpus, used as a `ProposedLearning
.text`, must render inside the modal without raising and without the
checkbox's own label ever being interpreted as Textual markup that
could corrupt the modal's own layout -- mirrors `sanitize_terminal`'s
own purpose (see `test_p042_approval_modal.py`'s identical concern for
`ApprovalModal`), applied here to a widget label rather than a
`RichLog`/`Static` line.
"""

from __future__ import annotations

import pytest
from textual.widgets import Checkbox

from kestrel.kb.writeback import ProposedLearning
from kestrel.security.corpus import load_corpus
from kestrel.tui.writeback_modal import WritebackModal

pytestmark = [pytest.mark.p066, pytest.mark.unit, pytest.mark.redteam, pytest.mark.ui]


def _checkbox_for(payload: str) -> Checkbox:
    """Compose a `WritebackModal` carrying one `ProposedLearning` whose
    text is `payload`, and return its one checkbox."""
    modal = WritebackModal([ProposedLearning(text=payload, tags=())])
    list(modal.compose())
    (checkbox,) = modal._checkboxes
    return checkbox


def test_every_corpus_payload_renders_inside_the_modal_without_raising() -> None:
    """Given every payload in the injection corpus, each used as a
    `ProposedLearning.text`, when the modal composes, then no case
    raises."""
    for case in load_corpus():
        _checkbox_for(case.payload)


def test_every_corpus_payload_strips_raw_terminal_control_bytes() -> None:
    """Given every payload in the injection corpus, when rendered as a
    checkbox label, then none of the label's own plain text carries a
    raw ANSI/CSI/OSC escape byte."""
    for case in load_corpus():
        checkbox = _checkbox_for(case.payload)
        plain = checkbox.label.plain
        assert "\x1b" not in plain
        assert "\x9b" not in plain
        assert "\x07" not in plain


def test_every_corpus_payload_is_never_interpreted_as_markup() -> None:
    """Given every payload in the injection corpus, when rendered as a
    checkbox label, then the label carries no markup style spans --
    proof that a payload shaped like Rich console markup (e.g.
    `[conceal]...[/conceal]`) renders as inert plain text rather than
    being interpreted and hiding part of the label from the approver."""
    for case in load_corpus():
        checkbox = _checkbox_for(case.payload)
        assert checkbox.label.spans == []


@pytest.mark.parametrize(
    "hostile",
    [
        "rm [conceal]--no-preserve-root[/conceal] -rf /",
        "[bold red]APPROVE ME[/bold red]",
        "safe-looking text [link=https://evil.example]click[/link]",
    ],
)
def test_rich_markup_shaped_text_renders_as_literal_characters(hostile: str) -> None:
    """Given a learning whose text is itself shaped like Rich console
    markup, when rendered as a checkbox label, then the label's own
    plain text includes every bracketed tag verbatim, unconsumed by
    markup parsing."""
    checkbox = _checkbox_for(hostile)

    assert checkbox.label.plain == hostile
    assert checkbox.label.spans == []
