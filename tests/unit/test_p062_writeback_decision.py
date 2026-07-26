"""Unit tests for `kestrel.kb.writeback._prompt_stdin_writeback`: the
default, real-stdin decision function `commit_learnings`'s caller wires
up, exercised against a scripted `input_fn` rather than the real
terminal -- mirroring `test_p019_execute_classification.py`'s own
coverage of `kestrel.managers.approval._prompt_stdin`.
"""

from __future__ import annotations

import pytest

from kestrel.kb.writeback import ProposedLearning, _prompt_stdin_writeback

pytestmark = [pytest.mark.p062, pytest.mark.unit, pytest.mark.sanity]

_LEARNING = ProposedLearning(text="prefer tabs in this repo", tags=("style",))


@pytest.mark.parametrize("reply", ["y", "yes", "Y", "YES", "Yes"])
def test_an_affirmative_reply_approves(reply: str) -> None:
    """Given a scripted reply of `"y"`/`"yes"` in any casing, when
    `_prompt_stdin_writeback` reads it, then the decision is
    `"approve"`."""
    assert _prompt_stdin_writeback(_LEARNING, input_fn=lambda _: reply) == "approve"


@pytest.mark.parametrize("reply", ["", "n", "no", "N", "maybe", "approve please"])
def test_anything_else_skips(reply: str) -> None:
    """Given a scripted reply that is empty, an explicit decline, or any
    other non-affirmative text, when `_prompt_stdin_writeback` reads it,
    then the decision is `"skip"`."""
    assert _prompt_stdin_writeback(_LEARNING, input_fn=lambda _: reply) == "skip"


def test_renders_the_learnings_own_text_and_tags_before_prompting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a learning with known text and tags, when
    `_prompt_stdin_writeback` runs, then both the learning's own text and
    its tags are printed to stdout before `input_fn` is ever consulted."""
    _prompt_stdin_writeback(_LEARNING, input_fn=lambda _: "n")

    printed = capsys.readouterr().out
    assert _LEARNING.text in printed
    assert "style" in printed


@pytest.mark.redteam
def test_terminal_escapes_in_the_learnings_own_text_and_tags_are_stripped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a learning whose text and tags carry a clear-screen CSI
    sequence and a window-retitling OSC sequence -- exactly the shape a
    hostile proposal reply could smuggle in, since both fields are
    model-generated -- when `_prompt_stdin_writeback` renders it, then
    neither escape sequence reaches stdout, though the surrounding text
    still does: a human approving from a corrupted or retitled terminal
    can't make the informed decision this gate exists for."""
    hostile = ProposedLearning(
        text="before \x1b[2Jclear after",
        tags=("\x1b]0;evil title\x07tag",),
    )

    _prompt_stdin_writeback(hostile, input_fn=lambda _: "n")

    printed = capsys.readouterr().out
    assert "\x1b" not in printed
    assert "\x07" not in printed
    assert "before clear after" in printed
    assert "tag" in printed
