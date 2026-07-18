"""Unit tests for `kestrel.tui.app.ArtifactPane.show_plan`: proves it
renders exactly `sanitize_terminal(render_plan_markdown(plan))`, the
same assertion shape `test_p040_artifact_pane.py` already established
for `show_report`, including a red-team case where a plan line carries
a real prompt-injection corpus payload and no raw control byte or
terminal escape sequence survives into the rendered output.
"""

from __future__ import annotations

import pytest

from kestrel.agent.plan import ImplementationPlan, PlanLine, render_plan_markdown
from kestrel.repl import sanitize_terminal
from kestrel.security.corpus import load_corpus
from kestrel.tui.app import ArtifactPane

pytestmark = [pytest.mark.p050, pytest.mark.unit]

_TASK_ID = "task-p050"


def _spy_update(pane: ArtifactPane, updates: list[str]) -> None:
    """Replace `pane.update` with a spy recording each call's own first
    (markdown) argument verbatim, instead of the real `Markdown.update`
    -- keeps these tests independent of Textual's own app-context
    requirement for an unmounted widget, mirroring
    `test_p040_artifact_pane.py`'s own helper."""
    pane.update = updates.append  # type: ignore[method-assign]


def _plan(*, lines: tuple[PlanLine, ...]) -> ImplementationPlan:
    """A minimal `ImplementationPlan` carrying `lines` verbatim; `raw_text`
    is never read by `render_plan_markdown`, so it is left empty."""
    return ImplementationPlan(task_id=_TASK_ID, raw_text="", lines=lines)


@pytest.mark.sanity
def test_show_plan_renders_the_plan_via_markdown_update() -> None:
    """Given a plan with several ordinary lines, when `show_plan` renders
    it, then `Markdown.update` is called exactly once with
    `sanitize_terminal(render_plan_markdown(plan))`."""
    pane = ArtifactPane()
    updates: list[str] = []
    _spy_update(pane, updates)
    plan = _plan(
        lines=(
            PlanLine(index=1, text="Read the existing config loader."),
            PlanLine(index=2, text="Add a new field for the retry timeout."),
        )
    )

    pane.show_plan(plan)

    assert updates == [sanitize_terminal(render_plan_markdown(plan))]


@pytest.mark.redteam
def test_show_plan_strips_a_prompt_injection_payload_with_no_bytes_surviving() -> None:
    """Given a plan whose one line carries the injection corpus's
    `readme_ignore_previous_instructions` payload verbatim, when
    `show_plan` renders it, then the update still matches
    `sanitize_terminal(render_plan_markdown(plan))` exactly, and no raw
    C0/C1 control byte or ANSI/CSI/OSC escape sequence survives into the
    rendered text -- proving the same terminal-escape guard `show_report`
    already carries applies identically to plan content, regardless of
    what that content is trying to say."""
    sentinel = object()
    payload = next(
        (
            case.payload
            for case in load_corpus()
            if case.id == "readme_ignore_previous_instructions"
        ),
        sentinel,
    )
    assert payload is not sentinel, "missing injection corpus case"

    pane = ArtifactPane()
    updates: list[str] = []
    _spy_update(pane, updates)
    plan = _plan(lines=(PlanLine(index=1, text=payload),))  # type: ignore[arg-type]

    pane.show_plan(plan)

    assert updates == [sanitize_terminal(render_plan_markdown(plan))]
    rendered = updates[0]
    assert not any(0x00 <= ord(ch) <= 0x1F and ch not in "\n\t" for ch in rendered)
    assert not any(0x80 <= ord(ch) <= 0x9F for ch in rendered)
    assert "\x1b" not in rendered
