"""Unit tests for `kestrel.tui.approval_modal.ApprovalModal`: its
`compose()` renders `request.summary`/`request.detail` (sanitized) into
the documented widget ids, and each of its three decision paths --
`action_decide_once`/`_always`/`_deny`, and a `Button.Pressed` for each
of the three button ids `compose` yields -- dismisses with the matching
`ApprovalDecision`.

Every case here constructs `ApprovalModal` directly and reads back its
own `compose()` return value or spies on `dismiss`, without ever
mounting a real Textual app: `Screen.dismiss` calls `self.app.pop_screen()`,
which requires a screen actually pushed onto a running app, so the
dismiss-mapping cases replace `dismiss` with a spy on the unmounted
instance instead of driving a real push/pop cycle -- `tests/system/
test_p042_approval_modal_live.py` is what proves the real, mounted
push-and-dismiss path end to end. `compose()`'s own return value is
inspected via Textual's pre-mount `_pending_children` bookkeeping, the
only place a widget's children live before an app ever mounts them.

Also covers two red-team cases: `request.detail` carrying the
`ansi_escape_laden_payload` injection-corpus case's own hostile ANSI
payload -- a realistic scenario, since `detail` is a model's own
proposed argv joined verbatim -- must never survive into the rendered
`Static`'s content; and a `summary`/`detail` carrying a Rich
console-markup span (e.g. `[conceal]...[/conceal]`), which
`sanitize_terminal` does not strip, must render as inert plain text
rather than being interpreted, since both `Static` widgets are built
with `markup=False`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from textual.widgets import Button, Static

from kestrel.managers.approval import ApprovalDecision, ApprovalRequest
from kestrel.repl import sanitize_terminal
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tui.approval_modal import ApprovalModal

pytestmark = [pytest.mark.p042, pytest.mark.unit, pytest.mark.sanity]

_HOSTILE_CASE_ID = "ansi_escape_laden_payload"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


def _request(
    *, summary: str = "Delete: rm somefile", detail: str = "rm somefile"
) -> ApprovalRequest:
    """A plain `"delete"`-kind request, defaulting to a realistic
    `rm` case matching `classify_destructive_action`'s own rendering."""
    return ApprovalRequest(kind="delete", summary=summary, detail=detail)


def _walk_pending(widget: object) -> Iterator[object]:
    """Yield `widget` and every descendant Textual will mount later,
    read from the compose-time `_pending_children` bookkeeping every
    container stores before it is ever actually mounted -- the only
    way to reach a `compose()` return value's nested widgets without
    mounting a real app."""
    yield widget
    for child in getattr(widget, "_pending_children", ()):
        yield from _walk_pending(child)


def _widgets_by_id(request: ApprovalRequest) -> dict[str, object]:
    """Compose `ApprovalModal(request)` and return every id-bearing
    widget it (recursively) yields, keyed by that widget's own id."""
    modal = ApprovalModal(request)
    (root,) = modal.compose()
    return {
        widget.id: widget  # type: ignore[attr-defined]
        for widget in _walk_pending(root)
        if getattr(widget, "id", None) is not None
    }


def test_compose_renders_the_summary_into_the_documented_id() -> None:
    """Given a request, when the modal composes, then
    `#approval_summary` holds `sanitize_terminal(request.summary)`."""
    request = _request(summary="Delete: rm -rf build/")
    widgets = _widgets_by_id(request)

    summary_widget = widgets["approval_summary"]
    assert isinstance(summary_widget, Static)
    assert summary_widget.content == sanitize_terminal(request.summary)


def test_compose_renders_the_detail_into_the_documented_id() -> None:
    """Given a request, when the modal composes, then
    `#approval_detail` holds `sanitize_terminal(request.detail)`."""
    request = _request(detail="rm -rf build/")
    widgets = _widgets_by_id(request)

    detail_widget = widgets["approval_detail"]
    assert isinstance(detail_widget, Static)
    assert detail_widget.content == sanitize_terminal(request.detail)


def test_compose_renders_three_decision_buttons_with_documented_ids() -> None:
    """Given a request, when the modal composes, then exactly the three
    documented buttons -- `once`, `always`, `deny` -- are present."""
    widgets = _widgets_by_id(_request())

    for button_id in ("once", "always", "deny"):
        assert isinstance(widgets[button_id], Button)


@pytest.mark.parametrize(
    ("action_name", "expected"),
    [
        ("action_decide_once", "once"),
        ("action_decide_always", "always"),
        ("action_decide_deny", "deny"),
    ],
)
def test_action_dismisses_with_the_matching_decision(
    action_name: str,
    expected: ApprovalDecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal, when `action_name` is invoked directly, then
    `dismiss` is called exactly once with `expected`."""
    modal = ApprovalModal(_request())
    dismissed: list[ApprovalDecision] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)

    getattr(modal, action_name)()

    assert dismissed == [expected]


@pytest.mark.parametrize(
    ("button_id", "expected"),
    [
        ("once", "once"),
        ("always", "always"),
        ("deny", "deny"),
    ],
)
def test_button_pressed_dismisses_with_the_matching_decision(
    button_id: str,
    expected: ApprovalDecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a modal, when a `Button.Pressed` event for `button_id`
    fires, then `dismiss` is called exactly once with `expected`."""
    modal = ApprovalModal(_request())
    dismissed: list[ApprovalDecision] = []
    monkeypatch.setattr(modal, "dismiss", dismissed.append)
    button = Button("label", id=button_id)

    modal.on_button_pressed(Button.Pressed(button))

    assert dismissed == [expected]


@pytest.mark.redteam
def test_hostile_detail_never_survives_into_the_rendered_static() -> None:
    """Given a `request.detail` carrying the `ansi_escape_laden_payload`
    corpus case's own hostile payload -- the realistic shape, since
    `detail` is a model's own proposed argv joined verbatim -- when the
    modal composes, then the rendered `#approval_detail` content carries
    none of the payload's raw escape bytes."""
    case = _find_case(_HOSTILE_CASE_ID)
    request = _request(detail=case.payload)

    widgets = _widgets_by_id(request)
    detail_widget = widgets["approval_detail"]
    assert isinstance(detail_widget, Static)
    content = detail_widget.content
    assert isinstance(content, str)

    assert content == sanitize_terminal(case.payload)
    assert "\x1b" not in content
    assert "\x9b" not in content
    assert "\x07" not in content


@pytest.mark.redteam
@pytest.mark.parametrize("widget_id", ["approval_summary", "approval_detail"])
def test_rich_markup_in_request_text_is_never_interpreted(widget_id: str) -> None:
    """Given a `summary`/`detail` carrying a Rich console-markup span --
    `[conceal]...[/conceal]`, a realistic shape since `sanitize_terminal`
    strips terminal escapes but never touches plain bracket syntax --
    when the modal composes, then the rendered widget's own `visual`
    shows the markup tags verbatim, as plain text with no style spans,
    rather than interpreting `conceal` and hiding the enclosed text from
    the approver."""
    hostile = "rm [conceal]--no-preserve-root[/conceal] -rf /"
    request = _request(summary=hostile, detail=hostile)

    widgets = _widgets_by_id(request)
    widget = widgets[widget_id]
    assert isinstance(widget, Static)

    visual = widget.visual
    assert visual.plain == hostile
    assert visual.spans == []
