"""Delimits untrusted external text before it enters a prompt.

Every byte that did not come from the user's own typed input -- file
contents, tool stdout/stderr, search results, and (once a web tool
exists) web content -- is data, never instructions. :func:`frame_untrusted`
is the single choke point every such byte passes through: it wraps the
text in a fixed, recognizable marker pair, ``<<<UNTRUSTED:{source}:{origin}>>>``
/ ``<<<END_UNTRUSTED>>>``, chosen to be vanishingly unlikely to appear in
real file or tool content by accident. Any occurrence of the three-byte
run ``<<<`` inside the wrapped text or its origin -- the shared prefix of
both markers -- is itself broken by an interposed zero-width character,
so hostile content can neither forge a fake closing delimiter nor spoof a
second, differently-sourced block that only looks well-formed. Origin
gets one more escaping pass on top of that, breaking ``>>>`` runs too,
since it is embedded inside the header's own opening/closing pair and an
unescaped ``>>>`` there could make the header appear to close early.
This module is pure and synchronous; it knows nothing about
:class:`~kestrel.provider.base.ProviderClient` or ``Message`` -- callers
place the framed string into a message's content themselves.
"""

from __future__ import annotations

from typing import Final, Literal

SourceKind = Literal["file", "tool_stdout", "tool_stderr", "search_result", "web"]

_OPEN_TEMPLATE: Final[str] = "<<<UNTRUSTED:{source}:{origin}>>>"
_CLOSE_MARKER: Final[str] = "<<<END_UNTRUSTED>>>"

# U+200B ZERO WIDTH SPACE, built from its code point (rather than the raw
# glyph) so the source file stays plain ASCII and diffable: invisible in
# any renderer a model or terminal would show, but enough to break a
# literal `<<<` run so no substring of an escaped string can equal -- or
# begin -- either marker.
_MARKER_BREAK: Final[str] = chr(0x200B)


def _break_markers(value: str) -> str:
    """Interpose a zero-width character inside every ``<<<`` run in ``value``.

    Both the opening and closing markers share the same three-character
    prefix, so breaking every occurrence of that prefix -- wherever it
    appears, whether it is trying to imitate the real closing delimiter
    or forge a whole second opening block -- is sufficient to defeat
    both attacks with one rule.
    """
    return value.replace("<<<", f"<<{_MARKER_BREAK}<")


def _break_closing_run(value: str) -> str:
    """Interpose a zero-width character inside every ``>>>`` run in ``value``.

    Only :func:`_escape_origin` needs this on top of :func:`_break_markers`:
    origin sits between the header's own ``<<<UNTRUSTED:`` prefix and its
    closing ``>>>``, so an unescaped ``>>>`` inside it would make the
    header appear -- to a naive parser or a pattern-matching reader -- to
    close earlier than it actually does, stranding the rest of origin as
    unframed trailing text on the same line. Body text never needs this
    independently: every marker that matters there, the real closing
    delimiter and a forged nested block alike, begins with ``<<<``, which
    :func:`_break_markers` already catches.
    """
    return value.replace(">>>", f">>{_MARKER_BREAK}>")


def _escape_origin(origin: str) -> str:
    """Escape ``origin`` for safe embedding inside the opening marker's line.

    ``origin`` is rendered verbatim otherwise (callers are responsible
    for not passing secrets as ``origin``); this collapses newlines to a
    visible backslash escape, so a hostile origin cannot split the
    single-line header into extra lines, and breaks both delimiter-forming
    runs -- ``<<<`` exactly as the body text is escaped, and ``>>>`` so
    origin cannot forge the header's own early close (see
    :func:`_break_closing_run`).
    """
    collapsed = (
        origin.replace("\r\n", "\\r\\n").replace("\n", "\\n").replace("\r", "\\r")
    )
    return _break_closing_run(_break_markers(collapsed))


def frame_untrusted(text: str, *, source: SourceKind, origin: str) -> str:
    """Wrap `text` in a delimited block naming its kind and origin
    (a path, a command, a URL), with an explicit instruction that its
    contents are data, never commands. `origin` is rendered verbatim
    (already-sanitized paths/commands only -- callers are responsible
    for not passing secrets as `origin`). The delimiter is a fixed,
    recognizable marker pair (`<<<UNTRUSTED:{source}:{origin}>>>` /
    `<<<END_UNTRUSTED>>>`) chosen to be vanishingly unlikely to appear
    in real file/tool content by accident, and any occurrence of the
    literal marker inside `text` is itself escaped (prefixed with a
    zero-width marker-breaking character) so content cannot forge a
    fake closing delimiter and "escape" the frame.
    """
    header = _OPEN_TEMPLATE.format(source=source, origin=_escape_origin(origin))
    safe_text = _break_markers(text)
    return f"{header}\n{safe_text}\n{_CLOSE_MARKER}"
