"""Turns a task's own description into one already-framed knowledge-base
context block, ready to seed a fresh task's history before its first turn.

`build_kb_context` is the single function this module exports: it embeds
`task_description` as a query, searches the caller's `KbService`, renders
whatever it finds as one line per note, and wraps the result through
`kestrel.security.framing.frame_untrusted` under the `"kb"` source kind --
the same choke point every other untrusted byte (a file read, a tool's
stdout, a web page) already passes through, since a retrieved note did
not come from the user typing it directly. A disabled knowledge base, an
empty result, or a knowledge-base outage all collapse to the same `None`
return, which `kestrel.agent.loop.run_task`'s own `kb_context` parameter
already treats as a no-op -- retrieval failing never fails the task it
would have helped.
"""

from __future__ import annotations

from kestrel.kb.service import KbService, KbServiceError
from kestrel.security.framing import frame_untrusted


async def build_kb_context(task_description: str, *, kb: KbService | None) -> str | None:
    """Retrieve knowledge-base notes relevant to `task_description` and
    render them as one already-framed context block.

    Returns `None` when `kb` is `None` (the knowledge base is disabled
    for this task), when the search itself returns no notes, or when the
    search raises `KbServiceError` -- a knowledge-base outage degrades to
    "as if retrieval found nothing" rather than failing the caller before
    a task's first turn even runs, matching `search`'s own `_kb_hits`
    degradation. Otherwise, every returned note's own text is rendered as
    one `"- {text}"` line, most similar first, and the joined result is
    passed through `frame_untrusted(..., source="kb",
    origin=kb.embedding_model_id)` -- the caller receives an
    already-framed string, never raw note text.
    """
    if kb is None:
        return None
    try:
        scored = await kb.search(task_description)
    except KbServiceError:
        return None
    if not scored:
        return None
    rendered = "\n".join(f"- {s.note.text}" for s in scored)
    return frame_untrusted(rendered, source="kb", origin=kb.embedding_model_id)
