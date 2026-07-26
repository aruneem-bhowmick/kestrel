"""Propose durable learnings from a finished task, and commit only the
ones a human explicitly approves.

`propose_learnings` sends one short, non-streamed model call -- the same
shape `kestrel.agent.critique._critique_async` already uses for its own
narrow, capped question -- asking the model to distill at most three
durable, reusable learnings out of a task's own `Walkthrough`. Nothing
proposed here is persisted automatically: `commit_learnings` writes only
the entries a caller-supplied set of `WritebackDecision`s marks
`"approve"`, via `KbService.add_note`. Proposal and decision-gathering
are kept as two separate calls a caller composes -- this module never
calls a decision function itself -- so a caller can render proposals,
collect approvals through whatever UI it has (a terminal prompt via
`_prompt_stdin_writeback`, a modal, a batch script), and only then commit,
without this module ever assuming which.

A proposal reply is untrusted, model-generated text, parsed as data:
`_parse_proposal_line` never raises on a malformed or hostile line, it
just contributes nothing to the result -- the safe failure mode for a
parser is "propose nothing", since nothing proposed is ever committed
without a human separately approving it regardless of what parsed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from kestrel.agent.walkthrough import Walkthrough, render_walkthrough_markdown
from kestrel.kb.service import KbService
from kestrel.kb.store import KnowledgeNote
from kestrel.provider.base import Message, ProviderClient
from kestrel.provider.errors import ProviderError
from kestrel.provider.events import StreamEvent, TextDelta
from kestrel.provider.retry import complete_with_retry
from kestrel.repl import sanitize_terminal

_WRITEBACK_SYSTEM_PROMPT: Final[str] = (
    "You just finished a coding task. Propose at most three short, "
    "durable, reusable learnings future tasks in this repo would "
    "benefit from knowing -- conventions, gotchas, fixes. Reply with "
    "one learning per line, each formatted exactly as "
    "'LEARNING: <text> | TAGS: <comma-separated tags>'. If nothing "
    "durable happened, reply with exactly 'NONE'."
)
_WRITEBACK_MAX_TOKENS: Final[int] = 256
_MAX_LEARNINGS: Final[int] = 3
_LEARNING_PREFIX: Final[str] = "LEARNING:"
_TAGS_SEPARATOR: Final[str] = "| TAGS:"


@dataclass(frozen=True, slots=True)
class ProposedLearning:
    """One model-proposed, not-yet-approved learning.

    Attributes:
        text: The learning's own text.
        tags: Free-form tags the model itself proposed alongside it.
    """

    text: str
    tags: tuple[str, ...]


class WritebackError(Exception):
    """The proposal call failed, or its reply could not be parsed as the
    documented `LEARNING: ... | TAGS: ...` line format. `str(self)`
    names the remedy."""


def _parse_proposal_line(line: str) -> ProposedLearning | None:
    """Parse one proposal reply line into a `ProposedLearning`.

    Only a line whose stripped form starts with `"LEARNING:"` is
    considered -- a blank line, the literal `"NONE"` reply, and any other
    non-conforming line all return `None`. A line that does start with
    the prefix but is missing the `"| TAGS:"` separator, or whose own
    text is empty once stripped (e.g. `"LEARNING: | TAGS: x"`), is
    treated as malformed and also returns `None`, rather than raising:
    one bad line degrades to "no learning from this line" instead of
    failing the whole batch. Tags are split on commas and stripped, so a
    trailing `"TAGS: "` with nothing after it yields an empty tags tuple
    rather than a tuple holding one blank string.

    This function never raises, regardless of its input -- it is the
    first thing a hostile or malformed model reply reaches.
    """
    stripped = line.strip()
    if not stripped.startswith(_LEARNING_PREFIX):
        return None
    body = stripped[len(_LEARNING_PREFIX) :]
    if _TAGS_SEPARATOR not in body:
        return None
    text_part, _, tags_part = body.partition(_TAGS_SEPARATOR)
    text = text_part.strip()
    if not text:
        return None
    tags = tuple(tag.strip() for tag in tags_part.split(",") if tag.strip())
    return ProposedLearning(text=text, tags=tags)


async def propose_learnings(
    walkthrough: Walkthrough, *, client: ProviderClient, model_id: str
) -> tuple[ProposedLearning, ...]:
    """Ask the model to propose durable learnings from `walkthrough`.

    Sends one short, non-streamed completion -- `stream=False`, a fixed
    system prompt, `max_tokens=_WRITEBACK_MAX_TOKENS` -- mirroring
    `agent.critique._critique_async`'s own shape. `walkthrough` is
    rendered via `render_walkthrough_markdown` and passed as the user
    turn's own content: a task's own Walkthrough is first-party, model-
    and tool-derived content already surfaced to the same user this
    session belongs to, so it needs no untrusted-content framing here.

    The reply is split into lines, each parsed via
    `_parse_proposal_line`; lines that fail to parse (including a bare
    `"NONE"` reply) contribute nothing, and the result is capped at
    `_MAX_LEARNINGS` entries -- any further well-formed lines beyond the
    third are silently dropped, not an error.

    Raises:
        WritebackError: the underlying model call fails.
    """
    messages: list[Message] = [
        {"role": "system", "content": _WRITEBACK_SYSTEM_PROMPT},
        {"role": "user", "content": render_walkthrough_markdown(walkthrough)},
    ]
    events: list[StreamEvent] = []
    try:
        async for event in complete_with_retry(
            client,
            messages,
            None,
            model_id,
            "high",
            stream=False,
            max_tokens=_WRITEBACK_MAX_TOKENS,
        ):
            events.append(event)
    except ProviderError as exc:
        raise WritebackError(f"propose_learnings: model call failed: {exc}") from exc

    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    parsed = [
        learning
        for line in text.splitlines()
        if (learning := _parse_proposal_line(line)) is not None
    ]
    return tuple(parsed[:_MAX_LEARNINGS])


WritebackDecision = Literal["approve", "skip"]


def _prompt_stdin_writeback(
    learning: ProposedLearning, *, input_fn: Callable[[str], str] = input
) -> WritebackDecision:
    """Render `learning` on the real terminal and read one line of reply.

    Mirrors `kestrel.managers.approval._prompt_stdin`'s own shape: prints
    `learning.text` and `learning.tags`, then reads a single line via
    `input_fn` (defaulting to the built-in `input`), matched case-
    insensitively. `"y"`/`"yes"` decides `"approve"`; anything else,
    including an empty line, decides `"skip"`. Both printed fields are
    run through `sanitize_terminal` first: `learning.text`/`learning.tags`
    are model-generated text a hostile reply could load with terminal
    control sequences, and a human approving from a corrupted or hidden
    prompt is exactly what this gate exists to prevent.
    """
    print(f"Proposed learning: {sanitize_terminal(learning.text)}")
    print(f"Tags: {', '.join(sanitize_terminal(tag) for tag in learning.tags)}")
    reply = input_fn("Commit this learning? [y]es / [N]o: ").strip().lower()
    if reply in ("y", "yes"):
        return "approve"
    return "skip"


async def commit_learnings(
    learnings: Sequence[ProposedLearning],
    *,
    decisions: Sequence[WritebackDecision],
    task_id: str,
    kb: KbService,
) -> tuple[KnowledgeNote, ...]:
    """Commit only the entries of `learnings` a human approved.

    `decisions` must be the same length as `learnings`, matched by
    index: `learnings[i]` is committed via `kb.add_note(text=learnings[i]
    .text, tags=learnings[i].tags, source_task=task_id)` only when
    `decisions[i] == "approve"`. `KbService.add_note` itself may return
    more than one persisted `KnowledgeNote` per call when the global
    namespace is enabled; this function flattens every persisted note
    across every committed learning into one tuple, in commit order.

    Raises:
        ValueError: `len(learnings) != len(decisions)`.
        KbServiceError: propagated from `kb.add_note` unchanged.
    """
    if len(learnings) != len(decisions):
        raise ValueError(
            f"commit_learnings: {len(learnings)} learnings but "
            f"{len(decisions)} decisions -- these must be the same length, "
            f"matched by index"
        )

    persisted: list[KnowledgeNote] = []
    for learning, decision in zip(learnings, decisions):
        if decision != "approve":
            continue
        persisted.extend(
            await kb.add_note(learning.text, tags=learning.tags, source_task=task_id)
        )
    return tuple(persisted)
