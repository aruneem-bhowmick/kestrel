"""Wires every tool this project offers a model into one schema list and
one dispatcher.

`all_schemas()` returns every tool's `ToolSchema`, in a fixed order,
ready to hand straight to a provider call's own `tools=` argument.
`dispatch()` takes one `ToolCallEvent` a model produced, looks up its
name, parses its arguments through that tool's own parser, calls its
executor, and returns a `ToolResult` -- catching that tool's own typed
error along the way, so a model's bad tool call becomes a normal,
recoverable result rather than a crash. Before this module existed,
each tool's schema, parser, and executor had no single place accounting
for all of them together; a caller wiring up a provider call or an
agent loop had to know every tool's own quirks by hand instead of going
through one seam.

To add a tool: implement its schema/parser/executor, then add exactly
one line to `_TOOLS`. Nothing else here -- or in any caller of
`all_schemas`/`dispatch` -- needs to change.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kestrel.provider.base import ToolSchema
from kestrel.provider.events import ToolCallEvent
from kestrel.security.framing import frame_untrusted
from kestrel.tools.edit_file import (
    EDIT_FILE_SCHEMA,
    EditFileError,
    edit_file,
    parse_edit_file_args,
)
from kestrel.tools.execute import (
    EXECUTE_SCHEMA,
    ExecuteError,
    execute,
    parse_execute_args,
)
from kestrel.tools.read_file import (
    READ_FILE_SCHEMA,
    ReadFileError,
    parse_read_file_args,
    read_file,
)
from kestrel.tools.search import (
    SEARCH_SCHEMA,
    SearchError,
    parse_search_args,
    search,
)
from kestrel.tools.verify import (
    VERIFY_SCHEMA,
    VerifyError,
    parse_verify_args,
    verify,
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One dispatched tool call's outcome, ready to become a tool-role message.

    Attributes:
        tool_call_id: Echoes the `ToolCallEvent.id` this result answers,
            so a caller can match it back to the request that produced it.
        content: The result text, already framed by
            `kestrel.security.framing.frame_untrusted` -- whether that
            framing happened inside the tool's own successful return, or
            was applied by `dispatch` itself for an unrecognized tool
            name, a malformed argument payload, or a caught tool error.
    """

    tool_call_id: str
    content: str


@dataclass(frozen=True, slots=True)
class _ToolBinding:
    """One tool's schema, argument parser, executor, and the error type
    both of those raise for a request they refuse.

    Not part of this module's public interface -- it exists only to
    give `_TOOLS` and `dispatch` a single shape to share, so adding a
    tool means adding one instance of this to `_TOOLS` rather than
    threading a new parallel list through every function here.
    """

    schema: ToolSchema
    parse_args: Callable[[str], Any]
    execute: Callable[..., str]
    error_type: type[Exception]


# Every tool this project offers a model, as one (schema, parser,
# executor, error type) binding. This tuple is the one place a new tool
# is registered: adding a tool means adding exactly one entry here, and
# nowhere else needs to change.
_TOOLS: Final[tuple[_ToolBinding, ...]] = (
    _ToolBinding(READ_FILE_SCHEMA, parse_read_file_args, read_file, ReadFileError),
    _ToolBinding(SEARCH_SCHEMA, parse_search_args, search, SearchError),
    _ToolBinding(EXECUTE_SCHEMA, parse_execute_args, execute, ExecuteError),
    _ToolBinding(EDIT_FILE_SCHEMA, parse_edit_file_args, edit_file, EditFileError),
    _ToolBinding(VERIFY_SCHEMA, parse_verify_args, verify, VerifyError),
)

_BY_NAME: Final[dict[str, _ToolBinding]] = {
    binding.schema.name: binding for binding in _TOOLS
}


def all_schemas() -> tuple[ToolSchema, ...]:
    """Every registered tool's schema, in `_TOOLS`'s fixed order --
    passed straight to `ProviderClient.complete(tools=...)`."""
    return tuple(binding.schema for binding in _TOOLS)


def _frame_error(tool_name: str, message: str) -> str:
    """Wrap `message` as untrusted `tool_stderr` content naming
    `tool_name`, matching the framing every tool's own successful
    result already carries -- so a caller downstream never has to tell
    an error result apart from a successful one before treating its
    content as data, never instructions."""
    return frame_untrusted(message, source="tool_stderr", origin=tool_name)


def _accepted_kwargs(executor: Callable[..., str]) -> frozenset[str]:
    """The keyword-only parameter names `executor` declares beyond its
    positional argument object -- the subset of a caller's superset
    `context` that `dispatch` actually passes through to it."""
    return frozenset(
        name
        for name, parameter in inspect.signature(executor).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )


def dispatch(event: ToolCallEvent, *, repo_root: Path, **context: Any) -> ToolResult:
    """Route `event` to its bound tool and return a framed `ToolResult`.

    Looks up `event.name` in `_TOOLS`; an unrecognized name returns a
    framed error result naming it, rather than raising -- a model
    hallucinating a tool name must be recoverable, not fatal. Parses
    `event.arguments_json` through the bound tool's own parser; a parse
    failure returns a framed error result the same way.

    Calls the bound executor with `repo_root` plus whichever of
    `context`'s entries it actually declares as keyword-only parameters
    (found via `inspect.signature`), so every caller passes one superset
    dict every time without needing to know which tool needs `approval`,
    `undo`, `turn_id`, or `task_id`, and which needs none of them.

    The tool's own typed error -- raised by either its parser or its
    executor -- is caught and framed as the result, exactly like the
    two refusal paths above. Every other exception (including a
    `TypeError` raised when an executor's own required keyword is
    missing from `context`) propagates unchanged: that signals a
    caller-side wiring mistake, not a recoverable model mistake, and
    must not be silently swallowed.
    """
    binding = _BY_NAME.get(event.name)
    if binding is None:
        return ToolResult(
            tool_call_id=event.id,
            content=_frame_error(
                event.name, f"{event.name!r} is not a registered tool"
            ),
        )

    try:
        args = binding.parse_args(event.arguments_json)
    except binding.error_type as exc:
        return ToolResult(
            tool_call_id=event.id, content=_frame_error(event.name, str(exc))
        )

    available: dict[str, Any] = {"repo_root": repo_root, **context}
    kwargs = {
        name: available[name]
        for name in _accepted_kwargs(binding.execute)
        if name in available
    }

    try:
        content = binding.execute(args, **kwargs)
    except binding.error_type as exc:
        return ToolResult(
            tool_call_id=event.id, content=_frame_error(event.name, str(exc))
        )

    return ToolResult(tool_call_id=event.id, content=content)
