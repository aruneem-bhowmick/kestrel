"""Anchor-based, journaled text edits to a real repo file for a model
to call as a tool.

`edit_file` replaces exactly one, unique occurrence of an exact anchor
string in a UTF-8 text file with new text. It never guesses which
occurrence a caller means: an anchor that appears zero times, or more
than once, is refused outright and the file is left untouched either
way. Every real (non-dry-run) edit is recorded through
`kestrel.managers.undo.UndoManager` before this function returns, so it
can always be reversed later; a `dry_run` call instead returns a
unified diff of what the edit *would* do, without touching the file or
the undo journal at all.

Path containment and the binary-content guard are shared with
`read_file` via `kestrel.tools._paths.resolve_repo_path`, so both tools
refuse a `..` climb or a symlink escaping the repo root identically.
Like its sibling tools, this module owns its own schema, argument
dataclass, and JSON-argument parsing, and raises `EditFileError` --
never a raw exception -- for anything it refuses.

Creating a new file through `edit_file` is out of scope: an anchor
checked against a path with no file on disk is refused the same way a
genuinely absent anchor is, rather than being interpreted as "create
this file with `new`'s content."
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kestrel.managers.undo import UndoEntry, UndoManager
from kestrel.provider.base import ToolSchema
from kestrel.security.framing import frame_untrusted
from kestrel.tools._paths import resolve_repo_path

_ALLOWED_ARG_FIELDS: Final[frozenset[str]] = frozenset(
    {"path", "old", "new", "dry_run"}
)

EDIT_FILE_SCHEMA = ToolSchema(
    name="edit_file",
    description="Replace one exact, unique occurrence of `old` with `new` in a file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "old": {
                "type": "string",
                "description": "Exact, unique anchor text to replace.",
            },
            "new": {"type": "string", "description": "Replacement text."},
            "dry_run": {
                "type": "boolean",
                "description": "Return a diff instead of writing the change.",
            },
        },
        "required": ["path", "old", "new"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class EditFileArgs:
    """One validated `edit_file` call's arguments.

    Attributes:
        path: Repo-relative path to the file to edit.
        old: The exact anchor text that must occur exactly once in the
            file's current content.
        new: The text to replace `old` with.
        dry_run: When `True`, return a unified diff of the change
            instead of writing it and recording it to the undo journal.
    """

    path: str
    old: str
    new: str
    dry_run: bool = False


class EditFileError(Exception):
    """Raised for an `edit_file` request this tool refuses or cannot satisfy.

    Covers an anchor that is absent or not unique, and every path or
    binary-content refusal `read_file` would also raise for the same
    underlying condition. `str(self)` is itself the message returned to
    the model, never a raw traceback.
    """


def _resolve_within_repo_root(path: str, *, repo_root: Path) -> Path:
    """Resolve `path` under `repo_root` through the shared
    `resolve_repo_path` containment guard, raising `EditFileError`
    (rather than that helper's own `ValueError`) when `path` escapes
    the repo root -- whether by a `..` climb or by following a symlink
    that points outside it."""
    try:
        return resolve_repo_path(path, repo_root=repo_root)
    except ValueError as exc:
        raise EditFileError(str(exc)) from exc


def _read_existing_text(candidate: Path, *, path: str) -> str:
    """Read `candidate` fully and decode it as UTF-8.

    Raises:
        EditFileError: `candidate` does not exist, names a directory,
            the OS-level read fails, or its content is not valid UTF-8
            text (the binary guard) -- the same refusals `read_file`
            raises for the same conditions.
    """
    if not candidate.exists():
        raise EditFileError(f"{path}: no such file")
    if candidate.is_dir():
        raise EditFileError(f"{path}: is a directory")
    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        raise EditFileError(f"{path}: could not be read ({exc})") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EditFileError(f"{path}: not valid UTF-8 text (binary guard)") from exc


def _render_diff(*, path: str, before: str, after: str) -> str:
    """Render the unified diff between `before` and `after`'s lines,
    labeling both sides with `path` -- the exact text `edit_file`
    returns for a `dry_run` call, without ever writing `after` to disk
    or touching the undo journal."""
    diff_lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
    )
    return "".join(diff_lines)


def edit_file(
    args: EditFileArgs,
    *,
    repo_root: Path,
    undo: UndoManager,
    turn_id: int,
    task_id: str,
) -> str:
    """Replace the one, unique occurrence of `args.old` in `args.path`
    with `args.new`, and return the outcome framed as untrusted tool
    output.

    Reads `args.path` under `repo_root` through the shared
    `resolve_repo_path` containment guard and the same binary-content
    guard `read_file` applies, then counts occurrences of `args.old` in
    the file's current content: zero raises `EditFileError` naming the
    anchor as not found, and more than one raises `EditFileError`
    naming the exact count -- this function never guesses which
    occurrence a caller means.

    When `args.dry_run` is `True`, returns a unified diff of the change
    without writing anything or recording it to `undo`. Otherwise,
    writes the new content to disk first and only then calls
    `undo.record` -- not the reverse -- so that a journal-write failure
    after a successful file write leaves a recoverable, half-finished
    operation (the file really changed; a human can inspect and repair
    the journal by hand), whereas recording before writing could let
    the journal claim a write that never actually happened.

    Raises:
        EditFileError: `args.path` escapes `repo_root` (by a `..`
            climb or a symlink pointing outside it); it does not exist
            or names a directory; its content is not valid UTF-8 (the
            binary guard); `args.old` occurs zero times or more than
            once in the file's current content; or the OS-level write
            fails once a unique anchor has been found. The file is
            left untouched in every one of these cases.
    """
    candidate = _resolve_within_repo_root(args.path, repo_root=repo_root)
    content = _read_existing_text(candidate, path=args.path)

    occurrences = content.count(args.old)
    if occurrences == 0:
        raise EditFileError(f"{args.path}: anchor not found")
    if occurrences > 1:
        raise EditFileError(
            f"{args.path}: anchor not unique ({occurrences} occurrences)"
        )

    new_content = content.replace(args.old, args.new, 1)

    if args.dry_run:
        diff = _render_diff(path=args.path, before=content, after=new_content)
        return frame_untrusted(diff, source="tool_stdout", origin=args.path)

    try:
        candidate.write_bytes(new_content.encode("utf-8"))
    except OSError as exc:
        raise EditFileError(f"{args.path}: could not be written ({exc})") from exc
    undo.record(
        UndoEntry(
            turn_id=turn_id,
            task_id=task_id,
            path=args.path,
            before=content,
            after=new_content,
        )
    )

    confirmation = f"{args.path}: replaced the anchor (1 occurrence)."
    return frame_untrusted(confirmation, source="tool_stdout", origin=args.path)


def _parse_dry_run(value: Any) -> bool:
    """Validate the optional `dry_run` field, defaulting to `False`
    when absent and raising `EditFileError` when present but not a
    boolean."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise EditFileError("arguments: 'dry_run' must be a boolean")
    return value


def parse_edit_file_args(arguments_json: str) -> EditFileArgs:
    """Parse and validate one `ToolCallEvent.arguments_json` payload for
    `edit_file` against `EDIT_FILE_SCHEMA`.

    Raises:
        EditFileError: `arguments_json` is not valid JSON, is not a
            JSON object, is missing a required field (`path`, `old`, or
            `new`), carries a field `EDIT_FILE_SCHEMA` does not
            declare, or gives any field a value of the wrong type --
            every case names the offending field, never a raw
            `json.JSONDecodeError` or `KeyError`.
    """
    try:
        raw: Any = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise EditFileError(f"arguments: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise EditFileError("arguments: expected a JSON object")

    unexpected = sorted(set(raw) - _ALLOWED_ARG_FIELDS)
    if unexpected:
        raise EditFileError(f"arguments: unexpected field(s) {unexpected}")

    for field in ("path", "old", "new"):
        if field not in raw:
            raise EditFileError(f"arguments: missing required field '{field}'")
        if not isinstance(raw[field], str):
            raise EditFileError(f"arguments: '{field}' must be a string")

    if raw["old"] == "":
        raise EditFileError("arguments: 'old' must not be empty")

    return EditFileArgs(
        path=raw["path"],
        old=raw["old"],
        new=raw["new"],
        dry_run=_parse_dry_run(raw.get("dry_run")),
    )
