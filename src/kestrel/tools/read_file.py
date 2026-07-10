"""Bounded, framed reads of a real repo file for a model to call as a tool.

`read_file` is the first concrete tool the agent loop can offer a model:
given a repo-relative path (and an optional 1-indexed, inclusive line
range), it returns that file's content already wrapped by
`kestrel.security.framing.frame_untrusted`, so a model can read code
without ever treating what it reads as instructions. No shared tool
dispatcher exists yet -- this module owns its own schema, argument
dataclass, and JSON-argument parsing so it is usable standalone; a later
dispatcher can register it alongside sibling tools without needing to
change any of this.

Every failure this tool can hit -- a path escaping the repo root
(including through a symlink), a missing file, a directory, binary
content, or a malformed argument -- raises `ReadFileError` with a
message meant to be handed straight back to the model, never a raw
traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kestrel.provider.base import ToolSchema
from kestrel.security.framing import frame_untrusted

# Returned content is capped at 64 KiB when no explicit line range is
# given; an explicit range is never truncated this way (the caller asked
# for exactly that slice).
_MAX_RETURNED_BYTES: Final[int] = 64 * 1024

# Only this much of a file is decoded to decide whether it is binary,
# so a large non-text file (an image, a binary blob) is rejected without
# paying to decode all of it.
_BINARY_GUARD_WINDOW: Final[int] = 8 * 1024

_ALLOWED_ARG_FIELDS: Final[frozenset[str]] = frozenset(
    {"path", "start_line", "end_line"}
)

READ_FILE_SCHEMA = ToolSchema(
    name="read_file",
    description="Read a UTF-8 text file, or a line range within it.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class ReadFileArgs:
    """One validated `read_file` call's arguments.

    Attributes:
        path: Repo-relative path to the file to read.
        start_line: 1-indexed, inclusive first line to return; `None`
            starts from the beginning of the file.
        end_line: 1-indexed, inclusive last line to return; `None` reads
            through the end of the file. A value past the file's last
            line clamps to it rather than erroring.
    """

    path: str
    start_line: int | None = None
    end_line: int | None = None


class ReadFileError(Exception):
    """Raised for a `read_file` request this tool refuses or cannot satisfy.

    `str(self)` is itself the message returned to the model -- every
    raise site names the offending path or argument rather than letting
    a lower-level exception (`OSError`, `UnicodeDecodeError`,
    `json.JSONDecodeError`) escape uninterpreted.
    """


def _resolve_within_repo_root(path: str, *, repo_root: Path) -> Path:
    """Resolve `path` against `repo_root`, following any symlink, and
    raise `ReadFileError` if the resolved location falls outside
    `repo_root` -- so neither a `..` climb nor a symlink pointing outside
    the root can read a file the caller does not own."""
    resolved_root = repo_root.resolve()
    candidate = (repo_root / path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ReadFileError(f"{path}: escapes the repository root")
    return candidate


def _slice_lines(
    text: str, *, start_line: int | None, end_line: int | None, path: str
) -> str:
    """Return the 1-indexed, inclusive `start_line..end_line` slice of
    `text`'s lines, clamping an `end_line` past EOF and raising
    `ReadFileError` for a `start_line` past EOF."""
    lines = text.splitlines(keepends=True)
    total = len(lines)
    first = start_line if start_line is not None else 1
    last = min(end_line, total) if end_line is not None else total

    if first > total:
        raise ReadFileError(
            f"{path}: start_line {first} is past the file's {total} lines"
        )

    return "".join(lines[first - 1 : last])


def _cap_to_max_bytes(text: str) -> str:
    """Truncate `text` to at most `_MAX_RETURNED_BYTES` UTF-8-encoded
    bytes, appending a trailing note naming how many bytes were cut when
    it does."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_RETURNED_BYTES:
        return text

    kept = encoded[:_MAX_RETURNED_BYTES].decode("utf-8", errors="ignore")
    cut = len(encoded) - len(kept.encode("utf-8"))
    return f"{kept}\n... [truncated: {cut} more bytes omitted]"


def read_file(args: ReadFileArgs, *, repo_root: Path) -> str:
    """Read `args.path` (optionally sliced to a line range) under
    `repo_root` and return it framed as untrusted file content.

    Raises:
        ReadFileError: `args.path` resolves outside `repo_root` (whether
            by `..` traversal or by following a symlink that points
            outside it); it does not exist or names a directory; the
            underlying OS-level read fails (e.g. a permissions error, or
            the file disappearing between the existence check above and
            the read itself); its content is not valid UTF-8 text
            (checked against the file's first 8 KiB, so binary content
            is rejected without decoding the whole file); or
            `args.start_line` is greater than `args.end_line`.

    When no line range is given, the returned content is capped at
    64 KiB, truncated with a trailing note naming how much was cut; an
    explicit line range is never truncated this way.
    """
    candidate = _resolve_within_repo_root(args.path, repo_root=repo_root)

    if not candidate.exists():
        raise ReadFileError(f"{args.path}: no such file")
    if candidate.is_dir():
        raise ReadFileError(f"{args.path}: is a directory")
    if (
        args.start_line is not None
        and args.end_line is not None
        and args.start_line > args.end_line
    ):
        raise ReadFileError(
            f"{args.path}: start_line {args.start_line} is after "
            f"end_line {args.end_line}"
        )

    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        raise ReadFileError(f"{args.path}: could not be read ({exc})") from exc

    try:
        raw[:_BINARY_GUARD_WINDOW].decode("utf-8")
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReadFileError(
            f"{args.path}: not valid UTF-8 text (binary guard)"
        ) from exc

    if args.start_line is not None or args.end_line is not None:
        body = _slice_lines(
            text, start_line=args.start_line, end_line=args.end_line, path=args.path
        )
    else:
        body = _cap_to_max_bytes(text)

    return frame_untrusted(body, source="file", origin=args.path)


def _parse_line_number(value: Any, *, field: str) -> int | None:
    """Validate an optional 1-indexed line-number field, raising
    `ReadFileError` naming `field` when it is present but not an integer
    greater than or equal to 1. `bool` is rejected even though it is a
    subclass of `int` in Python -- `true`/`false` in the source JSON is
    never a valid line number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReadFileError(f"arguments: '{field}' must be an integer >= 1")
    return value


def parse_read_file_args(arguments_json: str) -> ReadFileArgs:
    """Parse and validate one `ToolCallEvent.arguments_json` payload for
    `read_file` against `READ_FILE_SCHEMA`.

    Raises:
        ReadFileError: `arguments_json` is not valid JSON, is not a JSON
            object, is missing the required `path` field, carries a
            field `READ_FILE_SCHEMA` does not declare, or gives `path`,
            `start_line`, or `end_line` a value of the wrong type or
            range -- every case names the offending field, never a raw
            `json.JSONDecodeError` or `KeyError`.
    """
    try:
        raw: Any = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ReadFileError(f"arguments: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise ReadFileError("arguments: expected a JSON object")

    unexpected = sorted(set(raw) - _ALLOWED_ARG_FIELDS)
    if unexpected:
        raise ReadFileError(f"arguments: unexpected field(s) {unexpected}")

    if "path" not in raw:
        raise ReadFileError("arguments: missing required field 'path'")

    path = raw["path"]
    if not isinstance(path, str):
        raise ReadFileError("arguments: 'path' must be a string")

    return ReadFileArgs(
        path=path,
        start_line=_parse_line_number(raw.get("start_line"), field="start_line"),
        end_line=_parse_line_number(raw.get("end_line"), field="end_line"),
    )
