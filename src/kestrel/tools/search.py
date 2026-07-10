"""Regex search over repo file contents for a model to call as a tool.

`search` shells out to `rg` (ripgrep) for a regex pattern, optionally
scoped to a repo-relative subdirectory, and returns the matched lines
already wrapped by `kestrel.security.framing.frame_untrusted` -- so a
model can search code without ever treating a match's content as
instructions. This is the first tool in this package to run an external
binary rather than touch the filesystem directly.

Like `read_file`, this module owns its own schema, argument dataclass,
and JSON-argument parsing; no shared tool dispatcher exists yet, so it
is usable standalone.

Every failure this tool can hit -- a scope escaping the repo root, an
`rg` invocation that cannot run at all, or a pattern `rg` rejects as
invalid regex -- raises `SearchError` with a message meant to be handed
straight back to the model, never a raw traceback or subprocess exit
code. `rg` exiting 1 (its documented "no matches" exit code) is not a
failure: it is a normal, framed empty-results response.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kestrel.provider.base import ToolSchema
from kestrel.security.framing import frame_untrusted

# `rg` exit codes: 0 means at least one match, 1 means the search ran
# cleanly but found nothing (a normal outcome, not a failure), and
# anything else (2, conventionally) is a real error -- a bad pattern, a
# missing scope, or similar -- worth surfacing as SearchError.
_RG_EXIT_NO_MATCHES: Final[int] = 1

# A single matched line is capped at this many characters before being
# rendered, so one abnormally long line (a minified bundle, a data file)
# can never dominate the tool's response on its own.
_MAX_LINE_CHARS: Final[int] = 300

_DEFAULT_MAX_RESULTS: Final[int] = 50
_MIN_MAX_RESULTS: Final[int] = 1
_MAX_MAX_RESULTS: Final[int] = 200

_ALLOWED_ARG_FIELDS: Final[frozenset[str]] = frozenset(
    {"pattern", "scope", "max_results"}
)

SEARCH_SCHEMA = ToolSchema(
    name="search",
    description="Search file contents under the repo for a regex pattern.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "scope": {
                "type": "string",
                "description": "Repo-relative subdirectory; defaults to the repo root.",
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class SearchArgs:
    """One validated `search` call's arguments.

    Attributes:
        pattern: The regex pattern to search for, passed to `rg` verbatim.
        scope: Repo-relative subdirectory to search under; `None`
            searches the whole repo.
        max_results: Maximum number of hits to return, in file order;
            excess hits beyond this count are dropped, not an error.
    """

    pattern: str
    scope: str | None = None
    max_results: int = _DEFAULT_MAX_RESULTS


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One matched line from an `rg` invocation.

    Attributes:
        path: The matched file's path, relative to `repo_root` (or to
            `scope`, when one was given).
        line_number: The matched line's 1-indexed line number within `path`.
        line_text: The matched line's text, capped at `_MAX_LINE_CHARS`
            characters (truncated with a trailing marker when longer).
    """

    path: str
    line_number: int
    line_text: str


class SearchError(Exception):
    """Raised for a request this tool refuses or a failed rg invocation."""


def _resolve_within_repo_root(scope: str, *, repo_root: Path) -> None:
    """Raise `SearchError` if `scope`, resolved against `repo_root` and
    following any symlink, falls outside `repo_root` -- the same
    containment check `read_file` applies to its own `path` argument, so
    neither a `..` climb nor a symlink pointing outside the root can
    scope a search to a location the caller does not own."""
    resolved_root = repo_root.resolve()
    candidate = (repo_root / scope).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise SearchError(f"{scope}: escapes the repository root")


def _truncate_line(text: str) -> str:
    """Cap `text` at `_MAX_LINE_CHARS` characters, appending a trailing
    marker naming how many characters were cut when it is longer than
    that; text at or under the cap is returned unchanged."""
    if len(text) <= _MAX_LINE_CHARS:
        return text
    omitted = len(text) - _MAX_LINE_CHARS
    return f"{text[:_MAX_LINE_CHARS]}... [truncated: {omitted} more chars omitted]"


def _parse_hits(rg_stdout: str) -> list[SearchHit]:
    """Parse `rg --line-number --no-heading` output into `SearchHit`s, in
    the order `rg` printed them. Each line is split into at most three
    parts (`path`, `line_number`, `text`) so a matched line's own text
    may freely contain colons without corrupting the split."""
    hits: list[SearchHit] = []
    for line in rg_stdout.splitlines():
        if not line:
            continue
        path, _, remainder = line.partition(":")
        line_number_str, _, text = remainder.partition(":")
        hits.append(
            SearchHit(
                path=path,
                line_number=int(line_number_str),
                line_text=_truncate_line(text),
            )
        )
    return hits


def _render_hit(hit: SearchHit) -> str:
    """Render one `SearchHit` as the single-line `path:line: text` form
    this tool returns, one such line per hit."""
    return f"{hit.path}:{hit.line_number}: {hit.line_text}"


def search(args: SearchArgs, *, repo_root: Path) -> str:
    """Run `rg` for `args.pattern` under `repo_root` (or `args.scope`
    within it) and return the matches framed as untrusted search results.

    Invokes `rg --line-number --no-heading --sort path` -- the sort flag
    makes hit order deterministic across runs and filesystems rather than
    depending on directory-traversal order -- via `subprocess.run` with
    the pattern and scope passed as separate argv entries (no `shell=True`,
    no shell metacharacter interpretation). At most `args.max_results`
    hits are returned, in that deterministic file order; each hit's line
    text is capped at `_MAX_LINE_CHARS` characters.

    Raises:
        SearchError: `args.scope` resolves outside `repo_root` (whether
            by `..` traversal or by following a symlink that points
            outside it); `rg` cannot be started at all (e.g. it is not
            on `PATH`); or `rg` exits with a status other than 0 (matches
            found) or 1 (no matches, not an error) -- most commonly
            because `args.pattern` is not a regex `rg` accepts, in which
            case the message is `rg`'s own diagnostic.

    `rg` exiting 1 is not an error: it returns a framed message stating
    no matches were found, exactly like a search that matched nothing for
    any other reason.
    """
    cmd = ["rg", "--line-number", "--no-heading", "--sort", "path", args.pattern]
    if args.scope is not None:
        _resolve_within_repo_root(args.scope, repo_root=repo_root)
        cmd.append(args.scope)

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise SearchError(f"rg could not be run ({exc})") from exc

    if completed.returncode not in (0, _RG_EXIT_NO_MATCHES):
        message = completed.stderr.strip() or (
            f"rg exited with status {completed.returncode}"
        )
        raise SearchError(message)

    if completed.returncode == _RG_EXIT_NO_MATCHES:
        body = f"No matches for pattern {args.pattern!r}."
    else:
        hits = _parse_hits(completed.stdout)[: args.max_results]
        body = "\n".join(_render_hit(hit) for hit in hits)

    return frame_untrusted(body, source="search_result", origin=args.pattern)


def _parse_max_results(value: Any) -> int:
    """Validate the optional `max_results` field, defaulting to
    `_DEFAULT_MAX_RESULTS` when absent and raising `SearchError` when
    present but not an integer between `_MIN_MAX_RESULTS` and
    `_MAX_MAX_RESULTS` inclusive. `bool` is rejected even though it is a
    subclass of `int` in Python -- `true`/`false` in the source JSON is
    never a valid result cap."""
    if value is None:
        return _DEFAULT_MAX_RESULTS
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (_MIN_MAX_RESULTS <= value <= _MAX_MAX_RESULTS)
    ):
        raise SearchError(
            f"arguments: 'max_results' must be an integer between "
            f"{_MIN_MAX_RESULTS} and {_MAX_MAX_RESULTS}"
        )
    return value


def parse_search_args(arguments_json: str) -> SearchArgs:
    """Parse and validate one `ToolCallEvent.arguments_json` payload for
    `search` against `SEARCH_SCHEMA`.

    Raises:
        SearchError: `arguments_json` is not valid JSON, is not a JSON
            object, is missing the required `pattern` field, carries a
            field `SEARCH_SCHEMA` does not declare, or gives `pattern`,
            `scope`, or `max_results` a value of the wrong type or
            range -- every case names the offending field, never a raw
            `json.JSONDecodeError` or `KeyError`.
    """
    try:
        raw: Any = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise SearchError(f"arguments: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise SearchError("arguments: expected a JSON object")

    unexpected = sorted(set(raw) - _ALLOWED_ARG_FIELDS)
    if unexpected:
        raise SearchError(f"arguments: unexpected field(s) {unexpected}")

    if "pattern" not in raw:
        raise SearchError("arguments: missing required field 'pattern'")

    pattern = raw["pattern"]
    if not isinstance(pattern, str):
        raise SearchError("arguments: 'pattern' must be a string")

    scope = raw.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise SearchError("arguments: 'scope' must be a string")

    return SearchArgs(
        pattern=pattern,
        scope=scope,
        max_results=_parse_max_results(raw.get("max_results")),
    )
