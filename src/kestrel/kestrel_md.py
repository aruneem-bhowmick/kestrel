"""Loads KESTREL.md, a target repo's own optional project-memory file.

`KESTREL.md`, when present at a target repo's root, is free-form
markdown written by that repo's own maintainers -- conventions, house
style, anything they want an agent working in the repo to already know
-- in the same per-repo-memory tradition as `CLAUDE.md`. Unlike
`kestrel.toml`/`models.toml`, it has exactly one location and no search
path: a repo either has one at its root or it has none, and having none
is a perfectly normal outcome, not an error.

Because a repo's own maintainers write this file, its content is
trusted project memory, not data a tool fetched at runtime -- it is
never passed through `kestrel.security.framing.frame_untrusted`, unlike
every byte a `read_file`/`search`/`execute` call returns.

A `KESTREL.md` may optionally carry one specially-marked fenced code
block naming repo-configured lint/build/test commands, read as a small
TOML table:

    ```kestrel-verify
    lint = "ruff check ."
    build = "true"
    test = "pytest -q"
    ```

Any of the three keys may be omitted, and the block itself is entirely
optional -- a file with none simply configures no commands. This module
only parses that table; nothing here runs the commands it names.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_KESTREL_MD_FILENAME: Final[str] = "KESTREL.md"
_VERIFY_BLOCK_INFO_STRING: Final[str] = "kestrel-verify"
_ALLOWED_VERIFY_KEYS: Final[frozenset[str]] = frozenset({"lint", "build", "test"})

# A line-anchored match for a fenced code block (``` or ~~~, mirroring
# either fence style), capturing its info string and body -- deliberately
# not a full Markdown parser, since the only thing this loader needs is
# whichever block (if any) is tagged `kestrel-verify`.
_FENCE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<fence>```|~~~)[ \t]*(?P<info>\S*)[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"^(?P=fence)[ \t]*\r?\n?",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class VerifyCommands:
    """The repo-configured verification commands, each optional.

    Attributes:
        lint: Shell command string to run for a lint check, or None.
        build: Shell command string to run for a build check, or None.
        test: Shell command string to run for a test check, or None.
    """

    lint: str | None = None
    build: str | None = None
    test: str | None = None

    def as_mapping(self) -> Mapping[str, str]:
        """Every configured command, in the fixed order lint, build,
        test, keyed by its own name -- omits any name left `None`."""
        ordered = (("lint", self.lint), ("build", self.build), ("test", self.test))
        return {name: command for name, command in ordered if command is not None}


@dataclass(frozen=True, slots=True)
class KestrelMd:
    """One target repo's loaded project-memory file.

    Attributes:
        raw_text: The file's exact decoded content, unmodified -- meant
            to be folded verbatim into a model prompt rather than
            reformatted, summarized, or truncated.
        verify_commands: Parsed lint/build/test commands, empty when the
            file carries no ```kestrel-verify block.
    """

    raw_text: str
    verify_commands: VerifyCommands


class KestrelMdError(Exception):
    """A KESTREL.md file exists but cannot be loaded: its bytes are not
    valid UTF-8, or its ```kestrel-verify block does not parse as TOML,
    or that block names a key outside {lint, build, test} or gives one
    of those keys a non-string value. `str(self)` names the defect and
    the file path -- never a raw UnicodeDecodeError or TOMLDecodeError
    escaping to a caller.
    """


def _find_verify_block(raw_text: str) -> str | None:
    """Return the body of the first fenced code block in `raw_text`
    whose info string is exactly `kestrel-verify`, or `None` when no
    such block exists. Blocks are scanned in the order they appear, so
    when more than one carries that info string, only the first one
    found is ever returned."""
    for match in _FENCE_BLOCK_RE.finditer(raw_text):
        if match.group("info") == _VERIFY_BLOCK_INFO_STRING:
            return match.group("body")
    return None


def _parse_verify_block(body: str, *, path: Path) -> VerifyCommands:
    """Parse a ```kestrel-verify block's raw body as TOML into
    `VerifyCommands`.

    Raises:
        KestrelMdError: `body` is not valid TOML, names a key outside
            {lint, build, test}, or gives one of those keys a value that
            isn't a string.
    """
    try:
        data = tomllib.loads(body)
    except tomllib.TOMLDecodeError as exc:
        raise KestrelMdError(
            f"{path}: kestrel-verify block is not valid TOML ({exc})"
        ) from exc

    unexpected = sorted(set(data) - _ALLOWED_VERIFY_KEYS)
    if unexpected:
        raise KestrelMdError(
            f"{path}: kestrel-verify block names disallowed key "
            f"{unexpected[0]!r} (allowed: lint, build, test)"
        )

    for key, value in data.items():
        if not isinstance(value, str):
            raise KestrelMdError(
                f"{path}: kestrel-verify key {key!r} must be a string"
            )

    return VerifyCommands(
        lint=data.get("lint"),
        build=data.get("build"),
        test=data.get("test"),
    )


def load_kestrel_md(repo_root: Path) -> KestrelMd | None:
    """Read `repo_root / "KESTREL.md"`.

    Returns `None` when the file does not exist -- KESTREL.md is
    optional; a repo with none simply has no project memory to inject
    and no configured verify commands, never an error.

    When present: decode as UTF-8 (raising `KestrelMdError` naming the
    path on failure); extract the first fenced code block (``` or ~~~,
    a line-anchored regex match, not a full Markdown parser) whose info
    string is exactly `kestrel-verify`; parse that block's body as TOML
    into `VerifyCommands`, raising `KestrelMdError` naming any key
    outside {lint, build, test} or any value that isn't a string. A
    file with no ```kestrel-verify block at all is not an error --
    `verify_commands` is simply empty (`VerifyCommands()`). A second or
    later ```kestrel-verify block in the same file is ignored.

    Raises:
        KestrelMdError: the file's bytes are not valid UTF-8, or its
            ```kestrel-verify block fails to parse (see
            `_parse_verify_block`).
    """
    path = repo_root / _KESTREL_MD_FILENAME
    if not path.is_file():
        return None

    try:
        raw_text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KestrelMdError(f"{path}: not valid UTF-8 text ({exc})") from exc

    body = _find_verify_block(raw_text)
    verify_commands = (
        _parse_verify_block(body, path=path) if body is not None else VerifyCommands()
    )

    return KestrelMd(raw_text=raw_text, verify_commands=verify_commands)
