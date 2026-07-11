"""Runs a shell command in a sandboxed subprocess for a model to call
as a tool.

`execute` is the first tool in this package that lets a model cause
side effects beyond reading -- it hands a caller-supplied argv list to
`kestrel.tools.sandbox.run_sandboxed`, which confines it to a `bwrap`
sandbox scoped to the repo plus a scratch directory, with its network
namespace unshared. The command never sees a shell: `cmd` is an argv
list end to end, so no shell metacharacter in it is ever interpreted.
The command's stdout, stderr, exit code, and whether it timed out are
rendered as one block and returned already wrapped by
`kestrel.security.framing.frame_untrusted`, so a model can run a
command without ever treating its own output as instructions.

This tool never requests network access on its own -- every call goes
through the sandbox with `allow_network=False`; there is no argument
here that can turn it on. Escalating to network access is a separate,
approval-mediated decision with no call path from this function.

Before a command ever reaches the sandbox, `classify_destructive_action`
checks it against a small, fixed pattern table -- `rm`/`rmdir`, `chmod`,
and a force-flagged `git push` -- and a match is handed to an injected
`ApprovalManager` for a real approval decision; a command outside that
table runs unchecked. This is the spec's named-category gate, not a
general shell-command allowlist system.

Like `read_file` and `search`, this module owns its own schema,
argument dataclass, and JSON-argument parsing, and raises `ExecuteError`
-- never a raw exception -- for any argument this tool refuses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kestrel.managers.approval import ApprovalManager, ApprovalRequest
from kestrel.provider.base import ToolSchema
from kestrel.security.framing import frame_untrusted
from kestrel.tools.sandbox import SandboxResult, run_sandboxed

_ALLOWED_ARG_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "timeout_s"})

_DEFAULT_TIMEOUT_S: Final[float] = 60.0
_MIN_TIMEOUT_S: Final[float] = 1.0
_MAX_TIMEOUT_S: Final[float] = 300.0

EXECUTE_SCHEMA = ToolSchema(
    name="execute",
    description="Run a shell command in a sandboxed subprocess scoped to the repo.",
    parameters={
        "type": "object",
        "properties": {
            "cmd": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "timeout_s": {"type": "number", "minimum": 1, "maximum": 300},
        },
        "required": ["cmd"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class ExecuteArgs:
    """One validated `execute` call's arguments.

    Attributes:
        cmd: The command to run, as a non-empty argv list -- never a
            shell string.
        timeout_s: How long the command may run before being killed,
            in seconds; between 1 and 300 inclusive.
    """

    cmd: tuple[str, ...]
    timeout_s: float = _DEFAULT_TIMEOUT_S


class ExecuteError(Exception):
    """Raised for an `execute` request this tool refuses.

    `str(self)` is itself the message returned to the model -- every
    raise site names the offending argument rather than letting a
    lower-level exception (`json.JSONDecodeError`, `TypeError`) escape
    uninterpreted. A sandboxed command's own failure (a non-zero exit,
    a timeout) is never one of these -- it is a normal, framed result.
    """


_MAX_STREAM_BYTES: Final[int] = 64 * 1024


def _cap_stream(content: str) -> str:
    """Cap a stream's content at _MAX_STREAM_BYTES and append a truncation note if it exceeds it."""
    encoded = content.encode("utf-8")
    if len(encoded) <= _MAX_STREAM_BYTES:
        return content
    truncated_str = encoded[:_MAX_STREAM_BYTES].decode("utf-8", errors="ignore")
    omitted_bytes = len(encoded) - len(truncated_str.encode("utf-8"))
    return f"{truncated_str}\n... [truncated: {omitted_bytes} more bytes omitted]"


def _render_result(result: SandboxResult) -> str:
    """Render `result`'s exit code, timeout flag, stdout, and stderr as
    the single text block `execute` frames as its tool result."""
    stdout_capped = _cap_stream(result.stdout)
    stderr_capped = _cap_stream(result.stderr)
    return (
        f"exit_code: {result.exit_code}\n"
        f"timed_out: {result.timed_out}\n"
        f"stdout:\n{stdout_capped}\n"
        f"stderr:\n{stderr_capped}"
    )


_DELETE_COMMANDS: Final[frozenset[str]] = frozenset({"rm", "rmdir"})
_FORCE_PUSH_FLAGS: Final[frozenset[str]] = frozenset(
    {"--force", "-f", "--force-with-lease"}
)


def classify_destructive_action(cmd: Sequence[str]) -> ApprovalRequest | None:
    """Classify `cmd` against the fixed pattern table `execute` gates on.

    Recognizes exactly three shapes: `rm` or `rmdir` as `"delete"`,
    `chmod` as `"chmod"`, and a `git push` carrying `--force` or `-f`
    anywhere among its remaining arguments as `"force_push"`. Every
    other command -- including an empty `cmd` or a bare `git push` with
    no force flag -- returns `None`, meaning `execute` runs it
    unchecked; this table covers only the spec's named categories, not
    a general shell-command allowlist system.
    """
    if not cmd:
        return None
    head = cmd[0]
    joined = " ".join(cmd)
    if head in _DELETE_COMMANDS:
        return ApprovalRequest(
            kind="delete", summary=f"Delete: {joined}", detail=joined
        )
    if head == "chmod":
        return ApprovalRequest(
            kind="chmod", summary=f"Change permissions: {joined}", detail=joined
        )
    if (
        head == "git"
        and len(cmd) > 1
        and cmd[1] == "push"
        and any(
            flag in _FORCE_PUSH_FLAGS or flag.startswith("--force-with-lease=")
            for flag in cmd[2:]
        )
    ):
        return ApprovalRequest(
            kind="force_push", summary=f"Force-push: {joined}", detail=joined
        )
    return None


def execute(args: ExecuteArgs, *, repo_root: Path, approval: ApprovalManager) -> str:
    """Run `args.cmd` under `repo_root`'s sandbox and return its
    outcome framed as untrusted tool output.

    Before anything runs, `args.cmd` is classified via
    `classify_destructive_action`; a match is handed to
    `approval.check` for a real approval decision, and a command
    outside the pattern table runs unchecked. Always calls
    `run_sandboxed` with `allow_network=False` -- this function has no
    way to request network access for the command it runs; that is a
    separate, approval-mediated escalation with no call path from here.

    Raises:
        ApprovalDenied: `args.cmd` classified as a destructive action
            that `approval` denied. Propagated unchanged -- this
            function does not catch it on a caller's behalf, the same
            contract `ApprovalManager.check` documents.
        SandboxUnavailableError: `bwrap` is not on `PATH` -- propagated
            from `run_sandboxed` unchanged, since it names a real
            infrastructure precondition this tool cannot satisfy on its
            own, not a malformed request.

    A non-zero exit code or a timed-out command is not an error here:
    both render into the returned frame like any other outcome, so the
    model can see and react to them.
    """
    request = classify_destructive_action(args.cmd)
    if request is not None:
        approval.check(request)

    result = run_sandboxed(
        list(args.cmd),
        repo_root=repo_root,
        timeout_s=args.timeout_s,
        allow_network=False,
    )
    rendered = _render_result(result)
    return frame_untrusted(rendered, source="tool_stdout", origin=" ".join(args.cmd))


def _parse_cmd(value: Any) -> tuple[str, ...]:
    """Validate the required `cmd` field: a non-empty JSON array whose
    every element is a string, raising `ExecuteError` naming the defect
    otherwise. Rejecting anything else here is what keeps `cmd` an argv
    list all the way down to `run_sandboxed` -- never a shell string."""
    if not isinstance(value, list) or len(value) == 0:
        raise ExecuteError("arguments: 'cmd' must be a non-empty array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ExecuteError("arguments: 'cmd' must be a non-empty array of strings")
    return tuple(value)


def _parse_timeout(value: Any) -> float:
    """Validate the optional `timeout_s` field, defaulting to
    `_DEFAULT_TIMEOUT_S` when absent and raising `ExecuteError` when
    present but not a number between `_MIN_TIMEOUT_S` and
    `_MAX_TIMEOUT_S` inclusive. `bool` is rejected even though it is a
    subclass of `int` in Python -- `true`/`false` in the source JSON is
    never a valid timeout."""
    if value is None:
        return _DEFAULT_TIMEOUT_S
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecuteError("arguments: 'timeout_s' must be a number")
    if not (_MIN_TIMEOUT_S <= value <= _MAX_TIMEOUT_S):
        raise ExecuteError(
            f"arguments: 'timeout_s' must be between {_MIN_TIMEOUT_S:.0f} "
            f"and {_MAX_TIMEOUT_S:.0f}"
        )
    return float(value)


def parse_execute_args(arguments_json: str) -> ExecuteArgs:
    """Parse and validate one `ToolCallEvent.arguments_json` payload for
    `execute` against `EXECUTE_SCHEMA`.

    Raises:
        ExecuteError: `arguments_json` is not valid JSON, is not a JSON
            object, is missing the required `cmd` field, carries a
            field `EXECUTE_SCHEMA` does not declare, or gives `cmd` or
            `timeout_s` a value of the wrong type or range -- every
            case names the offending field, never a raw
            `json.JSONDecodeError` or `KeyError`.
    """
    try:
        raw: Any = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ExecuteError(f"arguments: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise ExecuteError("arguments: expected a JSON object")

    unexpected = sorted(set(raw) - _ALLOWED_ARG_FIELDS)
    if unexpected:
        raise ExecuteError(f"arguments: unexpected field(s) {unexpected}")

    if "cmd" not in raw:
        raise ExecuteError("arguments: missing required field 'cmd'")

    return ExecuteArgs(
        cmd=_parse_cmd(raw["cmd"]),
        timeout_s=_parse_timeout(raw.get("timeout_s")),
    )
