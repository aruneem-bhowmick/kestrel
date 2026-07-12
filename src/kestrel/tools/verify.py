"""Runs a target repo's own configured lint/build/test commands and
reports pass/fail back to a model, so a task's exit criterion is a real
command's exit code rather than the model's own claim of success.

`verify` reads whichever of `lint`/`build`/`test` the target repo's
`KESTREL.md` configures (see `kestrel.kestrel_md`) and runs each one, in
that fixed order, through the exact same `bwrap` sandbox
`kestrel.tools.execute.execute` uses -- scoped to the repo, network
namespace unshared. Every command runs to completion regardless of an
earlier one's outcome: a failing lint check does not prevent build or
test from also reporting their own result, since a caller fixing several
problems at once needs to see all of them together, not one at a time.

KESTREL.md is reloaded fresh on every call rather than cached, because a
prior turn's `edit_file` call may have just changed it and `verify`'s job
is to check the repo's *current* state, not a stale snapshot -- the
opposite trade-off from anything that snapshots KESTREL.md once per task
for prompt-cache stability.

Because these commands are configuration a repo's own maintainers wrote,
not a shell command a model improvised, they carry the same trust level
KESTREL.md itself carries and never pass through `execute`'s
destructive-action approval gate; gating an automatic verification step
behind an interactive prompt on every call would defeat its purpose as
an unattended exit-criterion check.

Beyond returning a short pass/fail summary to the model, every call
renders and persists a `VerificationReport` as markdown under
`.kestrel/artifacts/` and, when a caller supplies one, appends it to a
mutable `report_sink` list -- a plain Python list, not a type this
module defines, so a caller can inspect what ran without this module
needing to know anything about whatever consumes that list.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from kestrel.kestrel_md import KestrelMd, load_kestrel_md
from kestrel.provider.base import ToolSchema
from kestrel.security.framing import frame_untrusted
from kestrel.tools.execute import _cap_stream
from kestrel.tools.sandbox import run_sandboxed

_VERIFY_ORDER: Final[tuple[Literal["lint", "build", "test"], ...]] = (
    "lint",
    "build",
    "test",
)
_DEFAULT_VERIFY_TIMEOUT_S: Final[float] = 300.0

_ALLOWED_ARG_FIELDS: Final[frozenset[str]] = frozenset({"only"})

_ARTIFACTS_DIRNAME: Final[str] = ".kestrel/artifacts"

VERIFY_SCHEMA = ToolSchema(
    name="verify",
    description=(
        "Run the repo-configured lint/build/test commands from "
        "KESTREL.md and report pass/fail for each."
    ),
    parameters={
        "type": "object",
        "properties": {
            "only": {
                "type": "array",
                "items": {"type": "string", "enum": list(_VERIFY_ORDER)},
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class VerifyArgs:
    """One validated `verify` call's arguments.

    Attributes:
        only: Restricts the run to these command names, in whatever
            order they're supplied here -- they still execute in
            `_VERIFY_ORDER`, this only narrows which are included.
            `None` (the default) means every command KESTREL.md
            configures.
    """

    only: tuple[Literal["lint", "build", "test"], ...] | None = None


class VerifyError(Exception):
    """No KESTREL.md at `repo_root`, or it configures none of the
    requested commands (or none at all, when `only` is omitted).
    `str(self)` names the remedy -- the model should not call `verify`
    again for the same repo without a KESTREL.md change.
    """


@dataclass(frozen=True, slots=True)
class VerificationCommandResult:
    """One configured command's own outcome.

    Attributes:
        name: Which of lint/build/test this result is for.
        command: The exact shell command string KESTREL.md configured
            for `name`.
        exit_code: The command's exit status; `-1` when `timed_out` is
            `True`.
        timed_out: Whether the command was killed for exceeding its
            timeout bound rather than exiting on its own.
        stdout: Everything the command wrote to standard output.
        stderr: Everything the command wrote to standard error.
    """

    name: Literal["lint", "build", "test"]
    command: str
    exit_code: int
    timed_out: bool
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """One `verify()` call's outcome across every command it ran.

    Attributes:
        task_id: The agent-loop task this verification ran under.
        turn_id: The loop turn this verification ran during.
        commands: Every command actually run, in `_VERIFY_ORDER`.
        passed: True iff `commands` is non-empty and every result's
            `exit_code == 0` and none `timed_out`.
    """

    task_id: str
    turn_id: int
    commands: tuple[VerificationCommandResult, ...]
    passed: bool


def run_verification(
    kestrel_md: KestrelMd,
    *,
    only: Sequence[str] | None,
    repo_root: Path,
    task_id: str,
    turn_id: int,
    timeout_s: float = _DEFAULT_VERIFY_TIMEOUT_S,
) -> VerificationReport:
    """Run each of lint/build/test present in `kestrel_md.verify_commands`
    (filtered to `only` when given), in `_VERIFY_ORDER`, each via
    `run_sandboxed` (repo_root + tmp scoped, network off -- identical
    sandboxing to `execute`, no exemption for verification commands).
    Returns a VerificationReport regardless of pass/fail: a failing
    lint/test run is a normal, reportable outcome, not an exception.

    Raises:
        VerifyError: `kestrel_md.verify_commands` has no entry for any
            name in `only` (or none configured at all, when `only` is
            None) -- raised before running anything.
    """
    configured = kestrel_md.verify_commands.as_mapping()

    if only is None:
        selected_names = [name for name in _VERIFY_ORDER if name in configured]
        if not selected_names:
            raise VerifyError(
                "KESTREL.md configures no lint/build/test commands to "
                "verify -- add a kestrel-verify block before calling verify"
            )
    else:
        missing = [name for name in only if name not in configured]
        if missing:
            raise VerifyError(
                f"KESTREL.md does not configure a {missing[0]!r} command -- "
                "add it to the kestrel-verify block before calling verify "
                "with only including it"
            )
        selected_names = [name for name in _VERIFY_ORDER if name in only]

    results: list[VerificationCommandResult] = []
    for name in selected_names:
        command = configured[name]
        outcome = run_sandboxed(
            ["sh", "-c", command],
            repo_root=repo_root,
            timeout_s=timeout_s,
            allow_network=False,
        )
        results.append(
            VerificationCommandResult(
                name=name,
                command=command,
                exit_code=outcome.exit_code,
                timed_out=outcome.timed_out,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
            )
        )

    passed = bool(results) and all(
        result.exit_code == 0 and not result.timed_out for result in results
    )
    return VerificationReport(
        task_id=task_id, turn_id=turn_id, commands=tuple(results), passed=passed
    )


def render_verification_markdown(report: VerificationReport) -> str:
    """Render `report` as a markdown document: a top-level pass/fail
    heading, then one section per command naming its exit code, timeout
    flag, and capped stdout/stderr (reuses `execute`'s own
    `_MAX_STREAM_BYTES` cap and truncation-note convention)."""
    status = "PASSED" if report.passed else "FAILED"
    lines: list[str] = [f"# Verification: {status}", ""]

    for result in report.commands:
        lines.append(f"## {result.name}: `{result.command}`")
        lines.append(f"- exit_code: {result.exit_code}")
        lines.append(f"- timed_out: {result.timed_out}")
        lines.append("")
        lines.append("stdout:")
        lines.append("```")
        lines.append(_cap_stream(result.stdout))
        lines.append("```")
        lines.append("")
        lines.append("stderr:")
        lines.append("```")
        lines.append(_cap_stream(result.stderr))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def _allocate_report_path(artifacts_dir: Path, *, task_id: str, turn_id: int) -> Path:
    """Return an unused markdown path under `artifacts_dir` for this
    task/turn's report: `verification-{task_id}-{turn_id}.md` when that
    name is free, otherwise the same name with a numeric suffix
    appended. Calling `verify` more than once within the same task and
    turn -- e.g. a model narrowing its next call with `only` after an
    earlier one failed -- gets one report file per call instead of the
    later call silently overwriting the earlier one."""
    base = f"verification-{task_id}-{turn_id}"
    candidate = artifacts_dir / f"{base}.md"
    suffix = 1
    while candidate.exists():
        candidate = artifacts_dir / f"{base}-{suffix}.md"
        suffix += 1
    return candidate


def persist_verification_report(report: VerificationReport, *, repo_root: Path) -> Path:
    """Write `render_verification_markdown(report)` to a fresh path under
    `repo_root / ".kestrel" / "artifacts"`, named
    `verification-{report.task_id}-{report.turn_id}.md` (or that name
    with a numeric suffix, when an earlier call already claimed it),
    creating parent directories as needed; returns the written path."""
    artifacts_dir = repo_root / _ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = _allocate_report_path(
        artifacts_dir, task_id=report.task_id, turn_id=report.turn_id
    )
    path.write_text(render_verification_markdown(report), encoding="utf-8")
    return path


def _render_summary(
    report: VerificationReport, *, persisted_path: Path, repo_root: Path
) -> str:
    """Render the short pass/fail summary `verify` returns to the model:
    each command's own name and outcome, plus the persisted report's
    repo-relative path -- never the commands' own stdout/stderr, which
    lives only in the persisted markdown and each result's own fields."""
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"verify: {status}"]
    for result in report.commands:
        outcome = "timed out" if result.timed_out else f"exit_code={result.exit_code}"
        lines.append(f"- {result.name}: {outcome}")
    relative_path = persisted_path.relative_to(repo_root).as_posix()
    lines.append(f"Full report: {relative_path}")
    return "\n".join(lines)


def verify(
    args: VerifyArgs,
    *,
    repo_root: Path,
    task_id: str,
    turn_id: int,
    report_sink: list[VerificationReport] | None = None,
) -> str:
    """Load KESTREL.md fresh from `repo_root` on every call (verify's
    own view of configuration must never be stale relative to a prior
    turn's `edit_file` call that may have changed it -- deliberately
    the opposite trade-off from a stable-prefix snapshot frozen at task
    start for cache stability; verify's job is correctness, not
    cacheability). Raises `VerifyError` (framed by `dispatch`, not here,
    matching every other tool's contract) when no KESTREL.md exists or
    it configures nothing `only` asks for. Otherwise runs
    `run_verification`, persists the report via
    `persist_verification_report`, appends it to `report_sink` when
    given (harmless no-op when `None`), and returns a rendered
    pass/fail summary (not the full per-command output -- that lives
    only in the persisted markdown and each result's own fields) framed
    via `frame_untrusted(..., source="tool_stdout", origin="verify")`.
    """
    kestrel_md = load_kestrel_md(repo_root)
    if kestrel_md is None:
        raise VerifyError(
            f"no KESTREL.md at {repo_root} -- add one with a kestrel-verify "
            "block naming lint/build/test commands before calling verify"
        )

    report = run_verification(
        kestrel_md,
        only=args.only,
        repo_root=repo_root,
        task_id=task_id,
        turn_id=turn_id,
    )

    persisted_path = persist_verification_report(report, repo_root=repo_root)
    if report_sink is not None:
        report_sink.append(report)

    summary = _render_summary(
        report, persisted_path=persisted_path, repo_root=repo_root
    )
    return frame_untrusted(summary, source="tool_stdout", origin="verify")


def _parse_only(value: Any) -> tuple[Literal["lint", "build", "test"], ...] | None:
    """Validate the optional `only` field, defaulting to `None` (every
    configured command) when absent and raising `VerifyError` when
    present but not an array of strings each drawn from `_VERIFY_ORDER`."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise VerifyError("arguments: 'only' must be an array of strings")
    unknown = sorted(set(value) - set(_VERIFY_ORDER))
    if unknown:
        raise VerifyError(f"arguments: 'only' names unknown command(s) {unknown}")
    return cast(tuple[Literal["lint", "build", "test"], ...], tuple(value))


def parse_verify_args(arguments_json: str) -> VerifyArgs:
    """Parse and validate one `ToolCallEvent.arguments_json` payload for
    `verify` against `VERIFY_SCHEMA`.

    Raises:
        VerifyError: `arguments_json` is not valid JSON, is not a JSON
            object, carries a field `VERIFY_SCHEMA` does not declare, or
            gives `only` a value that isn't an array of strings drawn
            from `_VERIFY_ORDER` -- every case names the offending
            field, never a raw `json.JSONDecodeError`.
    """
    try:
        raw: Any = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise VerifyError(f"arguments: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise VerifyError("arguments: expected a JSON object")

    unexpected = sorted(set(raw) - _ALLOWED_ARG_FIELDS)
    if unexpected:
        raise VerifyError(f"arguments: unexpected field(s) {unexpected}")

    return VerifyArgs(only=_parse_only(raw.get("only")))
