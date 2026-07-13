"""Journal a task's turns to disk and reconstruct them after a crash or
a budget halt.

`SessionManager` is `kestrel.managers.UndoManager`'s sibling: the same
append-only-JSONL-under-`.kestrel/`, one-record-per-line discipline,
including the "write bytes with an explicit `\\n`, never text mode" and
"a malformed trailing line is tolerated, not fatal" rules, applied here
to a task's own conversation turns instead of its file mutations. Each
line is one `TurnRecord` -- the messages one turn added to history, that
turn's own priced `TurnCost`, and whatever `VerificationReport` the task
has most recently produced -- so a task interrupted mid-run can be
reconstructed via `load_session` and continued via
`kestrel.agent.loop.resume_task` from exactly where it left off, rather
than losing every turn since the process's last clean exit.

`aggregate_historical_spend` reads every session file under one repo's
`.kestrel/sessions/` directory and sums the cost of every turn whose
timestamp falls in a given window -- the historical-spend figure a
day/month budget cap needs, computed fresh on every call since it is a
rarely-invoked, task-boundary operation rather than a hot path worth
caching.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from kestrel.cost.meter import TurnCost
from kestrel.provider.base import Message
from kestrel.provider.events import ToolCallEvent
from kestrel.tools.verify import VerificationCommandResult, VerificationReport

_SESSIONS_DIRNAME = "sessions"


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One journaled turn (or compaction event) of a task.

    Attributes:
        turn_id: The agent-loop turn this record is associated with.
            NOT a unique key across the journal -- a compaction record
            and the real turn immediately following it legitimately
            share a `turn_id`.
        task_id: The task this record belongs to.
        timestamp: `time.time()`-style UTC epoch seconds when this
            record was created -- the basis for day/month spend
            aggregation.
        message_deltas: Every `Message` appended to history during this
            record's own turn, in order -- for a compaction event, this
            is the whole, just-folded history instead, since folding
            replaces history rather than appending to it.
        turn_cost: This record's own priced `TurnCost` -- every record
            is priced, including a compaction call's own summarization
            cost.
        verification: The `VerificationReport` this record's own turn
            produced, or `None` when it produced none.
    """

    turn_id: int
    task_id: str
    timestamp: float
    message_deltas: tuple[Message, ...]
    turn_cost: TurnCost
    verification: VerificationReport | None
    active_model_id: str | None = None
    degraded: bool = False


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Serialize one `Message` to a JSON-safe dict, converting any
    `ToolCallEvent`s under `tool_calls` to plain dicts of their own three
    fields; every other optional field is copied through as-is and
    omitted entirely when absent from `message`."""
    data: dict[str, Any] = {"role": message["role"], "content": message["content"]}
    if "tool_calls" in message:
        data["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments_json": call.arguments_json}
            for call in message["tool_calls"]
        ]
    if "tool_call_id" in message:
        data["tool_call_id"] = message["tool_call_id"]
    if "cache_breakpoint" in message:
        data["cache_breakpoint"] = message["cache_breakpoint"]
    return data


def _message_from_dict(data: Mapping[str, Any]) -> Message:
    """Parse one JSON dict back into a `Message`, reconstructing any
    `ToolCallEvent`s under `tool_calls`."""
    message: Message = {"role": data["role"], "content": data["content"]}
    if "tool_calls" in data:
        message["tool_calls"] = [
            ToolCallEvent(
                id=call["id"], name=call["name"], arguments_json=call["arguments_json"]
            )
            for call in data["tool_calls"]
        ]
    if "tool_call_id" in data:
        message["tool_call_id"] = data["tool_call_id"]
    if "cache_breakpoint" in data:
        message["cache_breakpoint"] = data["cache_breakpoint"]
    return message


def _turn_cost_to_dict(turn_cost: TurnCost) -> dict[str, Any]:
    """Serialize a `TurnCost` to a JSON-safe dict, rendering its
    `Decimal` field as a string -- `json.dumps` has no native support for
    `Decimal` and a float would silently reintroduce the rounding error
    the pricing module exists to avoid."""
    return {
        "model_id": turn_cost.model_id,
        "input_tokens": turn_cost.input_tokens,
        "output_tokens": turn_cost.output_tokens,
        "cached_tokens": turn_cost.cached_tokens,
        "usd": str(turn_cost.usd),
    }


def _turn_cost_from_dict(data: Mapping[str, Any]) -> TurnCost:
    """Parse one JSON dict back into a `TurnCost`, reconstructing its
    `usd` field as a `Decimal` from the string it was serialized as."""
    return TurnCost(
        model_id=data["model_id"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        cached_tokens=data["cached_tokens"],
        usd=Decimal(data["usd"]),
    )


def _verification_to_dict(report: VerificationReport) -> dict[str, Any]:
    """Serialize a `VerificationReport` to a JSON-safe dict."""
    return {
        "task_id": report.task_id,
        "turn_id": report.turn_id,
        "commands": [
            {
                "name": command.name,
                "command": command.command,
                "exit_code": command.exit_code,
                "timed_out": command.timed_out,
                "stdout": command.stdout,
                "stderr": command.stderr,
            }
            for command in report.commands
        ],
        "passed": report.passed,
    }


def _verification_from_dict(data: Mapping[str, Any]) -> VerificationReport:
    """Parse one JSON dict back into a `VerificationReport`."""
    return VerificationReport(
        task_id=data["task_id"],
        turn_id=data["turn_id"],
        commands=tuple(
            VerificationCommandResult(
                name=command["name"],
                command=command["command"],
                exit_code=command["exit_code"],
                timed_out=command["timed_out"],
                stdout=command["stdout"],
                stderr=command["stderr"],
            )
            for command in data["commands"]
        ),
        passed=data["passed"],
    )


def _record_to_json(record: TurnRecord) -> str:
    """Serialize `record` to a single JSONL line, with no trailing
    newline -- the field order (`turn_id`, `task_id`, `timestamp`,
    `message_deltas`, `turn_cost`, `verification`) is the journal's
    stable, tested wire format."""
    return json.dumps(
        {
            "turn_id": record.turn_id,
            "task_id": record.task_id,
            "timestamp": record.timestamp,
            "message_deltas": [_message_to_dict(m) for m in record.message_deltas],
            "turn_cost": _turn_cost_to_dict(record.turn_cost),
            "verification": (
                _verification_to_dict(record.verification)
                if record.verification is not None
                else None
            ),
            "active_model_id": record.active_model_id,
            "degraded": record.degraded,
        },
        ensure_ascii=False,
    )


def _record_from_json(line: str) -> TurnRecord:
    """Parse one JSONL line back into a `TurnRecord`."""
    data: dict[str, Any] = json.loads(line)
    verification = data["verification"]
    return TurnRecord(
        turn_id=data["turn_id"],
        task_id=data["task_id"],
        timestamp=data["timestamp"],
        message_deltas=tuple(_message_from_dict(m) for m in data["message_deltas"]),
        turn_cost=_turn_cost_from_dict(data["turn_cost"]),
        verification=(
            _verification_from_dict(verification) if verification is not None else None
        ),
        active_model_id=data.get("active_model_id"),
        degraded=data.get("degraded", False),
    )


def _read_journal(journal_path: Path) -> list[TurnRecord]:
    """Read every well-formed `TurnRecord` line from `journal_path`, in
    order; returns an empty list when the file does not exist.

    A malformed trailing line -- a session file truncated mid-write by a
    crash -- is tolerated and dropped rather than raised, mirroring
    `UndoManager`'s own recovery rule; a malformed line anywhere else in
    the file is a genuine corruption and re-raises.
    """
    if not journal_path.exists():
        return []
    raw_lines = journal_path.read_bytes().splitlines()
    lines = [line for line in raw_lines if line]
    records: list[TurnRecord] = []
    for i, line in enumerate(lines):
        try:
            records.append(_record_from_json(line.decode("utf-8")))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            if i == len(lines) - 1:
                break
            raise exc
    return records


class SessionManager:
    """Journals a task's turns under a repo and reads them back."""

    def __init__(
        self, *, repo_root: Path, task_id: str, journal_path: Path | None = None
    ) -> None:
        """Point this manager at `repo_root`/`task_id`, loading whatever
        records already exist at `journal_path` (default: `repo_root /
        ".kestrel" / "sessions" / f"{task_id}.jsonl"`) into memory.
        Neither the journal's parent directory nor the file itself is
        created here -- that happens lazily, on the first call to
        `record_turn`, mirroring `UndoManager.__init__`'s own
        lazy-creation contract."""
        self._repo_root = repo_root
        self._task_id = task_id
        self._journal_path = (
            journal_path
            if journal_path is not None
            else repo_root / ".kestrel" / _SESSIONS_DIRNAME / f"{task_id}.jsonl"
        )
        self._records: list[TurnRecord] = _read_journal(self._journal_path)

    @property
    def journal_path(self) -> Path:
        """The journal file this manager reads from and appends to."""
        return self._journal_path

    @property
    def records(self) -> tuple[TurnRecord, ...]:
        """Every record journaled so far, in append order."""
        return tuple(self._records)

    def record_turn(self, record: TurnRecord) -> None:
        """Append `record` to the journal (disk + in-memory), exactly
        like `UndoManager.record`'s own append discipline: binary mode,
        explicit `\\n`, byte-stable line endings across platforms."""
        if record.task_id != self._task_id:
            raise ValueError(
                f"record task_id {record.task_id!r} does not match journal task_id {self._task_id!r}"
            )
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{_record_to_json(record)}\n".encode("utf-8")
        with self._journal_path.open("ab") as handle:
            handle.write(line)
        self._records.append(record)


@dataclass(frozen=True, slots=True)
class SessionState:
    """A task's session, reconstructed from its journal.

    Attributes:
        history: Every `message_deltas` entry across every record,
            concatenated in record order.
        turns: Every record's own `turn_cost`, in order -- re-seeds a
            fresh `CostMeter` via its `initial_turns` parameter.
        last_verification: The most recent non-`None` verification
            across every record, or `None` if none exists.
        turns_used: The highest `turn_id` seen across every record -- a
            resumed task's turn-cap counter continues from here, not
            from zero.
    """

    history: tuple[Message, ...]
    turns: tuple[TurnCost, ...]
    last_verification: VerificationReport | None
    turns_used: int
    active_model_id: str | None
    degraded: bool


def load_session(repo_root: Path, task_id: str) -> SessionState:
    """Read every `TurnRecord` on disk for `task_id`, in order, and fold
    them into a `SessionState`.

    Raises:
        FileNotFoundError: no journal exists for `task_id` under
            `repo_root` -- there is nothing to resume.
    """
    journal_path = repo_root / ".kestrel" / _SESSIONS_DIRNAME / f"{task_id}.jsonl"
    if not journal_path.exists():
        raise FileNotFoundError(
            f"no session journal for task {task_id!r} at {journal_path}"
        )
    records = _read_journal(journal_path)

    history: list[Message] = []
    turns: list[TurnCost] = []
    last_verification: VerificationReport | None = None
    turns_used = 0
    active_model_id: str | None = None
    degraded = False
    for record in records:
        history.extend(record.message_deltas)
        turns.append(record.turn_cost)
        if record.verification is not None:
            last_verification = record.verification
        turns_used = max(turns_used, record.turn_id)
        active_model_id = record.active_model_id
        degraded = record.degraded

    return SessionState(
        history=tuple(history),
        turns=tuple(turns),
        last_verification=last_verification,
        turns_used=turns_used,
        active_model_id=active_model_id,
        degraded=degraded,
    )


def aggregate_historical_spend(
    repo_root: Path,
    *,
    now: float,
    window_s: float,
    exclude_task_id: str | None = None,
) -> Decimal:
    """Sum `turn_cost.usd` across every `TurnRecord` in every
    `repo_root/.kestrel/sessions/*.jsonl` file whose `timestamp` falls in
    `[now - window_s, now]`, excluding `exclude_task_id`'s own file
    entirely when given -- required when resuming a task, so that task's
    own spend is not double-counted once its resumed `CostMeter` re-adds
    it via `session_usd`. Read fresh every call (no caching): this is a
    rarely-called, task-boundary operation, not a hot path.
    """
    sessions_dir = repo_root / ".kestrel" / _SESSIONS_DIRNAME
    total = Decimal("0")
    if not sessions_dir.exists():
        return total

    window_start = now - window_s
    for journal_path in sorted(sessions_dir.glob("*.jsonl")):
        if exclude_task_id is not None and journal_path.stem == exclude_task_id:
            continue
        for record in _read_journal(journal_path):
            if window_start <= record.timestamp <= now:
                total += record.turn_cost.usd
    return total
