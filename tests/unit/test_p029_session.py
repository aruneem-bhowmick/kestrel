"""Tests for `SessionManager`: recording and replaying a task's own
turns as JSONL, `load_session`'s reconstruction of history/turns/
verification/turn-count from a journal, `aggregate_historical_spend`'s
cross-file spend rollup, `CostMeter`'s additive `initial_turns` seeding,
and the pinned wire format -- mirroring `test_p017_undo.py`'s own
coverage of the sibling undo journal.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.cost.meter import CostMeter, TurnCost
from kestrel.managers.session import (
    SessionManager,
    TurnRecord,
    aggregate_historical_spend,
    load_session,
)
from kestrel.provider.base import Message
from kestrel.provider.events import ToolCallEvent, UsageEvent
from kestrel.registry.model import ModelEntry
from kestrel.tools.verify import VerificationCommandResult, VerificationReport

pytestmark = [pytest.mark.p029, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p029_turn_record.golden"
)


def _turn_cost(
    usd: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached_tokens: int = 0,
    model_id: str = "glm-5.2",
) -> TurnCost:
    """A `TurnCost` built directly from a fixed `usd` figure, standing in
    for one already priced by `compute_turn_cost` -- this suite is about
    the journal's own round trip, not the pricing formula."""
    return TurnCost(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        usd=Decimal(usd),
    )


def _assistant_message(text: str) -> Message:
    """A minimal assistant-role `Message` carrying no tool calls."""
    return {"role": "assistant", "content": text}


def _tool_message(content: str, *, tool_call_id: str) -> Message:
    """A minimal tool-role `Message` result."""
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def _report(*, task_id: str, turn_id: int, passed: bool) -> VerificationReport:
    """A one-command `VerificationReport` for the given outcome."""
    return VerificationReport(
        task_id=task_id,
        turn_id=turn_id,
        commands=(
            VerificationCommandResult(
                name="test",
                command="pytest -q",
                exit_code=0 if passed else 1,
                timed_out=False,
                stdout="",
                stderr="",
            ),
        ),
        passed=passed,
    )


def _record(
    *,
    turn_id: int,
    task_id: str,
    timestamp: float,
    message_deltas: tuple[Message, ...] = (),
    turn_cost: TurnCost | None = None,
    verification: VerificationReport | None = None,
) -> TurnRecord:
    """A `TurnRecord` with sensible defaults for fields a given test
    doesn't care about."""
    return TurnRecord(
        turn_id=turn_id,
        task_id=task_id,
        timestamp=timestamp,
        message_deltas=message_deltas,
        turn_cost=turn_cost if turn_cost is not None else _turn_cost("0.000100"),
        verification=verification,
    )


@pytest.mark.sanity
def test_record_turn_then_records_round_trips_in_order(tmp_path: Path) -> None:
    """Given two turns recorded in order, when `records` is read back,
    then both come back in the same order, unchanged."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-1")
    record_a = _record(
        turn_id=1,
        task_id="t-1",
        timestamp=1000.0,
        message_deltas=(_assistant_message("a"),),
    )
    record_b = _record(
        turn_id=2,
        task_id="t-1",
        timestamp=1001.0,
        message_deltas=(_assistant_message("b"),),
    )

    manager.record_turn(record_a)
    manager.record_turn(record_b)

    assert manager.records == (record_a, record_b)


@pytest.mark.sanity
def test_load_session_reconstructs_history_as_concatenation_of_message_deltas(
    tmp_path: Path,
) -> None:
    """Given two journaled records with their own message deltas, when
    `load_session` is called, then `history` is the exact concatenation
    of both deltas, in record order."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-2")
    manager.record_turn(
        _record(
            turn_id=1,
            task_id="t-2",
            timestamp=1.0,
            message_deltas=(_assistant_message("first"),),
        )
    )
    manager.record_turn(
        _record(
            turn_id=2,
            task_id="t-2",
            timestamp=2.0,
            message_deltas=(
                _assistant_message("second"),
                _tool_message("result", tool_call_id="call-1"),
            ),
        )
    )

    state = load_session(tmp_path, "t-2")

    assert state.history == (
        _assistant_message("first"),
        _assistant_message("second"),
        _tool_message("result", tool_call_id="call-1"),
    )


@pytest.mark.sanity
def test_turns_matches_every_records_turn_cost_and_reseeds_identically(
    tmp_path: Path,
) -> None:
    """Given three journaled records, when `load_session` is called,
    then `turns` matches every record's own `turn_cost` in order, and
    re-seeding a fresh `CostMeter` from it totals the same as summing
    the original costs directly."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-3")
    costs = [_turn_cost("0.000100"), _turn_cost("0.000250"), _turn_cost("0.000075")]
    for i, cost in enumerate(costs, start=1):
        manager.record_turn(
            _record(turn_id=i, task_id="t-3", timestamp=float(i), turn_cost=cost)
        )

    state = load_session(tmp_path, "t-3")

    assert state.turns == tuple(costs)
    reseeded = CostMeter(initial_turns=state.turns)
    assert reseeded.session_usd == sum((c.usd for c in costs), start=Decimal(0))


def test_last_verification_is_most_recent_and_a_later_none_does_not_clear_it(
    tmp_path: Path,
) -> None:
    """Given a first record carrying a verification report and a second,
    later record carrying none, when `load_session` is called, then
    `last_verification` is still the first record's report -- the
    later `None` does not clear it."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-4")
    first_report = _report(task_id="t-4", turn_id=1, passed=False)
    manager.record_turn(
        _record(turn_id=1, task_id="t-4", timestamp=1.0, verification=first_report)
    )
    manager.record_turn(_record(turn_id=2, task_id="t-4", timestamp=2.0))

    state = load_session(tmp_path, "t-4")

    assert state.last_verification == first_report


def test_last_verification_updates_when_a_later_record_carries_a_new_one(
    tmp_path: Path,
) -> None:
    """Given two records each carrying their own verification report,
    when `load_session` is called, then `last_verification` is the
    later record's report, not the earlier one."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-4b")
    manager.record_turn(
        _record(
            turn_id=1,
            task_id="t-4b",
            timestamp=1.0,
            verification=_report(task_id="t-4b", turn_id=1, passed=False),
        )
    )
    second_report = _report(task_id="t-4b", turn_id=2, passed=True)
    manager.record_turn(
        _record(turn_id=2, task_id="t-4b", timestamp=2.0, verification=second_report)
    )

    state = load_session(tmp_path, "t-4b")

    assert state.last_verification == second_report


def test_turns_used_equals_the_highest_turn_id_seen(tmp_path: Path) -> None:
    """Given records whose turn ids are not contiguous, when
    `load_session` is called, then `turns_used` is the highest one seen,
    not the count of records."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-5")
    for turn_id in (1, 2, 5):
        manager.record_turn(
            _record(turn_id=turn_id, task_id="t-5", timestamp=float(turn_id))
        )

    state = load_session(tmp_path, "t-5")

    assert state.turns_used == 5


@pytest.mark.sanity
def test_load_session_on_nonexistent_task_id_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Given no journal at all for a task id, when `load_session` is
    called, then `FileNotFoundError` is raised -- there is nothing to
    resume."""
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path, "no-such-task")


def test_aggregate_historical_spend_sums_only_records_within_the_window(
    tmp_path: Path,
) -> None:
    """Given two session files under one repo, one with a record inside
    the window and one outside it, and another file entirely inside the
    window, when `aggregate_historical_spend` is called, then only the
    in-window records across every file are summed."""
    now = 10_000.0
    manager_a = SessionManager(repo_root=tmp_path, task_id="task-a")
    manager_a.record_turn(
        _record(
            turn_id=1,
            task_id="task-a",
            timestamp=now - 50,
            turn_cost=_turn_cost("0.001000"),
        )
    )
    manager_a.record_turn(
        _record(
            turn_id=2,
            task_id="task-a",
            timestamp=now - 500,
            turn_cost=_turn_cost("0.002000"),
        )
    )
    manager_b = SessionManager(repo_root=tmp_path, task_id="task-b")
    manager_b.record_turn(
        _record(
            turn_id=1,
            task_id="task-b",
            timestamp=now - 10,
            turn_cost=_turn_cost("0.000500"),
        )
    )

    total = aggregate_historical_spend(tmp_path, now=now, window_s=100.0)

    assert total == Decimal("0.001500")


def test_aggregate_historical_spend_exclude_task_id_omits_its_own_file_entirely(
    tmp_path: Path,
) -> None:
    """Given two session files, both with in-window records, when
    `aggregate_historical_spend` is called with `exclude_task_id` naming
    one of them, then that file's spend is left out entirely, not just
    de-duplicated."""
    now = 10_000.0
    manager_a = SessionManager(repo_root=tmp_path, task_id="task-a")
    manager_a.record_turn(
        _record(
            turn_id=1,
            task_id="task-a",
            timestamp=now - 5,
            turn_cost=_turn_cost("0.001000"),
        )
    )
    manager_b = SessionManager(repo_root=tmp_path, task_id="task-b")
    manager_b.record_turn(
        _record(
            turn_id=1,
            task_id="task-b",
            timestamp=now - 5,
            turn_cost=_turn_cost("0.000500"),
        )
    )

    total = aggregate_historical_spend(
        tmp_path, now=now, window_s=100.0, exclude_task_id="task-a"
    )

    assert total == Decimal("0.000500")


def test_aggregate_historical_spend_returns_zero_when_no_sessions_dir_exists(
    tmp_path: Path,
) -> None:
    """Given a repo with no `.kestrel/sessions/` directory at all, when
    `aggregate_historical_spend` is called, then it returns zero rather
    than raising."""
    assert aggregate_historical_spend(tmp_path, now=1.0, window_s=10.0) == Decimal("0")


def test_malformed_trailing_line_is_tolerated_and_dropped(tmp_path: Path) -> None:
    """Given a journal with one valid record followed by a truncated,
    malformed trailing line (as a crash mid-write would leave), when
    loaded, then the valid record is kept and the malformed tail is
    silently dropped -- mirroring `UndoManager`'s own recovery rule."""
    manager = SessionManager(repo_root=tmp_path, task_id="t-9")
    manager.record_turn(_record(turn_id=1, task_id="t-9", timestamp=1.0))

    with manager.journal_path.open("ab") as handle:
        handle.write(
            b'{"turn_id": 2, "task_id": "t-9", "timestamp": 2.0, "message_deltas": ['
        )

    state = load_session(tmp_path, "t-9")

    assert state.turns_used == 1
    assert len(state.turns) == 1


def test_malformed_middle_line_raises_instead_of_being_silently_dropped(
    tmp_path: Path,
) -> None:
    """Given a malformed line in the middle of an otherwise well-formed
    journal, when read, then the malformed line's own error propagates
    rather than being tolerated -- only a *trailing* line gets that
    leniency."""
    journal = tmp_path / ".kestrel" / "sessions" / "t-10.jsonl"
    journal.parent.mkdir(parents=True)
    valid = SessionManager(repo_root=tmp_path, task_id="t-10", journal_path=journal)
    valid.record_turn(_record(turn_id=1, task_id="t-10", timestamp=1.0))
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"turn_id": 2, "broken\n')
        handle.write(
            '{"turn_id": 3, "task_id": "t-10", "timestamp": 3.0, '
            '"message_deltas": [], "turn_cost": '
            '{"model_id": "glm-5.2", "input_tokens": 1, "output_tokens": 1, '
            '"cached_tokens": 0, "usd": "0.000001"}, "verification": null}\n'
        )

    with pytest.raises((ValueError, KeyError, TypeError)):
        load_session(tmp_path, "t-10")


def test_session_manager_starts_empty_when_no_journal_exists_yet(
    tmp_path: Path,
) -> None:
    """Given a fresh repo with no session journal, when a manager is
    constructed, then `records` is empty and nothing is created on disk."""
    manager = SessionManager(repo_root=tmp_path, task_id="fresh")

    assert manager.records == ()
    assert not manager.journal_path.exists()
    assert not manager.journal_path.parent.exists()


def test_record_turn_creates_the_journal_directory_and_file_lazily(
    tmp_path: Path,
) -> None:
    """Given a fresh repo, when the first turn is recorded, then the
    `.kestrel/sessions/` directory and the task's own journal file are
    created at that point, not before."""
    manager = SessionManager(repo_root=tmp_path, task_id="lazy")
    assert not manager.journal_path.exists()

    manager.record_turn(_record(turn_id=1, task_id="lazy", timestamp=1.0))

    assert manager.journal_path == tmp_path / ".kestrel" / "sessions" / "lazy.jsonl"
    assert manager.journal_path.exists()


def test_explicit_journal_path_overrides_the_default(tmp_path: Path) -> None:
    """Given an explicit `journal_path`, when a manager is constructed
    and records a turn, then it writes there instead of the default
    `.kestrel/sessions/<task_id>.jsonl` location."""
    custom = tmp_path / "custom" / "journal.jsonl"
    manager = SessionManager(repo_root=tmp_path, task_id="t-11", journal_path=custom)

    manager.record_turn(_record(turn_id=1, task_id="t-11", timestamp=1.0))

    assert custom.exists()
    assert not (tmp_path / ".kestrel").exists()


def test_journal_persists_across_manager_instances(tmp_path: Path) -> None:
    """Given turns recorded through one `SessionManager` instance, when a
    second instance is constructed pointed at the same journal, then it
    sees those same records -- the journal, not the in-process list, is
    the source of truth."""
    first = SessionManager(repo_root=tmp_path, task_id="t-12")
    first.record_turn(_record(turn_id=1, task_id="t-12", timestamp=1.0))

    second = SessionManager(
        repo_root=tmp_path, task_id="t-12", journal_path=first.journal_path
    )

    assert second.records == first.records


@pytest.mark.sanity
def test_cost_meter_initial_turns_matches_recording_one_at_a_time() -> None:
    """Given a sequence of usage events recorded one at a time into a
    fresh `CostMeter`, when a second `CostMeter` is constructed with
    `initial_turns` set to the first's own `turns`, then both meters'
    `turns` and `session_usd` match exactly."""
    entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    usage_events = [
        UsageEvent(input_tokens=1000, output_tokens=50, cached_tokens=0),
        UsageEvent(input_tokens=500, output_tokens=20, cached_tokens=100),
    ]

    incremental_meter = CostMeter()
    for usage in usage_events:
        incremental_meter.record(usage, entry)

    reseeded_meter = CostMeter(initial_turns=incremental_meter.turns)

    assert reseeded_meter.turns == incremental_meter.turns
    assert reseeded_meter.session_usd == incremental_meter.session_usd


@pytest.mark.regression
def test_turn_record_wire_format_matches_golden_snapshot(tmp_path: Path) -> None:
    """One canonical `TurnRecord`, exercising every field including a
    tool call, a verification report, and a nonzero cache count,
    recorded and read back as raw bytes, matches a pinned snapshot
    byte-for-byte -- the journal's line format is a durable contract
    once anything else starts reading it (a future `kestrel run
    --resume`, a `kestrel cost` historical rollup), not an
    implementation detail free to drift."""
    manager = SessionManager(repo_root=tmp_path, task_id="task-42")

    manager.record_turn(
        TurnRecord(
            turn_id=7,
            task_id="task-42",
            timestamp=1700000000.5,
            message_deltas=(
                {
                    "role": "assistant",
                    "content": "reading the file now",
                    "tool_calls": [
                        ToolCallEvent(
                            id="call-1",
                            name="read_file",
                            arguments_json='{"path": "a.py"}',
                        )
                    ],
                },
                {"role": "tool", "content": "A_CONTENT", "tool_call_id": "call-1"},
            ),
            turn_cost=TurnCost(
                model_id="glm-5.2",
                input_tokens=1000,
                output_tokens=50,
                cached_tokens=100,
                usd=Decimal("0.000710"),
            ),
            verification=VerificationReport(
                task_id="task-42",
                turn_id=7,
                commands=(
                    VerificationCommandResult(
                        name="test",
                        command="pytest -q",
                        exit_code=0,
                        timed_out=False,
                        stdout="ok",
                        stderr="",
                    ),
                ),
                passed=True,
            ),
        )
    )

    assert manager.journal_path.read_bytes() == _GOLDEN_FILE.read_bytes()
