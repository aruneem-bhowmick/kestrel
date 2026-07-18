"""System test: the agent loop's self-critique phase driven end to end
against a real `LiteLLMClient`, a real mock chat-completions server, and
`kestrel.agent.critique.make_self_critique_fn`'s own routed, non-streamed
completion -- proving a critique call actually reaches the wire and
targets its own routed model, distinct from the main task's.

Extends `test_p022_loop_scripted_task.py`'s own mock-server-plus-fixture-
repo pattern, substituting a second `read_file` turn for the `execute`
turn that suite uses, so this one never depends on `bwrap` being on
`PATH`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.critique import make_self_critique_fn
from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.session import SessionManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p047, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"
_CRITIQUE_REJECT = _CASSETTES / "critique_reject.sse"

_MAIN_MODEL_ID = "glm-5.2"
_MAIN_PROVIDER_MODEL = "z-ai/glm-5.2"
_CRITIQUE_MODEL_ID = "glm-5.2-cheap"
_CRITIQUE_PROVIDER_MODEL = "z-ai/glm-5.2-cheap"
_FILE_MARKER = "hello from the fixture module"


def _registry() -> Registry:
    """A two-entry `Registry`: the main task's own model, and a
    `"cheap"`-tagged entry self-critique routes to instead -- distinct
    `provider_model` values so a captured request body can tell the two
    calls apart."""
    main_entry = ModelEntry(
        id=_MAIN_MODEL_ID,
        backend="openrouter",
        provider_model=_MAIN_PROVIDER_MODEL,
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    critique_entry = ModelEntry(
        id=_CRITIQUE_MODEL_ID,
        backend="openrouter",
        provider_model=_CRITIQUE_PROVIDER_MODEL,
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.10"),
        usd_per_mtok_output=Decimal("0.20"),
        usd_per_mtok_cached=Decimal("0.02"),
        supports_tools=True,
        supports_cache=True,
        tags=frozenset({"cheap"}),
    )
    return Registry(
        models={_MAIN_MODEL_ID: main_entry, _CRITIQUE_MODEL_ID: critique_entry},
        source=None,
    )


def _write_fixture_repo(tmp_path: Path) -> None:
    """One small Python module the scripted `read_file` calls read."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")


def _build_deps(
    registry: Registry,
    client: LiteLLMClient,
    tmp_path: Path,
    *,
    meter: CostMeter | None = None,
    session: SessionManager | None = None,
) -> LoopDeps:
    """A `LoopDeps` bundle wired with a real, routed self-critique
    function bound to the registry's own `"cheap"`-tagged entry --
    mirroring what `kestrel.task_setup.build_task_deps` wires by
    default, but constructed directly the way this suite's sibling
    scripted-task tests build `LoopDeps`."""
    return LoopDeps(
        client=client,
        registry=registry,
        model_id=_MAIN_MODEL_ID,
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=meter if meter is not None else CostMeter(),
        session=session,
        self_critique_fn=make_self_critique_fn(
            client=client, model_id=_CRITIQUE_MODEL_ID
        ),
    )


async def test_a_critique_call_interleaves_with_every_real_turn(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a two-turn task (one `read_file` tool call, then a plain
    no-more-tools reply) with self-critique routed to a `"cheap"`-tagged
    entry, and a mock server scripted to approve both proposals, when
    the task runs, then: (1) the server receives exactly four requests,
    in the order [main-turn-1, critique-1, main-turn-2, critique-2]; (2)
    each critique request's own captured body names the critique
    model's own `provider_model`, distinct from the main turns'; (3) the
    task completes normally, proving an approved critique never blocks
    progress.
    """
    _write_fixture_repo(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    deps = _build_deps(registry, client, tmp_path)

    result = await run_task("read src/greet.py, then stop", deps, task_id="sys-p047-1")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert len(captured) == 4

    request_models = [json.loads(body)["model"] for body in captured]
    assert request_models == [
        _MAIN_PROVIDER_MODEL,
        _CRITIQUE_PROVIDER_MODEL,
        _MAIN_PROVIDER_MODEL,
        _CRITIQUE_PROVIDER_MODEL,
    ]


async def test_a_rejected_critique_skips_the_turn_and_tries_again(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a task whose first proposal is declined by self-critique,
    when the task runs, then that turn's own proposal is skipped (a
    synthetic explanation folds into history in its place, and the real
    `read_file` tool is never dispatched for it) and the loop tries
    again -- the existing self-critique-skip path, unchanged, now
    driven by a real routed call instead of a test-supplied lambda.
    """
    _write_fixture_repo(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_REJECT,
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    deps = _build_deps(registry, client, tmp_path)

    result = await run_task("read src/greet.py, then stop", deps, task_id="sys-p047-2")

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 3
    assert len(captured) == 6

    skipped_tool_messages = [
        message
        for message in result.history
        if message["role"] == "tool"
        and "not approved by self-critique" in message["content"]
    ]
    assert len(skipped_tool_messages) == 1

    read_tool_results = [
        message
        for message in result.history
        if message["role"] == "tool" and _FILE_MARKER in message["content"]
    ]
    assert len(read_tool_results) == 1


@pytest.mark.cost_regression
async def test_critique_calls_never_leak_into_the_tasks_own_priced_total_or_journal(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a two-turn task with self-critique approving both
    proposals, when it runs, then `deps.meter` and `deps.session` record
    exactly the two real turns' own scripted usage -- the critique
    cassettes' own token counts (distinct from both main turns') never
    appear in either, proving a real, separately-priced critique call
    cannot silently inflate or corrupt the task's own accounted spend.
    """
    _write_fixture_repo(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    meter = CostMeter()
    session = SessionManager(repo_root=tmp_path, task_id="sys-p047-3")
    deps = _build_deps(registry, client, tmp_path, meter=meter, session=session)

    result = await run_task("read src/greet.py, then stop", deps, task_id="sys-p047-3")

    assert result.reason == TerminationReason.TASK_COMPLETE
    recorded_usage = [(turn.input_tokens, turn.output_tokens) for turn in meter.turns]
    assert recorded_usage == [(55, 12), (70, 5)]
    assert meter.session_usd > Decimal("0")

    assert len(session.records) == 2
    journaled_usage = [
        (record.turn_cost.input_tokens, record.turn_cost.output_tokens)
        for record in session.records
    ]
    assert journaled_usage == [(55, 12), (70, 5)]
