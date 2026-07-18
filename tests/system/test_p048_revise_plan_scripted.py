"""System test: a PLAN-mode task's plan, revised end to end against a
real `LiteLLMClient` and a real mock chat-completions server -- proving
`extract_plan_from_result`/`revise_plan` work against genuine journaled
session state, not merely a hand-built `LoopResult` the way
`tests/unit/test_p048_plan.py` exercises them.

Reuses the same hermetic harness (`run_task`/`resume_task` driven
directly against `LoopDeps`, no CLI subprocess involved) `test_p022_loop
_scripted_task.py` and `test_p045_resume_inject_message.py` already
established. Since a PLAN-mode reply is plain text with no tool calls,
this suite needs no `bwrap` sandbox and is not skipped on a non-Linux
host.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, TerminationReason, run_task
from kestrel.agent.plan import PlanComment, extract_plan_from_result, revise_plan
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.session import SessionManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p048, pytest.mark.system]

_TASK_ID = "sys-p048-revise-1"

_INITIAL_PLAN_TEXT = (
    "1. Read the existing config loader.\n"
    "2. Add a new field for the retry timeout.\n"
    "3. Write a regression test for the new field."
)
_REVISED_PLAN_TEXT = (
    "1. Read the existing config loader and its legacy fallback path.\n"
    "2. Add a new field for the retry timeout, defaulting to 30 seconds.\n"
    "3. Write a regression test for the new field and the legacy fallback."
)


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the cassette's
    own `model` field."""
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
    return Registry(models={"glm-5.2": entry}, source=None)


def _write_plan_reply_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls -- standing in for a
    PLAN-mode model turn, whose reply is the plan itself. Built with
    `json.dumps` so `text`'s own embedded newlines always serialize into
    a single well-formed SSE data line."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000009,
            "model": "z-ai/glm-5.2",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000009,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000009,
            "model": "z-ai/glm-5.2",
            "choices": [],
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 25,
                "total_tokens": 115,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    lines: list[str] = []
    for chunk in chunks:
        lines.append("data: " + json.dumps(chunk))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def test_revise_plan_folds_comments_in_and_the_reply_reparses(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fresh PLAN-mode session already carrying one completed
    plan turn, when it is revised with one comment against a scripted
    cassette replying with a revised plan, then the returned
    `LoopResult.history`'s last message is that revised reply verbatim,
    and `extract_plan_from_result` on it parses to the revised lines.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    registry = _registry()

    initial_cassette = _write_plan_reply_cassette(
        tmp_path / "initial-plan.sse", text=_INITIAL_PLAN_TEXT
    )
    initial_base_url = mock_openai_server(cassette_sequence=[initial_cassette])
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", initial_base_url)

    session = SessionManager(repo_root=tmp_path, task_id=_TASK_ID)
    first_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        session=session,
    )

    first_result = await run_task(
        "draft an implementation plan for the retry-timeout feature",
        first_deps,
        task_id=_TASK_ID,
    )
    assert first_result.reason == TerminationReason.TASK_COMPLETE

    initial_plan = extract_plan_from_result(first_result, task_id=_TASK_ID)
    assert [line.text for line in initial_plan.lines] == _INITIAL_PLAN_TEXT.splitlines()

    revised_cassette = _write_plan_reply_cassette(
        tmp_path / "revised-plan.sse", text=_REVISED_PLAN_TEXT
    )
    revised_base_url = mock_openai_server(cassette_sequence=[revised_cassette])
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", revised_base_url)

    second_deps = LoopDeps(
        client=LiteLLMClient(registry),
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        session=SessionManager(repo_root=tmp_path, task_id=_TASK_ID),
    )
    comments = [
        PlanComment(
            line_index=initial_plan.lines[0].index,
            line_text=initial_plan.lines[0].text,
            comment="Also handle the legacy config fallback path.",
        )
    ]

    revised_result = await revise_plan(_TASK_ID, second_deps, comments)

    assert revised_result.reason == TerminationReason.TASK_COMPLETE
    assert revised_result.history[-1] == {
        "role": "assistant",
        "content": _REVISED_PLAN_TEXT,
    }

    revised_plan = extract_plan_from_result(revised_result, task_id=_TASK_ID)
    assert [line.text for line in revised_plan.lines] == _REVISED_PLAN_TEXT.splitlines()
