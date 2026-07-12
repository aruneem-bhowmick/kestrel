"""System test: the agent loop's verification gate driven end to end
against a real `LiteLLMClient`, a real mock chat-completions server, and
a real `verify` tool call running through the actual `bwrap` sandbox --
proving the gate wired in `run_task` actually withholds and then grants
`TASK_COMPLETE` around a genuine (not scripted) `VerificationReport`,
not just a faked one standing in for it the way
`test_p026_verification_gate.py`'s unit suite does.

Skipped locally when `bwrap` is not on `PATH`, exactly like
`tests/system/test_p022_loop_scripted_task.py`: the binary genuinely may
not be installed, and never will be on a non-Linux host. CI installs
`bubblewrap` on every runner, so this suite always actually runs there.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from kestrel.agent.loop import LoopDeps, LoopLimits, TerminationReason, run_task
from kestrel.cost.meter import CostMeter
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.sandbox import bwrap_available

pytestmark = [
    pytest.mark.p026,
    pytest.mark.system,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_VERIFY = _CASSETTES / "toolcall_verify.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassettes' own `model` field."""
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


def _write_kestrel_md(repo_root: Path, *, test_command: str) -> None:
    """Configure a single, trivially-passing `test` command for
    `verify` to run against `repo_root`."""
    (repo_root / "KESTREL.md").write_text(
        f'```kestrel-verify\ntest = "{test_command}"\n```\n', encoding="utf-8"
    )


async def test_verification_required_task_completes_only_after_a_real_passing_verify(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given `require_verification=True`, a fixture repo whose KESTREL.md
    configures a trivially-passing `test` command, and a mock server
    scripted to reply with a `verify` tool call followed by a plain
    no-more-tools reply, when the task runs, then the loop dispatches a
    real `verify` call through the actual sandbox, records a genuine
    passing `VerificationReport`, and only then lets the following
    no-tool-calls turn end the task TASK_COMPLETE -- after exactly two
    turns.
    """
    _write_kestrel_md(tmp_path, test_command="true")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_VERIFY, _DONE_CASSETTE],
        capture=captured,
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    registry = _registry()
    client = LiteLLMClient(registry)
    deps = LoopDeps(
        client=client,
        registry=registry,
        model_id="glm-5.2",
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
        require_verification=True,
        limits=LoopLimits(max_turns=10),
    )

    result = await run_task(
        "make sure the repo's own tests pass", deps, task_id="sys-p026-1"
    )

    assert result.reason == TerminationReason.TASK_COMPLETE
    assert result.turns_used == 2
    assert len(captured) == 2

    assert len(deps.verification_reports) == 1
    assert deps.verification_reports[0].passed is True

    artifacts_dir = tmp_path / ".kestrel" / "artifacts"
    assert any(artifacts_dir.glob("verification-sys-p026-1-*.md"))
