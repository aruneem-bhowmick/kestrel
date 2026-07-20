"""System test: a fresh task must never leave a prior task's own
`VerificationReport` lingering in the artifact pane once it starts,
even when the fresh task never calls `verify` itself -- and once that
fresh task ends, the pane shows its own `Walkthrough` rather than
either the stale report or the placeholder text a fresh app starts with.

Seeds the artifact pane directly via `ArtifactPane.show_report` rather
than driving a real `verify` call through the `bwrap` sandbox for a
first task -- this suite only proves `KestrelApp._run_task` resets the
pane at task start, a concern independent of `verify` itself, so it
needs no sandbox and carries no `bwrap` skip guard, unlike
`test_p040_artifact_pane_live.py`. Reuses
`test_p038_tui_conversation_stream.py`'s own mock-server-plus-fixture-
repo pattern; neither `read_file` (the only tool call this suite's own
scripted task makes) touches the sandbox, matching
`test_p039_tool_log_diff_live.py`'s own precedent for a bwrap-free
suite.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import sanitize_terminal
from kestrel.tools.verify import VerificationCommandResult, VerificationReport
from kestrel.tui.app import ArtifactPane, KestrelApp

pytestmark = [pytest.mark.p040, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_FILE_MARKER = "hello from the fixture module"


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


def _write_fixture_repo(tmp_path: Path) -> None:
    """Write the one file the scripted `read_file` call reads."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(f"# {_FILE_MARKER}\n", encoding="utf-8")


def _stale_report() -> VerificationReport:
    """A report standing in for an earlier task's own real
    `VerificationReport`, seeded directly onto the pane so this suite
    never needs a real `verify` call to prove the reset."""
    return VerificationReport(
        task_id="earlier-task",
        turn_id=1,
        commands=(
            VerificationCommandResult(
                name="test",
                command="pytest -q",
                exit_code=0,
                timed_out=False,
                stdout="",
                stderr="",
            ),
        ),
        passed=True,
    )


async def test_a_fresh_task_resets_the_artifact_pane_even_without_its_own_verify_call(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the artifact pane already shows an earlier task's own
    `VerificationReport`, when a fresh task that never calls `verify`
    is submitted through `#task_input` and runs to completion, then the
    pane no longer shows that stale report -- it shows this task's own
    `Walkthrough` instead, reading `_no verification ran_`, never
    reverting to the placeholder text `on_mount` shows before any task
    has ever run, since a completed FAST-mode task now always leaves a
    real artifact behind."""
    _write_fixture_repo(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_READ_FILE, _DONE_CASSETTE],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )

    async with app.run_test() as pilot:
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        artifact_pane.show_report(_stale_report())
        assert "# Verification: PASSED" in artifact_pane.source

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "read src/greet.py"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert "# Verification: PASSED" not in artifact_pane.source
        assert artifact_pane.source != "_no artifact yet_"

        walkthrough_persisted = list(
            (tmp_path / ".kestrel" / "artifacts").glob("walkthrough-*.md")
        )
        assert len(walkthrough_persisted) == 1
        walkthrough_text = walkthrough_persisted[0].read_text(encoding="utf-8")
        assert artifact_pane.source == sanitize_terminal(walkthrough_text)
        assert "_no verification ran_" in artifact_pane.source
