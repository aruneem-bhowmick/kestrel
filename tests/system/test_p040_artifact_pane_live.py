"""System test: a scripted task whose own `verify` tool call runs a
real, trivially-passing command through the actual `bwrap` sandbox
drives the artifact pane's live rendering -- proving `TuiLoopObserver.
on_verification`/`ArtifactPane.show_report` against the real cockpit and
a genuine `VerificationReport`, not a stand-in one.

Reuses `test_p038_tui_conversation_stream.py`'s own mock-server-plus-
fixture-repo pattern and `test_p026_verification_gate_scripted.py`'s own
KESTREL.md convention (a `test` command that trivially passes). Skipped
locally when `bwrap` is not on `PATH`, exactly like both of those
suites; CI installs `bubblewrap` on every runner, so this suite always
actually runs there.
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
from kestrel.tools.sandbox import bwrap_available, run_sandboxed
from kestrel.tui.app import ArtifactPane, KestrelApp


def _can_initialize_network_namespace() -> bool:
    """Check whether this environment can actually run a sandboxed
    command at all -- the same prerequisite `test_p026_verification_gate_
    scripted.py` checks before trusting `bwrap_available()` alone."""
    if not bwrap_available():
        return False
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_sandboxed(["true"], repo_root=Path(tmpdir), timeout_s=5.0)
            return result.exit_code == 0 and not result.timed_out
    except Exception:
        return False


pytestmark = [
    pytest.mark.p040,
    pytest.mark.system,
    pytest.mark.ui,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
    pytest.mark.skipif(
        not _can_initialize_network_namespace(),
        reason="bwrap cannot initialize network namespace (missing capabilities or AppArmor restrictions)",
    ),
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


async def test_artifact_pane_renders_the_real_verification_report(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo whose KESTREL.md configures a trivially-
    passing `test` command, and a mock server scripted to reply with a
    `verify` tool call followed by a plain no-more-tools reply, when a
    task is submitted through `#task_input`, then the artifact pane's
    rendered content matches `sanitize_terminal` of the exact markdown
    the `verify` tool actually persisted to `.kestrel/artifacts/`."""
    _write_kestrel_md(tmp_path, test_command="true")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[_TOOLCALL_VERIFY, _DONE_CASSETTE],
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
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = "make sure the repo's own tests pass"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        artifacts_dir = tmp_path / ".kestrel" / "artifacts"
        persisted = list(artifacts_dir.glob("verification-*.md"))
        assert len(persisted) == 1
        persisted_text = persisted[0].read_text(encoding="utf-8")

        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(persisted_text)
        assert "# Verification: PASSED" in artifact_pane.source
