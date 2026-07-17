"""Red-team system test: a hostile terminal escape sequence echoed to a
configured verify command's own stdout -- the exact
`ansi_escape_laden_payload` injection-corpus case -- must never survive
into the artifact pane's own rendered markdown, even though the
persisted report on disk (read by `verify` itself, matched against
`tests/integration/test_p025_verify_sandbox.py`'s own red-team case)
still carries it unsanitized, for a human or later tool to inspect.

Mirrors `test_p039_diff_redteam.py`'s own drive-through-the-real-cockpit
approach, this time proving the same guarantee at the artifact pane's
own rendering layer rather than the diff pane's. Skipped locally when
`bwrap` is not on `PATH`, matching every other suite in this project
that runs a real `verify`/`execute` call through the sandbox.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.sandbox import bwrap_available, run_sandboxed
from kestrel.tui.app import ArtifactPane, KestrelApp

_HOSTILE_CASE_ID = "ansi_escape_laden_payload"


def _can_initialize_network_namespace() -> bool:
    """Check whether this environment can actually run a sandboxed
    command at all -- the same prerequisite every other `bwrap`-gated
    suite in this project checks before trusting `bwrap_available()`
    alone."""
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
    pytest.mark.redteam,
    pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH"),
    pytest.mark.skipif(
        not _can_initialize_network_namespace(),
        reason="bwrap cannot initialize network namespace (missing capabilities or AppArmor restrictions)",
    ),
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_VERIFY = _CASSETTES / "toolcall_verify.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_PAYLOAD_FILENAME = "hostile_payload.txt"


def _find_case(case_id: str) -> InjectionCase:
    """Return the corpus case with `case_id`, raising `AssertionError`
    naming it if the corpus has none -- so a renamed or removed fixture
    fails with a clear, test-scoped error instead of an opaque
    `StopIteration`."""
    for case in load_corpus():
        if case.id == case_id:
            return case
    raise AssertionError(f"injection corpus case {case_id!r} not found")


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching the
    cassette's own `model` field."""
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


def _write_fixture_repo(tmp_path: Path, *, payload: str) -> None:
    """Write `payload` to a fixture file and configure KESTREL.md's own
    `test` command to `cat` it straight to stdout -- an exit-code-0
    command, so the resulting report's `passed=True` structure survives
    alongside the hostile content it carries."""
    (tmp_path / _PAYLOAD_FILENAME).write_text(payload, encoding="utf-8")
    (tmp_path / "KESTREL.md").write_text(
        f'```kestrel-verify\ntest = "cat {_PAYLOAD_FILENAME}"\n```\n',
        encoding="utf-8",
    )


async def test_artifact_pane_never_renders_raw_escape_bytes_from_a_hostile_command_output(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a KESTREL.md `test` command that echoes the
    `ansi_escape_laden_payload` corpus case's own payload to stdout, when
    a task submitted through `#task_input` runs `verify`, then the
    artifact pane's rendered markdown carries none of the payload's raw
    escape bytes while the report's own PASSED heading and command
    section survive, and the persisted report on disk still carries the
    real payload text, unbroken."""
    case = _find_case(_HOSTILE_CASE_ID)
    _write_fixture_repo(tmp_path, payload=case.payload)

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

        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        rendered = artifact_pane.source

        assert "\x1b" not in rendered
        assert "\x9b" not in rendered
        assert "\x07" not in rendered

        assert "# Verification: PASSED" in rendered
        assert "## test: `cat hostile_payload.txt`" in rendered
        assert "before" in rendered
        assert "after" in rendered

        artifacts_dir = tmp_path / ".kestrel" / "artifacts"
        persisted = list(artifacts_dir.glob("verification-*.md"))
        assert len(persisted) == 1
        persisted_text = persisted[0].read_text(encoding="utf-8")
        assert case.payload.strip() in persisted_text
