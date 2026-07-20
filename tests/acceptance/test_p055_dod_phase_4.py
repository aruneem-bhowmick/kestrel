"""Acceptance suite proving the plan-review-execute-walkthrough workflow
end to end, the way a real user would actually work through it -- not
each contributing suite's own isolated scenario re-asserted in
isolation, but one assembled arc per clause: a PLAN-mode task lands a
persisted `ImplementationPlan` artifact; a reviewer's line comments feed
back into a revision of that same plan, continuing the same task rather
than starting a new one; switching to FAST mode and resubmitting
executes the plan as a real, tool-enabled continuation, picking its own
turn numbering up where the plan left off; a plain FAST-mode run lands a
persisted `Walkthrough` naming its own real verification result and
matching its own priced cost exactly; and reasoning effort maps onto
each backend's own native knob distinctly per mode, proven against both
supported backends at once -- the literal "GLM max/high verified" claim.

Every scenario below drives a real `KestrelApp`, `kestrel.cli.main`, or
`kestrel.task_setup.build_task_deps` against the hermetic mock chat-
completions server (see `tests/fixtures/mock_openai.py`) and a real
fixture repo -- none of it depends on a live model or a live credential.
Every collaborator exercised here already has its own dedicated suite
elsewhere; this module's own job is composing them into the same
end-to-end shape a person using Kestrel would actually walk through.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from kestrel.agent.loop import run_task
from kestrel.agent.plan import (
    ImplementationPlan,
    parse_plan_lines,
    render_plan_markdown,
)
from kestrel.cli import main
from kestrel.config import KestrelConfig, ManagersConfig, SelfCritiqueConfig
from kestrel.managers.mode import Mode, ModeManager
from kestrel.registry.model import ModelEntry, Registry
from kestrel.repl import sanitize_terminal
from kestrel.task_setup import build_task_deps
from kestrel.tools.sandbox import bwrap_available, run_sandboxed
from kestrel.tui import app as app_module
from kestrel.tui.app import ArtifactPane, DiffPane, KestrelApp
from kestrel.tui.plan_comment_modal import PlanCommentModal

pytestmark = [
    pytest.mark.p055,
    pytest.mark.acceptance,
    pytest.mark.system,
    pytest.mark.dod_phase_4,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_TOOLCALL_VERIFY = _CASSETTES / "toolcall_verify.sse"
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_GREET_STUB = "# TODO: implement greet\n"
_PLAN_TASK_TEXT = "read src/greet.py, then plan how to add auth"
_EXECUTE_TEXT = "implement it"
_INITIAL_PLAN_TEXT = (
    "1. Add an authentication middleware module.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_REVISED_PLAN_TEXT = (
    "1. Add an authentication middleware module built on Alembic.\n"
    "2. Wire it into the request pipeline.\n"
    "3. Add unit tests for the new middleware."
)
_COMMENT_TEXT = "use Alembic instead"

_TASK_ID_RE = re.compile(r"^task_id: (?P<task_id>\S+)$", re.MULTILINE)
_TOTAL_USD_RE = re.compile(r"^total_usd: \$(?P<total_usd>[\d.]+)$", re.MULTILINE)
_WALKTHROUGH_LINE_RE = re.compile(r"^walkthrough: (?P<path>.+)$", re.MULTILINE)


def _registry() -> Registry:
    """A single OpenRouter-routed `Registry` entry matching every
    scripted cassette's own `model` field."""
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


def _write_fixture_repo(repo_root: Path) -> None:
    """Write a small fixture repo satisfying every scripted tool call
    across this suite's PLAN/FAST scenarios: a `src/greet.py` module for
    a `read_file` call, and a top-level `greet.py` stub for an
    `edit_file` call to replace."""
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (repo_root / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


def _write_plan_reply_cassette(path: Path, *, text: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply is
    `text` verbatim and requests no tool calls -- standing in for a
    PLAN-mode model turn, whose reply is the plan itself."""
    chunks = [
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000050,
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
            "created": 1700000050,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-plan",
            "object": "chat.completion.chunk",
            "created": 1700000050,
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


def _spy_on_turn_started(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap `TuiLoopObserver.on_turn_started` to record every `turn_id`
    it receives, in arrival order, across every task this test drives --
    proving a continued task's own turn numbering picks up where the
    prior one left off rather than resetting back to `1`."""
    observed: list[int] = []
    original = app_module.TuiLoopObserver.on_turn_started

    def _wrapped(self: object, *, turn_id: int, active_model_id: str) -> None:
        """Record `turn_id`, then forward the call to the real hook so
        the status bar still refreshes exactly as it would unspied."""
        observed.append(turn_id)
        original(self, turn_id=turn_id, active_model_id=active_model_id)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module.TuiLoopObserver, "on_turn_started", _wrapped)
    return observed


# --- DoD: PLAN mode emits an ImplementationPlan artifact --------------------


@pytest.mark.ui
async def test_dod_plan_mode_emits_implementation_plan_artifact(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the cockpit's mode switched to `"plan"` and a mock server
    scripted to reply with one `read_file` exploration turn followed by
    a plain numbered-plan reply, when a task is submitted through
    `#task_input`, then: a `.kestrel/artifacts/plan-*.md` file exists
    whose content is exactly `render_plan_markdown`'s own output for the
    parsed reply, and the artifact pane shows that identical rendering.
    """
    _write_fixture_repo(tmp_path)

    plan_cassette = _write_plan_reply_cassette(
        tmp_path / "plan-reply.sse", text=_INITIAL_PLAN_TEXT
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            plan_cassette,
            _CRITIQUE_APPROVE,
        ],
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
        pilot.app.action_set_mode("plan")
        await pilot.pause()

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _PLAN_TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        task_id = pilot.app._last_completed_task_id
        assert task_id is not None
        expected_plan = ImplementationPlan(
            task_id=task_id,
            raw_text=_INITIAL_PLAN_TEXT,
            lines=parse_plan_lines(_INITIAL_PLAN_TEXT),
        )
        expected_markdown = render_plan_markdown(expected_plan)

        persisted_path = tmp_path / ".kestrel" / "artifacts" / f"plan-{task_id}.md"
        assert persisted_path.is_file()
        assert persisted_path.read_text(encoding="utf-8") == expected_markdown

        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(expected_markdown)


# --- DoD: user comments on plan lines are incorporated ----------------------


@pytest.mark.ui
async def test_dod_user_comments_on_plan_lines_are_incorporated(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a completed PLAN-mode task on screen, when `c` opens
    `PlanCommentModal`, one comment against line 1 is submitted, and the
    (now-empty) `#task_input` is resubmitted, then: the revision
    continues the same task id via `resume_task` (never a fresh
    `run_task`); the artifact pane shows the revised plan's own
    rendering; and `.kestrel/artifacts/` carries both the original
    `plan-{task_id}.md` and the revision's own numeric-suffixed
    `plan-{task_id}-1.md`, each matching `render_plan_markdown`'s output
    for its own reply exactly.
    """
    _write_fixture_repo(tmp_path)

    initial_plan_cassette = _write_plan_reply_cassette(
        tmp_path / "initial-plan-reply.sse", text=_INITIAL_PLAN_TEXT
    )
    revised_plan_cassette = _write_plan_reply_cassette(
        tmp_path / "revised-plan-reply.sse", text=_REVISED_PLAN_TEXT
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            initial_plan_cassette,
            _CRITIQUE_APPROVE,
            revised_plan_cassette,
            _CRITIQUE_APPROVE,
        ],
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
        pilot.app.action_set_mode("plan")
        await pilot.pause()

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _PLAN_TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        initial_task_id = pilot.app._last_completed_task_id
        assert initial_task_id is not None

        await pilot.press("f4")
        await pilot.press("c")
        await pilot.pause()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, PlanCommentModal)
        modal.query_one("#plan_comment_line_number", Input).value = "1"
        modal.query_one("#plan_comment_text", Input).value = _COMMENT_TEXT
        modal.query_one("#submit", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert len(pilot.app._pending_plan_comments) == 1

        task_input.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        revised_task_id = pilot.app._last_completed_task_id
        assert revised_task_id == initial_task_id

        expected_revised_plan = ImplementationPlan(
            task_id=revised_task_id,
            raw_text=_REVISED_PLAN_TEXT,
            lines=parse_plan_lines(_REVISED_PLAN_TEXT),
        )
        artifact_pane = pilot.app.query_one("#artifact", ArtifactPane)
        assert artifact_pane.source == sanitize_terminal(
            render_plan_markdown(expected_revised_plan)
        )

        artifacts_dir = tmp_path / ".kestrel" / "artifacts"
        initial_path = artifacts_dir / f"plan-{initial_task_id}.md"
        revised_path = artifacts_dir / f"plan-{initial_task_id}-1.md"
        expected_initial_plan = ImplementationPlan(
            task_id=initial_task_id,
            raw_text=_INITIAL_PLAN_TEXT,
            lines=parse_plan_lines(_INITIAL_PLAN_TEXT),
        )
        assert initial_path.read_text(encoding="utf-8") == render_plan_markdown(
            expected_initial_plan
        )
        assert revised_path.read_text(encoding="utf-8") == render_plan_markdown(
            expected_revised_plan
        )


# --- DoD: FAST/EXECUTE honors the plan ---------------------------------------


@pytest.mark.ui
async def test_dod_fast_execute_honors_the_plan(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a completed PLAN-mode task on screen, when the cockpit's
    mode switches to `"fast"` and `#task_input` is resubmitted with a
    typed execution instruction, then: a real `edit_file` mutation lands
    and the diff pane shows it, and the turn-id sequence observed across
    the plan task and its execution continues `1, 2, 3, 4` -- picking up
    from the plan task's own turn count rather than resetting back to
    `1`.
    """
    _write_fixture_repo(tmp_path)

    plan_cassette = _write_plan_reply_cassette(
        tmp_path / "plan-reply.sse", text=_INITIAL_PLAN_TEXT
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            plan_cassette,
            _CRITIQUE_APPROVE,
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
    )
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)

    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=tmp_path,
    )
    turns_observed = _spy_on_turn_started(monkeypatch)

    async with app.run_test() as pilot:
        pilot.app.action_set_mode("plan")
        await pilot.pause()

        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _PLAN_TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        plan_task_id = pilot.app._last_completed_task_id
        assert plan_task_id is not None

        pilot.app.action_set_mode("fast")
        await pilot.pause()

        task_input.focus()
        task_input.value = _EXECUTE_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app._last_completed_task_id == plan_task_id
        assert pilot.app._plan_task_id is None
        assert pilot.app._last_plan is None
        assert turns_observed == [1, 2, 3, 4]

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        diff_text = diff_pane.content.code
        assert "-# TODO: implement greet" in diff_text
        assert "+def greet(name: str) -> str:" in diff_text


# --- DoD: Walkthrough artifact generated with verification + cost -----------


def _can_initialize_network_namespace() -> bool:
    """Whether this environment can actually run a sandboxed command at
    all -- the same prerequisite the rest of this codebase's own real-
    `bwrap` suites check before trusting `bwrap_available()` alone."""
    if not bwrap_available():
        return False
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_sandboxed(["true"], repo_root=Path(tmpdir), timeout_s=5.0)
            return result.exit_code == 0 and not result.timed_out
    except Exception:
        return False


def _write_run_config(config_dir: Path) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    OpenRouter-routed entry, redirected to the hermetic mock server
    entirely through the `KESTREL_OPENROUTER_BASE_URL` environment seam."""
    models_toml = config_dir / "models.toml"
    models_toml.write_text(
        """\
[[models]]
id = "glm-5.2"
backend = "openrouter"
provider_model = "z-ai/glm-5.2"
api_key_env = "OPENROUTER_API_KEY"
context_window = 200000
max_output = 16384
usd_per_mtok_input = 0.60
usd_per_mtok_output = 2.20
usd_per_mtok_cached = 0.11
supports_tools = true
supports_cache = true
""",
        encoding="utf-8",
    )
    kestrel_toml = config_dir / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "glm-5.2"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    return kestrel_toml


def _run_env(monkeypatch: pytest.MonkeyPatch, *, openrouter_base: str) -> None:
    """Set the environment variables `main`'s own config/registry
    resolution and the provider layer need against the mock backend."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", openrouter_base)
    monkeypatch.delenv("KESTREL_CONFIG", raising=False)


def _write_kestrel_md(repo_root: Path, *, test_command: str) -> None:
    """Configure a single, trivially-passing `test` command for
    `verify` to run against `repo_root`."""
    (repo_root / "KESTREL.md").write_text(
        f'```kestrel-verify\ntest = "{test_command}"\n```\n', encoding="utf-8"
    )


@pytest.mark.cost_regression
@pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH")
@pytest.mark.skipif(
    not _can_initialize_network_namespace(),
    reason="bwrap cannot initialize network namespace (missing capabilities or AppArmor restrictions)",
)
def test_dod_walkthrough_artifact_generated_with_verification_and_cost(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a fixture repo whose KESTREL.md configures a trivially-
    passing `test` command, and a mock server scripted to reply with a
    real `verify` tool call (run through the actual `bwrap` sandbox, not
    a stand-in) followed by a plain no-more-tools reply, when `kestrel
    run` executes through `main`, then: a `verification-*.md` artifact
    is persisted naming a real, passing outcome; the persisted
    `walkthrough-*.md` artifact's own `## Verification` section embeds
    that exact same report, not a placeholder; and its `total_usd` line
    matches the summary's own printed `total_usd` figure exactly --
    `LoopResult.total_usd`, not merely "a nonzero number."
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _write_kestrel_md(repo_dir, test_command="true")

    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_VERIFY,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ],
    )
    _run_env(monkeypatch, openrouter_base=base_url)
    config_path = _write_run_config(tmp_path)

    exit_code = main(
        [
            "run",
            "make sure the repo's own tests pass",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "reason: TASK_COMPLETE" in out

    task_id_match = _TASK_ID_RE.search(out)
    assert task_id_match is not None, out
    total_usd_match = _TOTAL_USD_RE.search(out)
    assert total_usd_match is not None, out
    walkthrough_match = _WALKTHROUGH_LINE_RE.search(out)
    assert walkthrough_match is not None, out

    artifacts_dir = repo_dir / ".kestrel" / "artifacts"
    verification_persisted = list(artifacts_dir.glob("verification-*.md"))
    assert len(verification_persisted) == 1
    verification_text = verification_persisted[0].read_text(encoding="utf-8")
    assert "# Verification: PASSED" in verification_text

    walkthrough_path = Path(walkthrough_match["path"])
    assert walkthrough_path.exists()
    walkthrough_text = walkthrough_path.read_text(encoding="utf-8")

    assert "## Verification" in walkthrough_text
    assert walkthrough_text.endswith(verification_text)
    assert f"- total_usd: ${total_usd_match['total_usd']}" in walkthrough_text


# --- DoD: effort levels map correctly per backend (GLM max/high verified) ---


def _zai_registry(*, endpoint: str) -> Registry:
    """A single-entry `Registry` mirroring the packaged default's own
    zai-routed GLM entry, pointed at `endpoint` in place of Z.ai's real
    one -- the zai branch has no environment-variable redirection seam,
    so a test points it at a fake backend the same way a real deployment
    points it at Z.ai's own endpoint: through the registry entry itself."""
    entry = ModelEntry(
        id="glm-5.2-zai",
        backend="zai",
        provider_model="glm-5.2",
        endpoint=endpoint,
        api_key_env="ZAI_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={"glm-5.2-zai": entry}, source=None)


_NO_SELF_CRITIQUE = KestrelConfig(
    managers=ManagersConfig(self_critique=SelfCritiqueConfig(enabled=False))
)


@pytest.mark.parametrize(
    ("backend", "mode", "model_id", "expected_effort_key"),
    [
        pytest.param("openrouter", "fast", "glm-5.2", "medium", id="openrouter-fast"),
        pytest.param("openrouter", "plan", "glm-5.2", "high", id="openrouter-plan"),
        pytest.param("zai", "fast", "glm-5.2-zai", "high", id="zai-fast"),
        pytest.param("zai", "plan", "glm-5.2-zai", "max", id="zai-plan"),
    ],
)
async def test_dod_effort_levels_map_correctly_per_backend_glm_max_high_verified(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    mode: Mode,
    model_id: str,
    expected_effort_key: str,
) -> None:
    """Given a `Registry` entry for `backend` and a task submitted in
    `mode` via `build_task_deps(mode_manager=ModeManager(mode=mode))`,
    when driven against a mock server capturing the raw outgoing
    request, then the captured body's own effort-bearing key
    (`reasoning.effort` for openrouter, `thinking.effort` for zai)
    equals `expected_effort_key` -- FAST maps to "medium"/"high" and
    PLAN maps to "high"/"max" for openrouter/zai respectively, the
    literal "GLM max/high verified" clause, exercised against both
    backends the packaged default registry actually routes GLM through.
    """
    captured: list[bytes] = []
    if backend == "openrouter":
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        base_url = mock_openai_server(_DONE_CASSETTE, capture=captured)
        monkeypatch.setenv("KESTREL_OPENROUTER_BASE_URL", base_url)
        registry = _registry()
    else:
        monkeypatch.setenv("ZAI_API_KEY", "sk-test-key")
        base_url = mock_openai_server(_DONE_CASSETTE, capture=captured)
        registry = _zai_registry(endpoint=base_url)

    task_id = f"effort-{backend}-{mode}"
    setup = build_task_deps(
        config=_NO_SELF_CRITIQUE,
        registry=registry,
        model_id=model_id,
        kestrel_md=None,
        repo_root=tmp_path,
        task_id=task_id,
        mode_manager=ModeManager(mode=mode),
    )

    await run_task("hi", setup.deps, task_id)

    assert len(captured) == 1
    body = json.loads(captured[0])
    if backend == "openrouter":
        assert body["reasoning"]["effort"] == expected_effort_key
    else:
        assert body["thinking"]["effort"] == expected_effort_key
