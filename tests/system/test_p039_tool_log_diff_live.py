"""System test: a scripted task that both reads a file and edits one
drives the tool log's live started/finished lines, the diff pane's real
unified-diff rendering, and the loading indicator's visibility --
exercising `TuiLoopObserver.on_tool_call_started`/`on_tool_call_finished`
against the real cockpit, not stand-in panes.

Reuses `test_p038_tui_conversation_stream.py`'s own mock-server-plus-
fixture-repo pattern. Neither `read_file` nor `edit_file` touches the
`bwrap` sandbox (only `execute`/`verify` do), so this suite needs no
`bwrap`-availability guard.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from textual.widgets import Input

from kestrel.config import KestrelConfig
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tui import app as app_module
from kestrel.tui.app import DiffPane, KestrelApp, LoadingIndicator, ToolLogPane
from kestrel.tui.observer_bridge import TuiLoopObserver

pytestmark = [pytest.mark.p039, pytest.mark.system, pytest.mark.ui]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
# `build_task_deps` now routes every turn's self-critique check through
# its own real, routed call by default, so the scripted sequence below
# must reply to it too -- one extra request per real turn, interleaved
# right after that turn's own.
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "read src/greet.py, then implement greet in greet.py"


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
    """Write a fixture repo satisfying both scripted tool calls: a
    `src/greet.py` module for the `read_file` call, and a top-level
    `greet.py` stub -- the exact anchor `toolcall_edit_greet.sse`'s own
    `edit_file` call replaces -- for the `edit_file` call."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (tmp_path / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


def _spy_on_inflight_changes(
    monkeypatch: pytest.MonkeyPatch, app: KestrelApp
) -> list[bool]:
    """Wrap `TuiLoopObserver.__init__` so every `on_inflight_change`
    call this test's own task makes also records the loading
    indicator's own real, post-update `display` value into the
    returned list -- not a value merely inferred from `count` -- so
    this test actually proves `KestrelApp`'s own closure sets that
    widget's visibility correctly, rather than only proving the
    observer computed the right count.

    `app` is passed in (already constructed, though not yet mounted)
    rather than queried lazily: `_wrapped` itself only ever runs once a
    task is actually submitted, by which point `app` is mounted, so a
    plain reference captured now resolves correctly then.
    """
    observed: list[bool] = []
    original_init = TuiLoopObserver.__init__

    def _spy_init(self: TuiLoopObserver, *args: object, **kwargs: object) -> None:
        """Swap in `_wrapped` for whatever `on_inflight_change` callback
        this call site passed, then build the real observer as usual."""
        real_callback = kwargs.pop("on_inflight_change", None)

        def _wrapped(count: int) -> None:
            """Forward `count` to the real callback the app itself
            built -- updating the loading indicator's own `display`
            attribute for real -- then record that attribute's actual,
            post-update value."""
            if real_callback is not None:
                real_callback(count)  # type: ignore[operator]
            loading_indicator = app.query_one("#loading_indicator", LoadingIndicator)
            observed.append(loading_indicator.display)

        kwargs["on_inflight_change"] = _wrapped
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module.TuiLoopObserver, "__init__", _spy_init)
    return observed


async def test_tool_log_diff_pane_and_spinner_update_live(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a fixture repo and a mock server scripted to reply with a
    `read_file` call, then an `edit_file` call, then a plain no-more-
    tools reply, when a task is submitted through `#task_input`, then:
    (1) the tool log shows a started/finished pair for `read_file`
    followed by a started/finished pair for `edit_file`, in that order;
    (2) the diff pane's rendered content shows the real `edit_file`
    mutation's own before/after text; (3) the loading indicator's
    visibility toggles true then false around each of the two calls,
    and is never left `True` once the task ends.
    """
    _write_fixture_repo(tmp_path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
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
    inflight_observed = _spy_on_inflight_changes(monkeypatch, app)

    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        tool_log = pilot.app.query_one("#tool_log", ToolLogPane)
        lines = [strip.text for strip in tool_log.lines]
        content = "\n".join(lines)

        started_read = content.index("-> read_file(")
        finished_read = content.index("<- read_file (")
        started_edit = content.index("-> edit_file(")
        finished_edit = content.index("<- edit_file (")
        assert started_read < finished_read < started_edit < finished_edit

        diff_pane = pilot.app.query_one("#diff", DiffPane)
        diff_text = diff_pane.content.code
        assert "-# TODO: implement greet" in diff_text
        assert "+def greet(name: str) -> str:" in diff_text
        assert '+    return f"Hello, {name}!"' in diff_text

        assert inflight_observed == [True, False, True, False]

        loading_indicator = pilot.app.query_one("#loading_indicator")
        assert loading_indicator.display is False
