"""Manually invoked script producing the cockpit's screenshot set for the
README.

Not a pytest test and not run in CI: a rendered SVG screenshot embeds
font and layout metrics that can shift across Textual point releases
with nothing about the application itself changing, so pinning these
byte-exact the way ``tests/golden/*.golden`` pins plain text would flag
routine dependency bumps instead of real regressions. Run this by hand
after a visible UI change --

    uv run python scripts/tui_screenshots.py

-- and review the output by eye before committing it.

Drives a real ``KestrelApp`` against a hermetic mock chat-completions
server and a small fixture repo, submits the same scripted task
``tests/system/test_p039_tool_log_diff_live.py`` drives, lets it
complete, then captures three named SVG files under
``assets/screenshots/``: the conversation pane and tool log right after
the task finishes, the diff pane in focus, and the command palette open
over the assembled cockpit.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from fixtures.mock_openai import MockOpenAIServer  # noqa: E402
from textual.widgets import Input  # noqa: E402

from kestrel.config import KestrelConfig  # noqa: E402
from kestrel.registry.model import ModelEntry, Registry  # noqa: E402
from kestrel.tui.app import DiffPane, KestrelApp  # noqa: E402

_CASSETTES = _REPO_ROOT / "tests" / "fixtures" / "cassettes"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"

_GREET_STUB = "# TODO: implement greet\n"
_TASK_TEXT = "read src/greet.py, then implement greet in greet.py"
_SCREENSHOTS_DIR = _REPO_ROOT / "assets" / "screenshots"


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


def _write_fixture_repo(repo_root: Path) -> None:
    """Write the same small fixture repo
    `test_p039_tool_log_diff_live.py` scripts its own task against: a
    `src/greet.py` module for the `read_file` call, and a top-level
    `greet.py` stub for the `edit_file` call to replace."""
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "greet.py").write_text(
        "# hello from the fixture module\n", encoding="utf-8"
    )
    (repo_root / "greet.py").write_text(_GREET_STUB, encoding="utf-8")


async def _capture_screenshots(repo_root: Path) -> None:
    """Drive one scripted task to completion, then save the three named
    screenshots the README embeds -- the conversation and tool log,
    the diff pane in focus, and the command palette open over the
    cockpit."""
    app = KestrelApp(
        config=KestrelConfig(),
        registry=_registry(),
        model_id="glm-5.2",
        kestrel_md=None,
        repo_root=repo_root,
    )
    async with app.run_test() as pilot:
        task_input = pilot.app.query_one("#task_input", Input)
        task_input.focus()
        task_input.value = _TASK_TEXT
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        pilot.app.save_screenshot(
            filename="conversation-and-tools.svg", path=str(_SCREENSHOTS_DIR)
        )

        pilot.app.query_one("#diff", DiffPane).focus()
        await pilot.pause()
        pilot.app.save_screenshot(filename="diff-view.svg", path=str(_SCREENSHOTS_DIR))

        await pilot.press("ctrl+p")
        await pilot.pause()
        pilot.app.save_screenshot(
            filename="command-palette.svg", path=str(_SCREENSHOTS_DIR)
        )


def main() -> None:
    """Boot a hermetic mock chat-completions server, point a fresh
    fixture repo's scripted task at it, and capture the cockpit's
    screenshot set."""
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-tui-screenshots")
    server = MockOpenAIServer(
        cassette_path=None,
        cassette_sequence=[_TOOLCALL_READ_FILE, _TOOLCALL_EDIT_GREET, _DONE_CASSETTE],
        status_code=200,
        extra_headers=None,
    )
    server.start()
    os.environ["KESTREL_OPENROUTER_BASE_URL"] = server.base_url
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            _write_fixture_repo(repo_root)
            asyncio.run(_capture_screenshots(repo_root))
    finally:
        server.stop()
    print(f"wrote 3 screenshots to {_SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
