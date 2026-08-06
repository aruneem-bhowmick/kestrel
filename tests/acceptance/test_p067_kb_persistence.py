"""Acceptance suite proving knowledge-base notes persist across
sessions: at the storage layer directly, by closing and reopening a
`KnowledgeStore` at the same on-disk path, and end to end, by driving
two real `kestrel run` subprocesses against the same fixture repo and
confirming the second one's own outgoing request carries the first
one's committed learning.

Neither scenario exercises a new mechanism -- the first is the
Definition-of-Done-facing restatement of
`tests/unit/test_p057_kb_store.py`'s own
`test_closing_and_reopening_preserves_every_note` case, applied to a
real per-repo path rather than a hand-picked temp file; the second
drives the identical CLI wiring
`tests/system/test_p063_cli_run_kb_end_to_end.py`'s own suite already
covers, trimmed to the one assertion this document's own persistence
clause needs: that a second, independent process's own request body
actually carries what an earlier process's writeback committed.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.kb.service import DEFAULT_EMBEDDING_DIM
from kestrel.kb.store import KnowledgeNote, KnowledgeStore, resolve_kb_path

pytestmark = [pytest.mark.p067, pytest.mark.dod_phase_5, pytest.mark.acceptance]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_LEARNINGS_TWO = _CASSETTES / "learnings_two.sse"
_LEARNINGS_NONE = _CASSETTES / "learnings_none.sse"

_TIMEOUT_S = 30.0
# A fixed, non-zero, `DEFAULT_EMBEDDING_DIM`-long vector -- the same trick
# `test_p063_cli_run_kb_end_to_end.py`'s own `_KB_VECTOR` plays: the mock
# Ollama server below replays this exact vector for every embed call
# regardless of the text sent, so a note stored from one call and a query
# embedded by another always score as a perfect match.
_KB_VECTOR: tuple[float, ...] = (1.0,) + (0.0,) * (DEFAULT_EMBEDDING_DIM - 1)
_LEARNING_ONE_TEXT = (
    "This repo's Makefile targets assume tabs, not spaces, for recipe indentation"
)


def test_a_note_survives_closing_and_reopening_the_store_at_the_same_path(
    tmp_path: Path,
) -> None:
    """Given a note added through one `KnowledgeStore` opened at a real
    per-repo path (`resolve_kb_path(repo_root, global_=False)`, the
    exact path a live `KbService` would use), when that instance is
    closed and a brand new `KnowledgeStore` is opened at the identical
    path, then a search against the fresh instance still finds the note
    -- persistence proven at the storage layer a real caller actually
    resolves to, not a hand-picked temp file.
    """
    repo_root = tmp_path / "repo"
    db_path = resolve_kb_path(repo_root, global_=False)
    note = KnowledgeNote(
        id=None,
        text="Run `uv run pytest -m sanity` before pushing any change here",
        embedding=_KB_VECTOR,
        repo=str(repo_root.resolve()),
        tags=("testing",),
        source_task="p067-persist-1",
        timestamp=0.0,
    )

    first_session = KnowledgeStore(db_path=db_path, embedding_dim=DEFAULT_EMBEDDING_DIM)
    try:
        first_session.add_note(note)
    finally:
        first_session.close()

    second_session = KnowledgeStore(
        db_path=db_path, embedding_dim=DEFAULT_EMBEDDING_DIM
    )
    try:
        results = second_session.search(_KB_VECTOR, top_k=5)
    finally:
        second_session.close()

    assert {scored.note.text for scored in results} == {note.text}


def _write_kb_run_config(config_dir: Path, *, ollama_base_url: str) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    OpenRouter-routed chat entry and one Ollama-routed, `"local"`-tagged
    embedding entry pointed at `ollama_base_url`, and disabling self-
    critique -- identical in shape to
    `tests/system/test_p063_cli_run_kb_end_to_end.py`'s own
    `_write_kb_run_config`, since this scenario drives the identical CLI
    wiring, restated here as a Definition-of-Done proof rather than a
    first-time coverage suite. Returns the `kestrel.toml` path.
    """
    models_toml = config_dir / "models.toml"
    models_toml.write_text(
        f"""\
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

[[models]]
id = "nomic-embed-text"
backend = "ollama"
provider_model = "nomic-embed-text"
endpoint = "{ollama_base_url}"
context_window = 8192
max_output = 1
usd_per_mtok_input = 0
usd_per_mtok_output = 0
usd_per_mtok_cached = 0
supports_tools = false
supports_cache = false
tags = ["local"]
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

[managers.self_critique]
enabled = false
""",
        encoding="utf-8",
    )
    return kestrel_toml


def _run_env(openrouter_base: str) -> dict[str, str]:
    """Build the subprocess environment for a `kestrel run` call against
    the hermetic mock OpenAI-compatible backend -- identical in shape to
    `tests/system/test_p063_cli_run_kb_end_to_end.py`'s own `_run_env`.
    The Ollama route needs no such variable: its entry's own `endpoint`
    field already names the mock Ollama server directly.
    """
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


def test_a_second_run_task_retrieves_the_first_runs_committed_learning(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    mock_ollama_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a first real `kestrel run` subprocess that completes a task
    and approves both of a scripted writeback proposal's learnings over
    stdin, when a second, brand new `kestrel run` subprocess then runs
    against that same fixture repo, then its own retrieval step folds
    the first run's own committed learning into the very first outgoing
    model request -- `learnings persist across sessions`, proven through
    a real subprocess boundary rather than in-process shared state.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    ollama_base_url = mock_ollama_server(embeddings=[list(_KB_VECTOR)])
    config_path = _write_kb_run_config(tmp_path, ollama_base_url=ollama_base_url)

    first_base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE, _LEARNINGS_TWO]
    )
    first_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "say hello",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        input="y\ny\n",
        capture_output=True,
        encoding="utf-8",
        env=_run_env(first_base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert first_result.returncode == 0, first_result.stderr
    assert "kb: committed 2 learning(s)" in first_result.stdout

    second_captured: list[bytes] = []
    second_base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE, _LEARNINGS_NONE],
        capture=second_captured,
    )

    second_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "say hello again",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(second_base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert second_result.returncode == 0, second_result.stderr
    assert len(second_captured) >= 1
    first_request = second_captured[0]
    assert b"<<<UNTRUSTED:kb:" in first_request
    assert _LEARNING_ONE_TEXT.encode("utf-8") in first_request
