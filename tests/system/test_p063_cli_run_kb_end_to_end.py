"""System test: `kestrel run`'s own knowledge-base retrieval-before and
writeback-after-Walkthrough CLI wiring, driven as a real subprocess
against the packaged console script -- the first end-to-end exercise of
retrieval (`kestrel.kb.retrieval.build_kb_context`) and writeback
(`kestrel.kb.writeback`) together, through the exact command a real user
would type.

Two real subprocess `kestrel run` invocations share one fixture repo and
one mock Ollama embedding server: the first completes a task, approves
both of a scripted writeback proposal's learnings over stdin, and a
freshly opened `KnowledgeStore` confirms they landed in
`.kestrel/kb.sqlite3`; the second, a brand new process, proves its own
retrieval step actually finds and injects that first run's own committed
notes into its outgoing model call -- retrieval-after-writeback round
-tripping through a real subprocess boundary, not merely reused
in-process state.

Self-critique is disabled via config for both runs, exactly like
`test_p059_kb_reaches_dispatch_context.py`'s own system suite, so the
only model calls either run makes are its own scripted task turn and
(when the task completes) the writeback proposal call -- this suite is
about the retrieval/writeback wiring, not self-critique routing already
covered elsewhere.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.kb.service import DEFAULT_EMBEDDING_DIM
from kestrel.kb.store import KnowledgeStore, resolve_kb_path

pytestmark = [pytest.mark.p063, pytest.mark.system]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
_LEARNINGS_TWO = _CASSETTES / "learnings_two.sse"
_LEARNINGS_NONE = _CASSETTES / "learnings_none.sse"

_TIMEOUT_S = 30.0
# A fixed, non-zero, `DEFAULT_EMBEDDING_DIM`-long vector: `KnowledgeStore
# .search` rejects an all-zero query, and the mock Ollama server below
# replays this exact vector for every embed call regardless of the text
# sent, so a note stored from one call and a query embedded by another
# always score as a perfect match -- no real semantic model is needed to
# prove the retrieval/writeback round trip itself.
_KB_VECTOR: tuple[float, ...] = (1.0,) + (0.0,) * (DEFAULT_EMBEDDING_DIM - 1)
_LEARNING_ONE_TEXT = (
    "This repo's Makefile targets assume tabs, not spaces, for recipe indentation"
)
_LEARNING_TWO_TEXT = "Run `uv run pytest -m sanity` before pushing any change here"


def _write_kb_run_config(config_dir: Path, *, ollama_base_url: str) -> Path:
    """Write a `kestrel.toml` + `models.toml` pair naming one
    OpenRouter-routed chat entry and one Ollama-routed, `"local"`-tagged
    embedding entry pointed at `ollama_base_url`, and disabling self-
    critique so a scripted task's only real calls are its own turn and
    (once it completes) the writeback proposal -- returns the
    `kestrel.toml` path. The chat entry's own base URL is redirected to
    the hermetic mock server through the `KESTREL_OPENROUTER_BASE_URL`
    environment variable seam (see `_run_env`), never through this file;
    the knowledge base is left at its own default, `[kb] enabled = true`,
    since exercising that default end to end is this suite's whole point.
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
    `tests/acceptance/test_p023_dod_phase_1.py`'s own helper. The Ollama
    route needs no such variable: its entry's own `endpoint` field
    already names the mock Ollama server directly."""
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


def test_kb_retrieval_and_writeback_round_trip_across_two_real_subprocesses(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    mock_ollama_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a mock Ollama server always answering with the same fixed
    embedding vector, when a first real `kestrel run` subprocess
    completes a task and approves both learnings a scripted writeback
    proposal offers, then it prints `kb: committed 2 learning(s)` and a
    freshly, directly opened `KnowledgeStore` at the fixture repo's own
    `.kestrel/kb.sqlite3` holds both notes' exact text. When a second,
    brand new `kestrel run` subprocess against that same repo then runs,
    its own retrieval step finds those same two notes and folds their
    text into the very first outgoing model request -- proving retrieval
    genuinely reads back what an earlier, separate process's own
    writeback committed, not merely in-process shared state.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    ollama_base_url = mock_ollama_server(embeddings=[list(_KB_VECTOR)])
    config_path = _write_kb_run_config(tmp_path, ollama_base_url=ollama_base_url)

    # --- first run: completes, proposes, and commits two learnings ---
    first_captured: list[bytes] = []
    first_base_url = mock_openai_server(
        cassette_sequence=[_DONE_CASSETTE, _LEARNINGS_TWO],
        capture=first_captured,
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
    assert "TASK_COMPLETE" in first_result.stdout
    assert "kb: committed 2 learning(s)" in first_result.stdout

    store = KnowledgeStore(
        db_path=resolve_kb_path(repo_dir, global_=False),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )
    try:
        stored = store.search(_KB_VECTOR, top_k=10)
    finally:
        store.close()
    stored_texts = {scored.note.text for scored in stored}
    assert stored_texts == {_LEARNING_ONE_TEXT, _LEARNING_TWO_TEXT}

    # --- second run: a fresh process retrieves what the first committed ---
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
    assert "TASK_COMPLETE" in second_result.stdout
    assert len(second_captured) >= 1
    first_request = second_captured[0]
    assert b"<<<UNTRUSTED:kb:" in first_request
    assert _LEARNING_ONE_TEXT.encode("utf-8") in first_request
    assert _LEARNING_TWO_TEXT.encode("utf-8") in first_request
