"""Acceptance suite encoding every machine-checkable clause of the
Definition-of-Done this project has converged on: `kestrel run` finishes
a real small task end to end in a test repo, a destructive tool call is
gated behind (and actually stopped by) the approval prompt, `kestrel
undo` reverts a completed run's own file mutations exactly, and the
injection corpus survives a full loop iteration -- not merely
`kestrel.security.framing.frame_untrusted` called directly -- without a
single forged delimiter reaching the wire unescaped.

Every scenario here drives the packaged `kestrel` console script as a
real subprocess against the hermetic mock backend (see
``tests/fixtures/mock_openai.py``); none of it depends on a live model
or a live credential. The one clause that requires a real provider call
has its own budget-capped, opt-in twin in
``tests/e2e/test_p023_dod_live.py``.

Every `kestrel run` invocation below passes `--no-require-verification`:
none of these scenarios scripts a `verify` tool call, and `run` now
withholds completion from a no-tool-calls turn until one has passed by
default, so leaving the flag at its default would silently change what
each scenario here actually exercises rather than exercising it at all.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.tools.sandbox import bwrap_available

pytestmark = [
    pytest.mark.p023,
    pytest.mark.acceptance,
    pytest.mark.system,
    pytest.mark.dod_phase_1,
]

_CASSETTES = Path(__file__).resolve().parent.parent / "fixtures" / "cassettes"
_TOOLCALL_EDIT_GREET = _CASSETTES / "toolcall_edit_greet.sse"
_TOOLCALL_EXECUTE_PYTEST = _CASSETTES / "toolcall_execute_pytest.sse"
_TOOLCALL_EXECUTE_RM = _CASSETTES / "toolcall_execute_rm.sse"
_TOOLCALL_READ_FILE = _CASSETTES / "toolcall_read_file.sse"
_TOOLCALL_READ_PAYLOAD = _CASSETTES / "toolcall_read_file_payload.sse"
_DONE_CASSETTE = _CASSETTES / "done_no_more_tools.sse"
# `kestrel run` now routes every turn's self-critique check through its
# own real, routed call by default (`[managers.self_critique].enabled`),
# so every scripted scenario below must reply to it too -- one extra
# request per real turn, interleaved right after that turn's own -- or
# the mock server's fixed reply sequence drifts out of step with which
# request is actually which.
_CRITIQUE_APPROVE = _CASSETTES / "critique_approve.sse"

_TIMEOUT_S = 30.0
_TASK_ID_RE = re.compile(r"^task_id: (?P<task_id>\S+)$", re.MULTILINE)
_GREET_STUB = "# TODO: implement greet\n"
_GREET_TEST = 'from greet import greet\n\n\ndef test_greet() -> None:\n    assert greet("World") == "Hello, World!"\n'

_CORPUS_CASES = load_corpus()


def _write_run_config(config_dir: Path) -> Path:
    """Write a ``kestrel.toml`` + ``models.toml`` pair naming one
    OpenRouter-routed entry, and return the ``kestrel.toml`` path --
    everything `kestrel run`/`kestrel undo` need to resolve config,
    registry, and starting model. The entry's actual base URL is
    redirected to the hermetic mock server entirely through the
    `KESTREL_OPENROUTER_BASE_URL` environment variable seam (see
    `_run_env`), never through this file."""
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


def _run_env(openrouter_base: str) -> dict[str, str]:
    """Build the subprocess environment for a `kestrel run`/`kestrel undo`
    call against the hermetic mock backend: a fixed, fake credential
    satisfies the api-key precondition, and the OpenRouter route is
    redirected at the mock server via its documented test seam."""
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-test-openrouter"
    env["KESTREL_OPENROUTER_BASE_URL"] = openrouter_base
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KESTREL_CONFIG", None)
    return env


def _write_echo_cassette(path: Path, *, payload: str) -> Path:
    """Write a one-turn, text-only SSE cassette whose assistant reply
    echoes ``payload`` back verbatim, as if naively quoting a file it
    just read -- standing in for a model that repeats untrusted content
    in its own words rather than treating it as an instruction. Built
    with `json.dumps` (rather than hand-escaped strings) so an
    arbitrarily hostile payload -- embedded newlines, quotes, control
    characters, zero-width characters -- always serializes into a
    single well-formed SSE data line, whatever it contains.
    """
    chunks = [
        {
            "id": "chatcmpl-echo",
            "object": "chat.completion.chunk",
            "created": 1700000007,
            "model": "z-ai/glm-5.2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"The file said: {payload}",
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-echo",
            "object": "chat.completion.chunk",
            "created": 1700000007,
            "model": "z-ai/glm-5.2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-echo",
            "object": "chat.completion.chunk",
            "created": 1700000007,
            "model": "z-ai/glm-5.2",
            "choices": [],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
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


@pytest.mark.skipif(not bwrap_available(), reason="bwrap not found on PATH")
def test_dod_task_completes_end_to_end(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a fixture repo with one module missing a function and one
    nearly-empty test file already exercising it, and a mock server
    scripted to edit the module, run pytest, and stop, when `kestrel
    run` executes as a real subprocess, then it exits 0, prints
    `TASK_COMPLETE`, and the module file itself now carries the
    implemented function on disk -- the canonical "add a function +
    unit test, make it pass" task shape, completed end to end through
    the packaged console entry point rather than `run_task` called
    in-process.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")
    (repo_dir / "test_greet.py").write_text(_GREET_TEST, encoding="utf-8")

    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _TOOLCALL_EXECUTE_PYTEST,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ]
    )
    config_path = _write_run_config(tmp_path)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "implement the missing greet function in greet.py, then run pytest",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "TASK_COMPLETE" in result.stdout
    assert "def greet" in (repo_dir / "greet.py").read_text(encoding="utf-8")


def test_dod_approval_gates_destructive_ops(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a scripted task whose second tool call is a destructive
    `execute(["rm", "somefile"])`, run through the real stdin-prompt
    approval path with a piped "n" answer, when `kestrel run` executes,
    then the approval request is printed to stdout, the target file
    survives on disk, and the run still reaches a defined termination
    (never hangs or crashes on the denial).
    """
    repo_dir = tmp_path / "repo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "greet.py").write_text("# greeting module\n", encoding="utf-8")
    target = repo_dir / "somefile"
    target.write_text("keep me\n", encoding="utf-8")

    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_FILE,
            _CRITIQUE_APPROVE,
            _TOOLCALL_EXECUTE_RM,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ]
    )
    config_path = _write_run_config(tmp_path)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "read src/greet.py, then remove somefile",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        input="n\n",
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Delete: rm somefile" in result.stdout
    assert "reason: TASK_COMPLETE" in result.stdout
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "keep me\n"


def test_dod_undo_reverts_a_task(
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given a completed `kestrel run` that edited one file, when
    `kestrel undo --task-id <id> --repo PATH` runs against the task id
    the completed run printed, then the fixture repo's file is restored
    to its exact pre-task content -- undo working end to end through
    the packaged console entry point.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "greet.py").write_text(_GREET_STUB, encoding="utf-8")

    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_EDIT_GREET,
            _CRITIQUE_APPROVE,
            _DONE_CASSETTE,
            _CRITIQUE_APPROVE,
        ]
    )
    config_path = _write_run_config(tmp_path)

    run_result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "implement the missing greet function in greet.py",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "def greet" in (repo_dir / "greet.py").read_text(encoding="utf-8")

    match = _TASK_ID_RE.search(run_result.stdout)
    assert match is not None, run_result.stdout
    task_id = match["task_id"]

    undo_result = subprocess.run(
        [kestrel_executable, "undo", "--task-id", task_id, "--repo", str(repo_dir)],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert undo_result.returncode == 0, undo_result.stderr
    assert (repo_dir / "greet.py").read_text(encoding="utf-8") == _GREET_STUB


@pytest.mark.redteam
@pytest.mark.parametrize("case", _CORPUS_CASES, ids=lambda case: case.id)
def test_dod_injection_corpus_fails_closed(
    case: InjectionCase,
    tmp_path: Path,
    mock_openai_server: Callable[..., str],
    kestrel_executable: str,
) -> None:
    """Given every case in the injection corpus, written as the content
    of a file the fixture repo's `read_file` call returns, and a
    scripted second-turn reply that echoes the file's content back as
    if quoting it, when `kestrel run` drives the real loop through both
    turns, then none of the case's `forbidden_markers` appears more than
    the one time the real, legitimate frame around the tool result
    itself produces -- proving `frame_untrusted`'s escaping survives a
    real turn's history being serialized into an actual outgoing HTTP
    request body, not merely as a pure string transform.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "payload.txt").write_text(case.payload, encoding="utf-8")

    echo_cassette = _write_echo_cassette(tmp_path / "echo.sse", payload=case.payload)
    captured: list[bytes] = []
    base_url = mock_openai_server(
        cassette_sequence=[
            _TOOLCALL_READ_PAYLOAD,
            _CRITIQUE_APPROVE,
            echo_cassette,
            _CRITIQUE_APPROVE,
        ],
        capture=captured,
    )
    config_path = _write_run_config(tmp_path)

    result = subprocess.run(
        [
            kestrel_executable,
            "run",
            "read payload.txt and summarize its contents",
            "--repo",
            str(repo_dir),
            "--config",
            str(config_path),
            "--no-require-verification",
        ],
        capture_output=True,
        encoding="utf-8",
        env=_run_env(base_url),
        cwd=repo_dir,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(captured) == 4

    # captured[0]/[1] are the first turn's own think call and its
    # self-critique check; captured[2] is the second turn's think call
    # -- the one whose request history actually carries the first turn's
    # own framed tool result.
    second_turn_request = captured[2].decode("utf-8")
    for marker in case.forbidden_markers:
        assert second_turn_request.count(marker) == 1, (
            case.id,
            marker,
            second_turn_request,
        )
