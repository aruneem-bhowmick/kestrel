"""Unit tests for the `kestrel unload-aux` CLI subcommand: argparse
parsing of its lone `--config` flag, `_run_unload_aux_command`'s own
success/failure handling of `unload_aux_model`, and `main`'s dispatch
into that branch -- all hermetic, with `unload_aux_model` itself
monkeypatched rather than reaching a real Ollama server (see
`tests/system/test_p065_unload_aux_subprocess.py` for that).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.cli as cli_module
from kestrel.cli import _build_parser, _run_unload_aux_command, main
from kestrel.config import GeneralConfig, KestrelConfig
from kestrel.managers.resource_guard import UnloadAuxError
from kestrel.registry.model import ModelEntry, Registry

pytestmark = [pytest.mark.p065, pytest.mark.unit, pytest.mark.sanity]


def _registry_and_config() -> tuple[Registry, KestrelConfig]:
    """Build a two-entry registry -- an OpenRouter chat entry left as the
    configured default model, and a `"local"`-tagged Ollama entry -- so a
    test asserting on the resolved entry proves `_run_unload_aux_command`
    actually routed through the `"embed"` task class's own tag, rather
    than merely falling back to `config.general.default_model`."""
    chat_entry = ModelEntry(
        id="glm-5.2",
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200000,
        max_output=16384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    embed_entry = ModelEntry(
        id="nomic-embed-text",
        backend="ollama",
        provider_model="nomic-embed-text",
        endpoint="http://ollama.invalid:11434",
        context_window=8192,
        max_output=1,
        usd_per_mtok_input=Decimal("0"),
        usd_per_mtok_output=Decimal("0"),
        usd_per_mtok_cached=Decimal("0"),
        supports_tools=False,
        supports_cache=False,
        tags=frozenset({"local"}),
    )
    registry = Registry(
        models={"glm-5.2": chat_entry, "nomic-embed-text": embed_entry}, source=None
    )
    config = KestrelConfig(general=GeneralConfig(default_model="glm-5.2"))
    return registry, config


# --- argparse: the subcommand and its lone flag -------------------------


def test_unload_aux_parses_with_no_flags() -> None:
    """Given a bare `unload-aux` invocation, when parsed, then the
    command is recognized and `--config` is left unset (`SUPPRESS`)
    rather than defaulting to some placeholder value."""
    parser = _build_parser()
    args = parser.parse_args(["unload-aux"])
    assert args.command == "unload-aux"
    assert not hasattr(args, "config") or args.config is None


def test_unload_aux_config_flag_works_before_and_after_the_subcommand() -> None:
    """Given `--config` in either position around `unload-aux`, when
    parsed, then both resolve to the same value -- matching `run`'s own
    dual-position `--config` contract."""
    parser = _build_parser()
    before = parser.parse_args(["--config", "/tmp/x.toml", "unload-aux"])
    after = parser.parse_args(["unload-aux", "--config", "/tmp/x.toml"])
    assert before.config == "/tmp/x.toml"
    assert after.config == "/tmp/x.toml"


# --- _run_unload_aux_command: success/failure handling -------------------


async def _fake_unload_success(*, entry: ModelEntry) -> None:
    """Stand in for `unload_aux_model`, succeeding without ever reaching
    a real Ollama server."""


async def _fake_unload_failure(*, entry: ModelEntry) -> None:
    """Stand in for `unload_aux_model`, always failing exactly like a
    real Ollama server refusing the unload request would."""
    raise UnloadAuxError(
        f"could not unload {entry.id!r} from Ollama at {entry.endpoint!r}: boom"
    )


def test_run_unload_aux_command_prints_confirmation_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a fake `unload_aux_model` that succeeds, when
    `_run_unload_aux_command` runs, then it resolves the `"embed"`-tagged
    registry entry (not the configured default model), prints
    `"unloaded {id}"`, and returns `0`."""
    registry, config = _registry_and_config()
    monkeypatch.setattr(cli_module, "unload_aux_model", _fake_unload_success)

    exit_code = _run_unload_aux_command(config, registry)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "unloaded nomic-embed-text\n"
    assert captured.err == ""


def test_run_unload_aux_command_reports_the_error_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a fake `unload_aux_model` that raises `UnloadAuxError`, when
    `_run_unload_aux_command` runs, then it prints the error's own
    message to stderr, prints nothing to stdout, and returns `1`."""
    registry, config = _registry_and_config()
    monkeypatch.setattr(cli_module, "unload_aux_model", _fake_unload_failure)

    exit_code = _run_unload_aux_command(config, registry)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "nomic-embed-text" in captured.err
    assert "boom" in captured.err


# --- main(): dispatch, before config/registry ever try to resolve a model ---


def test_main_unload_aux_missing_config_exits_one_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given `--config` names a file that does not exist, when
    `unload-aux` executes through `main`, then it exits 1 with a
    readable message instead of a real `unload_aux_model` call ever
    being attempted."""
    missing = tmp_path / "missing.toml"

    exit_code = main(["unload-aux", "--config", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not found" in captured.err


def test_main_unload_aux_dispatches_to_the_new_branch_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given a real `kestrel.toml` naming a `"local"`-tagged Ollama
    entry, when `unload-aux` executes through `main` with
    `unload_aux_model` faked out, then it loads config and registry from
    disk (never touching `_resolve_startup`, which would otherwise
    demand a `--repo`/KESTREL.md this subcommand has no use for) and
    exits 0 having printed the confirmation line."""
    models_toml = tmp_path / "models.toml"
    models_toml.write_text(
        """\
[[models]]
id = "nomic-embed-text"
backend = "ollama"
provider_model = "nomic-embed-text"
endpoint = "http://ollama.invalid:11434"
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
    kestrel_toml = tmp_path / "kestrel.toml"
    kestrel_toml.write_text(
        f"""\
[general]
default_model = "nomic-embed-text"

[paths]
models_file = "{models_toml.as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "unload_aux_model", _fake_unload_success)

    exit_code = main(["unload-aux", "--config", str(kestrel_toml)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "unloaded nomic-embed-text\n"
