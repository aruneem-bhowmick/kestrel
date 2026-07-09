"""Command-line entry point for the ``kestrel`` console script."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from kestrel import __version__
from kestrel.config import ConfigError, load_config
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.registry.loader import load_registry
from kestrel.registry.model import RegistryError, UnknownModelError
from kestrel.repl import run_repl


def _build_parser() -> ArgumentParser:
    """Build the top-level argument parser.

    The parser accepts the full flag and subcommand surface up front, even
    though most paths do not yet have real behavior wired to them. Building
    the complete surface now means later functionality slots into an
    existing parser instead of requiring one written from scratch.
    """
    parser = ArgumentParser(
        prog="kestrel",
        description="Kestrel: a terminal coding assistant.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed kestrel version and exit.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a kestrel.toml configuration file.",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default=None,
        help="Model registry id to use as the active model.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor", help="Run environment and configuration diagnostics."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Subcommands: (none)=repl, doctor. Flags: --version,
    --config PATH, --model ID. ``doctor`` is not yet implemented and
    returns 2; every other path either prints the version or starts the
    REPL against the resolved configuration, registry, and starting model.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "doctor":
        print("kestrel doctor: not yet implemented", file=sys.stderr)
        return 2

    explicit_config = Path(args.config) if args.config else None
    try:
        config, _config_source = load_config(explicit_config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        registry = load_registry(config.paths.models_file)
    except RegistryError as exc:
        print(exc, file=sys.stderr)
        return 1

    model_id = args.model or config.general.default_model
    try:
        registry.get(model_id)
    except UnknownModelError as exc:
        print(exc, file=sys.stderr)
        return 1

    client = LiteLLMClient(registry)
    return run_repl(config, registry, client, model_id)


if __name__ == "__main__":
    sys.exit(main())
