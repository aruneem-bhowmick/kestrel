"""Command-line entry point for the ``kestrel`` console script."""

from __future__ import annotations

import sys
from argparse import SUPPRESS, ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from kestrel import __version__
from kestrel.config import ConfigError, load_config
from kestrel.doctor import all_checks_passed, render_report, run_doctor
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
    doctor_parser = subparsers.add_parser(
        "doctor", help="Run environment and configuration diagnostics."
    )
    doctor_parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Also probe the default model's endpoint with a real, "
            "budget-capped completion."
        ),
    )
    doctor_parser.add_argument(
        "--config",
        metavar="PATH",
        default=SUPPRESS,
        help="Path to a kestrel.toml configuration file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Subcommands: (none)=repl, doctor [--live]. Flags:
    --version, --config PATH, --model ID. ``doctor`` prints one aligned
    line per flight check and exits 0 unless any check FAILed; every
    other path either prints the version or starts the REPL against the
    resolved configuration, registry, and starting model.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "doctor":
        config_path = Path(args.config) if args.config else None
        results = run_doctor(config_path, live=args.live)
        print(render_report(results), end="")
        return 0 if all_checks_passed(results) else 1

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
