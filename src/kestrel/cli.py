"""Command-line entry point for the ``kestrel`` console script."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Sequence

from kestrel import __version__


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
    --config PATH, --model ID. Only --version is implemented; every other
    path prints a not-yet-implemented notice and returns 2.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    print("kestrel: not yet implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
