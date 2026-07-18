"""Unit tests for `kestrel run`'s `--mode` flag: argparse accepts
`plan`/`fast`/omitted (defaulting to `fast`), and any other value is
rejected via argparse's own error path -- hermetic and network-free,
mirroring `tests/unit/test_p033_cli.py`'s own argparse-wiring split.
"""

from __future__ import annotations

import pytest

from kestrel.cli import _build_parser

pytestmark = [pytest.mark.p049, pytest.mark.unit, pytest.mark.sanity]


def test_run_mode_defaults_to_fast() -> None:
    """Given `run` with no `--mode` flag, when parsed, then `args.mode`
    defaults to `"fast"` -- every existing `kestrel run` invocation
    behaves exactly as it did before `--mode` existed."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo"])
    assert args.mode == "fast"


def test_run_mode_accepts_plan() -> None:
    """Given `--mode plan`, when parsed, then `args.mode` carries it."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo", "--mode", "plan"])
    assert args.mode == "plan"


def test_run_mode_accepts_fast_explicitly() -> None:
    """Given `--mode fast` typed explicitly, when parsed, then
    `args.mode` is `"fast"`, identical to the flag's own default."""
    parser = _build_parser()
    args = parser.parse_args(["run", "task", "--repo", "/tmp/repo", "--mode", "fast"])
    assert args.mode == "fast"


def test_run_mode_rejects_an_unknown_value() -> None:
    """Given `--mode` set to a value outside `{"plan", "fast"}`, when
    parsed, then argparse rejects it via its own error path (a
    `SystemExit`) before any application code ever sees it."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "task", "--repo", "/tmp/repo", "--mode", "turbo"])
