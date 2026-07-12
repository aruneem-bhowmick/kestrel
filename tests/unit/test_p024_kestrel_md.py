"""Tests for the KESTREL.md project-memory loader: fenced
``kestrel-verify`` block extraction (both ``` and ~~~ fences), TOML
parsing into `VerifyCommands`, and the UTF-8/TOML/unexpected-key error
paths raised as `KestrelMdError`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from kestrel.kestrel_md import (
    KestrelMd,
    KestrelMdError,
    VerifyCommands,
    load_kestrel_md,
)

pytestmark = [pytest.mark.p024, pytest.mark.unit]

_GOLDEN_FILE = (
    Path(__file__).resolve().parent.parent / "golden" / "p024_verify_commands.golden"
)


def _write_kestrel_md(repo_root: Path, content: str) -> Path:
    """Write `content` as UTF-8 bytes to `repo_root / "KESTREL.md"`,
    creating `repo_root` as needed, and return the written path."""
    repo_root.mkdir(parents=True, exist_ok=True)
    path = repo_root / "KESTREL.md"
    path.write_bytes(content.encode("utf-8"))
    return path


@pytest.mark.sanity
def test_missing_file_returns_none(tmp_path: Path) -> None:
    """Given a repo root with no KESTREL.md, when loaded, then the
    result is None rather than an error -- project memory is optional."""
    assert load_kestrel_md(tmp_path) is None


@pytest.mark.sanity
def test_file_with_no_fenced_block_has_empty_verify_commands(
    tmp_path: Path,
) -> None:
    """Given a KESTREL.md with prose but no fenced block at all, when
    loaded, then raw_text is preserved verbatim and verify_commands is
    empty."""
    content = "# Conventions\n\nBe kind to the tests.\n"
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded == KestrelMd(raw_text=content, verify_commands=VerifyCommands())


@pytest.mark.sanity
def test_kestrel_verify_block_parses_lint_and_test_leaving_build_none(
    tmp_path: Path,
) -> None:
    """Given a ```kestrel-verify block naming lint and test but not
    build, when loaded, then both configured commands parse and build
    is None."""
    content = (
        "# Conventions\n\n"
        "```kestrel-verify\n"
        'lint = "ruff check ."\n'
        'test = "pytest -q"\n'
        "```\n"
    )
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded is not None
    assert loaded.verify_commands == VerifyCommands(
        lint="ruff check .", build=None, test="pytest -q"
    )


def test_tilde_fence_parses_identically_to_backtick_fence(tmp_path: Path) -> None:
    """Given a kestrel-verify block delimited with ~~~ instead of ```,
    when loaded, then it parses the same as the backtick form."""
    content = "~~~kestrel-verify\nbuild = \"make\"\n~~~\n"
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded is not None
    assert loaded.verify_commands == VerifyCommands(build="make")


def test_unrelated_fenced_block_is_ignored(tmp_path: Path) -> None:
    """Given a file with an unrelated ```python block and a real
    kestrel-verify block, when loaded, then only the tagged block is
    parsed -- the unrelated block's content never reaches
    VerifyCommands."""
    content = (
        "```python\n"
        "lint = 'not this one'\n"
        "```\n\n"
        "```kestrel-verify\n"
        'test = "pytest -q"\n'
        "```\n"
    )
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded is not None
    assert loaded.verify_commands == VerifyCommands(test="pytest -q")


def test_unexpected_key_in_verify_block_raises_naming_the_key(
    tmp_path: Path,
) -> None:
    """Given a kestrel-verify block naming a key outside
    lint/build/test, when loaded, then KestrelMdError names the
    offending key."""
    content = '```kestrel-verify\ndeploy = "echo nope"\n```\n'
    _write_kestrel_md(tmp_path, content)

    with pytest.raises(KestrelMdError, match="deploy"):
        load_kestrel_md(tmp_path)


def test_non_string_value_in_verify_block_raises_naming_the_key(
    tmp_path: Path,
) -> None:
    """Given a kestrel-verify block giving lint a non-string value,
    when loaded, then KestrelMdError names the offending key."""
    content = "```kestrel-verify\nlint = 123\n```\n"
    _write_kestrel_md(tmp_path, content)

    with pytest.raises(KestrelMdError, match="lint"):
        load_kestrel_md(tmp_path)


def test_invalid_toml_in_verify_block_raises(tmp_path: Path) -> None:
    """Given a kestrel-verify block that is not valid TOML, when
    loaded, then KestrelMdError is raised rather than a raw
    TOMLDecodeError escaping."""
    content = "```kestrel-verify\nlint = not valid toml\n```\n"
    _write_kestrel_md(tmp_path, content)

    with pytest.raises(KestrelMdError):
        load_kestrel_md(tmp_path)


def test_non_utf8_bytes_raise_naming_the_path(tmp_path: Path) -> None:
    """Given a KESTREL.md containing bytes that are not valid UTF-8,
    when loaded, then KestrelMdError names the path rather than a raw
    UnicodeDecodeError escaping."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "KESTREL.md"
    path.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(KestrelMdError, match=r"KESTREL\.md"):
        load_kestrel_md(tmp_path)


def test_raw_text_is_byte_identical_to_file_content(tmp_path: Path) -> None:
    """Given a file with irregular whitespace and blank lines, when
    loaded, then raw_text reproduces it exactly -- no reformatting."""
    content = "line one\n\n\n  indented\ttab\r\nline two\n"
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded is not None
    assert loaded.raw_text == content


def test_second_kestrel_verify_block_is_ignored(tmp_path: Path) -> None:
    """Given two kestrel-verify blocks in one file, when loaded, then
    only the first is parsed."""
    content = (
        "```kestrel-verify\n"
        'lint = "first"\n'
        "```\n\n"
        "```kestrel-verify\n"
        'lint = "second"\n'
        "```\n"
    )
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)

    assert loaded is not None
    assert loaded.verify_commands == VerifyCommands(lint="first")


def test_as_mapping_omits_unset_fields_and_orders_lint_build_test() -> None:
    """Given a VerifyCommands with only test and lint set, when
    rendered as a mapping, then it contains exactly those two entries
    in lint-then-test order, with build omitted."""
    commands = VerifyCommands(test="pytest -q", lint="ruff check .")

    assert list(commands.as_mapping().items()) == [
        ("lint", "ruff check ."),
        ("test", "pytest -q"),
    ]


@pytest.mark.regression
def test_kestrel_md_wire_format_matches_golden_snapshot(tmp_path: Path) -> None:
    """One canonical KestrelMd, with every verify command set, loaded
    and normalized to sorted JSON, matches a pinned snapshot
    byte-for-byte -- an accidental change to the dataclass shape shows
    up as a diff here instead of surfacing later as a silent behavior
    change."""
    content = (
        "# Project conventions\n\n"
        "Keep changes small and covered by tests.\n\n"
        "```kestrel-verify\n"
        'lint = "ruff check ."\n'
        'build = "true"\n'
        'test = "pytest -q"\n'
        "```\n"
    )
    _write_kestrel_md(tmp_path, content)

    loaded = load_kestrel_md(tmp_path)
    assert loaded is not None

    normalized = json.dumps(dataclasses.asdict(loaded), indent=2, sort_keys=True)
    assert normalized + "\n" == _GOLDEN_FILE.read_text()
