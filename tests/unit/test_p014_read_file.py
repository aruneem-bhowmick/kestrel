"""Tests for the `read_file` tool: path resolution (including the
symlink and `..`-escape guards), the binary guard, line-range slicing,
byte-cap truncation, and argument parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.tools.read_file import (
    ReadFileArgs,
    ReadFileError,
    parse_read_file_args,
    read_file,
)

pytestmark = [pytest.mark.p014, pytest.mark.unit]


def _write(root: Path, relative: str, content: str) -> None:
    """Write `content` as UTF-8 bytes to `relative` under `root`, creating
    parent directories as needed. Writing bytes directly (rather than
    through text-mode newline translation) keeps line endings exactly as
    given, regardless of platform."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


@pytest.mark.sanity
def test_reads_a_small_file_whole(tmp_path: Path) -> None:
    """Given a small text file, when read with no line range, then the
    framed output contains the exact content between the real opening
    and closing frame markers."""
    _write(tmp_path, "greet.py", "print('hello')\n")

    framed = read_file(ReadFileArgs(path="greet.py"), repo_root=tmp_path)

    assert framed.startswith("<<<UNTRUSTED:file:greet.py>>>\n")
    assert "print('hello')\n" in framed
    assert framed.endswith("<<<END_UNTRUSTED>>>")


@pytest.mark.sanity
def test_start_and_end_line_slice_is_1_indexed_inclusive(tmp_path: Path) -> None:
    """Given a multi-line file, when read with `start_line`/`end_line`,
    then the returned slice is exactly the 1-indexed, inclusive line
    range -- neither more nor fewer lines."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\nfour\nfive\n")

    framed = read_file(
        ReadFileArgs(path="lines.txt", start_line=2, end_line=4), repo_root=tmp_path
    )

    assert "two\nthree\nfour\n" in framed
    assert "one" not in framed
    assert "five" not in framed


@pytest.mark.sanity
def test_path_escaping_repo_root_via_dotdot_raises(tmp_path: Path) -> None:
    """Given a path that climbs above `repo_root` with `..`, when read,
    then `ReadFileError` is raised rather than the file outside the repo
    being returned."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write(tmp_path, "secret.txt", "outside the repo")

    with pytest.raises(ReadFileError, match="escapes the repository root"):
        read_file(ReadFileArgs(path="../secret.txt"), repo_root=repo_root)


def test_path_escaping_repo_root_via_symlink_raises(tmp_path: Path) -> None:
    """Given a symlink inside `repo_root` pointing at a file outside it,
    when read, then `ReadFileError` is raised -- resolving the symlink
    before the containment check must not let a request escape it."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "secret.txt"
    _write(tmp_path, "secret.txt", "outside the repo")
    link = repo_root / "link.txt"

    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable in this environment: {exc}")

    with pytest.raises(ReadFileError, match="escapes the repository root"):
        read_file(ReadFileArgs(path="link.txt"), repo_root=repo_root)


@pytest.mark.sanity
def test_nonexistent_path_raises_naming_the_path(tmp_path: Path) -> None:
    """Given a path with no file on disk, when read, then `ReadFileError`
    names that path rather than a generic message."""
    with pytest.raises(ReadFileError, match="missing.py"):
        read_file(ReadFileArgs(path="missing.py"), repo_root=tmp_path)


@pytest.mark.sanity
def test_directory_path_raises(tmp_path: Path) -> None:
    """Given a path naming a directory, when read, then `ReadFileError`
    is raised instead of an `IsADirectoryError` escaping to the caller."""
    (tmp_path / "a_dir").mkdir()

    with pytest.raises(ReadFileError, match="directory"):
        read_file(ReadFileArgs(path="a_dir"), repo_root=tmp_path)


@pytest.mark.sanity
def test_binary_content_raises_naming_the_binary_guard(tmp_path: Path) -> None:
    """Given a file whose content begins with a PNG file signature, when
    read, then `ReadFileError` names the binary guard rather than a raw
    `UnicodeDecodeError` escaping to the caller."""
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    (tmp_path / "image.png").write_bytes(png_header)

    with pytest.raises(ReadFileError, match="binary guard"):
        read_file(ReadFileArgs(path="image.png"), repo_root=tmp_path)


def test_os_level_read_failure_raises_read_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a file that exists and passes the directory check but fails
    to read at the OS level (e.g. a permissions error, or the file
    disappearing between the check and the read), when read, then
    `ReadFileError` names the path instead of a raw `OSError` escaping to
    the caller. `read_bytes` calls `open` internally, so patching `open`
    covers both the whole-file and bounded-prefix read paths; the patch
    only targets the file under test, so anything else opened during the
    test (e.g. coverage instrumentation reading source files) is
    unaffected."""
    _write(tmp_path, "flaky.txt", "content\n")
    original_open = Path.open

    def _raise_os_error(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "flaky.txt":
            raise OSError("simulated read failure")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raise_os_error)

    with pytest.raises(ReadFileError, match="flaky.txt"):
        read_file(ReadFileArgs(path="flaky.txt"), repo_root=tmp_path)


def test_content_over_cap_truncates_with_a_note(tmp_path: Path) -> None:
    """Given a file larger than the 64 KiB return cap, when read with no
    line range, then the framed output is truncated and carries a
    trailing note naming how many bytes were cut."""
    oversized = "x" * (70 * 1024)
    _write(tmp_path, "big.txt", oversized)

    framed = read_file(ReadFileArgs(path="big.txt"), repo_root=tmp_path)

    assert "truncated" in framed
    assert len(framed) < len(oversized)
    assert framed.endswith("<<<END_UNTRUSTED>>>")


def test_truncation_boundary_splitting_a_multibyte_character_is_handled_cleanly(
    tmp_path: Path,
) -> None:
    """Given a file whose size exceeds the 64 KiB cap by just enough that
    the cut lands in the middle of a multi-byte UTF-8 character, when
    read, then the incomplete character is dropped cleanly instead of
    raising the binary guard, and the truncation note still reports the
    correct byte count."""
    prefix = "a" * 65535
    content = prefix + "€" + "tail beyond the cap"
    _write(tmp_path, "boundary.txt", content)

    framed = read_file(ReadFileArgs(path="boundary.txt"), repo_root=tmp_path)

    assert prefix in framed
    assert "€" not in framed
    assert "truncated" in framed
    assert framed.endswith("<<<END_UNTRUSTED>>>")


def test_invalid_byte_well_before_the_truncation_boundary_still_raises(
    tmp_path: Path,
) -> None:
    """Given a file larger than the 64 KiB cap whose very first byte is
    invalid UTF-8, when read, then `ReadFileError` still names the
    binary guard -- the truncation boundary's tolerance only forgives an
    incomplete character exactly at the cut, not a genuinely invalid
    byte anywhere else in the truncated prefix."""
    content = b"\xff" + (b"a" * (70 * 1024))
    (tmp_path / "invalid_then_big.bin").write_bytes(content)

    with pytest.raises(ReadFileError, match="binary guard"):
        read_file(ReadFileArgs(path="invalid_then_big.bin"), repo_root=tmp_path)


def test_start_line_after_end_line_raises(tmp_path: Path) -> None:
    """Given `start_line` greater than `end_line`, when read, then
    `ReadFileError` is raised rather than an empty string being
    returned."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\n")

    with pytest.raises(ReadFileError, match="start_line"):
        read_file(
            ReadFileArgs(path="lines.txt", start_line=3, end_line=1),
            repo_root=tmp_path,
        )


def test_end_line_past_eof_clamps_to_the_last_line(tmp_path: Path) -> None:
    """Given `end_line` beyond the file's last line, when read, then the
    slice clamps to the last real line instead of raising."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\n")

    framed = read_file(
        ReadFileArgs(path="lines.txt", start_line=2, end_line=1000),
        repo_root=tmp_path,
    )

    assert "two\nthree\n" in framed
    assert "one" not in framed


def test_start_line_past_eof_raises(tmp_path: Path) -> None:
    """Given `start_line` beyond the file's last line, when read, then
    `ReadFileError` is raised rather than an empty slice being
    returned."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\n")

    with pytest.raises(ReadFileError, match="start_line"):
        read_file(ReadFileArgs(path="lines.txt", start_line=100), repo_root=tmp_path)


def test_only_start_line_given_reads_through_eof(tmp_path: Path) -> None:
    """Given only `start_line`, when read, then the slice runs from
    `start_line` through the file's last line."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\n")

    framed = read_file(ReadFileArgs(path="lines.txt", start_line=2), repo_root=tmp_path)

    assert "two\nthree\n" in framed
    assert "one" not in framed


def test_only_end_line_given_reads_from_the_beginning(tmp_path: Path) -> None:
    """Given only `end_line`, when read, then the slice runs from the
    file's first line through `end_line`."""
    _write(tmp_path, "lines.txt", "one\ntwo\nthree\n")

    framed = read_file(ReadFileArgs(path="lines.txt", end_line=2), repo_root=tmp_path)

    assert "one\ntwo\n" in framed
    assert "three" not in framed


def test_malformed_json_raises_via_parse_read_file_args() -> None:
    """Given a syntactically invalid JSON string, when parsed, then
    `ReadFileError` is raised instead of a raw `json.JSONDecodeError`
    escaping to the caller."""
    with pytest.raises(ReadFileError, match="invalid JSON"):
        parse_read_file_args("{not json")


def test_arguments_json_not_an_object_raises() -> None:
    """Given valid JSON that is not an object, when parsed, then
    `ReadFileError` names the shape mismatch."""
    with pytest.raises(ReadFileError, match="expected a JSON object"):
        parse_read_file_args("[1, 2, 3]")


def test_missing_path_field_raises_via_parse_read_file_args() -> None:
    """Given arguments with no `path` field, when parsed, then
    `ReadFileError` names the missing field."""
    with pytest.raises(ReadFileError, match="missing required field 'path'"):
        parse_read_file_args("{}")


def test_unexpected_extra_field_raises_via_parse_read_file_args() -> None:
    """Given arguments carrying a field the schema does not declare, when
    parsed, then `ReadFileError` names the offending field."""
    with pytest.raises(ReadFileError, match="unexpected field"):
        parse_read_file_args('{"path": "a.py", "recursive": true}')


def test_path_field_wrong_type_raises_via_parse_read_file_args() -> None:
    """Given a `path` field that is not a string, when parsed, then
    `ReadFileError` is raised naming the expected type."""
    with pytest.raises(ReadFileError, match="'path' must be a string"):
        parse_read_file_args('{"path": 123}')


@pytest.mark.parametrize("bad_value", ['"two"', "0", "-1", "true"])
def test_line_number_field_invalid_raises_via_parse_read_file_args(
    bad_value: str,
) -> None:
    """Given a `start_line` that is neither an integer nor a positive
    one -- including a JSON boolean, which is an `int` subclass in Python
    but never a valid line number -- when parsed, then `ReadFileError`
    names the offending field."""
    with pytest.raises(ReadFileError, match="'start_line' must be"):
        parse_read_file_args(f'{{"path": "a.py", "start_line": {bad_value}}}')


def test_parse_read_file_args_builds_the_expected_dataclass() -> None:
    """Given well-formed arguments carrying every field, when parsed,
    then the resulting `ReadFileArgs` carries each value exactly."""
    args = parse_read_file_args('{"path": "a.py", "start_line": 2, "end_line": 5}')

    assert args == ReadFileArgs(path="a.py", start_line=2, end_line=5)
