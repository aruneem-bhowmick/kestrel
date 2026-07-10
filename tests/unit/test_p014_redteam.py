"""Red-team proof that reading a hostile file through `read_file` still
leaves the returned content wrapped by the real frame markers -- a
hostile file's content cannot smuggle itself out of its own frame.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.security.corpus import load_corpus
from kestrel.tools.read_file import ReadFileArgs, read_file

pytestmark = [pytest.mark.p014, pytest.mark.unit, pytest.mark.redteam]

_CASE = next(
    case for case in load_corpus() if case.id == "readme_ignore_previous_instructions"
)


def test_hostile_file_content_still_carries_the_real_frame_markers(
    tmp_path: Path,
) -> None:
    """Given a file whose content is one of the injection corpus's
    hostile payloads, when read through `read_file`, then the result
    still starts with the real opening header naming this file, still
    ends with the real closing delimiter, and the hostile payload itself
    survives unchanged between them."""
    (tmp_path / "README.md").write_bytes(_CASE.payload.encode("utf-8"))

    framed = read_file(ReadFileArgs(path="README.md"), repo_root=tmp_path)

    assert framed.startswith("<<<UNTRUSTED:file:README.md>>>\n")
    assert framed.endswith("<<<END_UNTRUSTED>>>")
    assert _CASE.payload in framed
    for marker in _CASE.forbidden_markers:
        assert framed.count(marker) == 1
