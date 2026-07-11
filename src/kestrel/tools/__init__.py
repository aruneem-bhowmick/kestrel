"""Tools a model can call: bounded, framed reads and searches, an
anchor-based file editor with a journaled undo trail, and a sandboxed
command runner.

Every tool in this package returns its result already wrapped by
`kestrel.security.framing.frame_untrusted`, so nothing downstream needs
to remember to frame tool output itself.
"""

from kestrel.tools.edit_file import (
    EDIT_FILE_SCHEMA,
    EditFileArgs,
    EditFileError,
    edit_file,
    parse_edit_file_args,
)
from kestrel.tools.execute import (
    EXECUTE_SCHEMA,
    ExecuteArgs,
    ExecuteError,
    execute,
    parse_execute_args,
)
from kestrel.tools.read_file import (
    READ_FILE_SCHEMA,
    ReadFileArgs,
    ReadFileError,
    parse_read_file_args,
    read_file,
)
from kestrel.tools.search import (
    SEARCH_SCHEMA,
    SearchArgs,
    SearchError,
    SearchHit,
    parse_search_args,
    search,
)

__all__ = [
    "READ_FILE_SCHEMA",
    "ReadFileArgs",
    "ReadFileError",
    "parse_read_file_args",
    "read_file",
    "SEARCH_SCHEMA",
    "SearchArgs",
    "SearchError",
    "SearchHit",
    "parse_search_args",
    "search",
    "EXECUTE_SCHEMA",
    "ExecuteArgs",
    "ExecuteError",
    "execute",
    "parse_execute_args",
    "EDIT_FILE_SCHEMA",
    "EditFileArgs",
    "EditFileError",
    "edit_file",
    "parse_edit_file_args",
]
