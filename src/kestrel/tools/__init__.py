"""Tools a model can call: bounded, framed reads and searches, a
sandboxed command runner, and (eventually) an edit capability.

Every tool in this package returns its result already wrapped by
`kestrel.security.framing.frame_untrusted`, so nothing downstream needs
to remember to frame tool output itself.
"""

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
]
