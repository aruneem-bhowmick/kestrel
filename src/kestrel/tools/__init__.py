"""Tools a model can call: bounded, framed reads and searches, an
anchor-based file editor with a journaled undo trail, a sandboxed
command runner, and a verification runner for a repo's own configured
lint/build/test commands.

Every tool in this package returns its result already wrapped by
`kestrel.security.framing.frame_untrusted`, so nothing downstream needs
to remember to frame tool output itself. `kestrel.tools.registry` is
where all of them come together: `all_schemas()` for a provider call's
`tools=` argument, `schemas_for()` to narrow that list to a chosen
subset of tool names, and `dispatch()` to route one `ToolCallEvent` back
to its bound tool.
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
from kestrel.tools.registry import ToolResult, all_schemas, dispatch, schemas_for
from kestrel.tools.search import (
    SEARCH_SCHEMA,
    SearchArgs,
    SearchError,
    SearchHit,
    parse_search_args,
    search,
)
from kestrel.tools.verify import (
    VERIFY_SCHEMA,
    VerificationCommandResult,
    VerificationReport,
    VerifyArgs,
    VerifyError,
    parse_verify_args,
    verify,
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
    "ToolResult",
    "all_schemas",
    "schemas_for",
    "dispatch",
    "VERIFY_SCHEMA",
    "VerificationCommandResult",
    "VerificationReport",
    "VerifyArgs",
    "VerifyError",
    "parse_verify_args",
    "verify",
]
