"""Tools a model can call: bounded, framed reads and (eventually) search,
edit, and execute capabilities.

Every tool in this package returns its result already wrapped by
`kestrel.security.framing.frame_untrusted`, so nothing downstream needs
to remember to frame tool output itself.
"""

from kestrel.tools.read_file import (
    READ_FILE_SCHEMA,
    ReadFileArgs,
    ReadFileError,
    parse_read_file_args,
    read_file,
)

__all__ = [
    "READ_FILE_SCHEMA",
    "ReadFileArgs",
    "ReadFileError",
    "parse_read_file_args",
    "read_file",
]
