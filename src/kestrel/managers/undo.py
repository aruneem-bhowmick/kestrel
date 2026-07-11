"""Journaled, reversible file edits.

`UndoManager` is Kestrel's first piece of on-disk runtime state: every
file mutation a caller tells it about is appended to a journal under
the target repo before anything else touches it, so a mutation can
always be traced and reversed even across separate process runs.
`kestrel.tools.edit_file` is its first real caller, recording through
it on every write rather than mutating a file and hoping nothing goes
wrong.

The journal is JSONL (JSON Lines): one `UndoEntry` per line, appended
in the order it was recorded, never rewritten or truncated in place.
Every line round-trips through `json.dumps`/`json.loads` on its own,
so a reader can process the file incrementally and a writer never has
to rewrite anything it already wrote. This is also the format
Kestrel's session log will use once it exists, so this module
establishes the on-disk shape rather than inventing a one-off.

Reverting never deletes or rewrites a journal entry -- the journal is
append-only for its entire lifetime. Instead, `revert_last`,
`revert_turn`, and `revert_task` each record a *compensating* entry
for every mutation they undo: a new `UndoEntry` for the same path with
`before` and `after` swapped relative to the one being reverted. This
buys two things a truncate-on-revert design would not. First, the
journal is always a complete, honest history of every content
transition a path went through, revert included -- nothing is erased.
Second, calling a revert method again immediately afterward is always
well-defined: it targets the compensating entry just appended and
restores the pre-revert content, rather than raising because the
journal it would have rewound out from under it is now shorter (or
empty) than the caller expects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class UndoEntry:
    """One journaled file mutation.

    Attributes:
        turn_id: The agent-loop turn this mutation happened during.
        task_id: The task this mutation belongs to.
        path: Repo-relative path of the file that changed.
        before: The file's content immediately before this mutation,
            or `None` if the file did not exist yet (this mutation
            created it).
        after: The file's content immediately after this mutation, or
            `None` if this mutation deleted the file.
    """

    turn_id: int
    task_id: str
    path: str
    before: str | None
    after: str | None


class UndoConflictError(Exception):
    """A revert's target file does not currently hold the content the
    journal recorded as its own `after` value.

    Raised instead of silently overwriting whatever is actually on
    disk -- something changed the file out of band (or a later,
    unreverted mutation from a different turn or task still owns its
    current content) since the entry being reverted was recorded.
    """

    def __init__(
        self, message: str, *, reverted: list[UndoEntry] | None = None
    ) -> None:
        super().__init__(message)
        self.reverted = reverted if reverted is not None else []


def _entry_to_json(entry: UndoEntry) -> str:
    """Serialize `entry` to a single JSONL line, with no trailing
    newline -- the field order (`turn_id`, `task_id`, `path`, `before`,
    `after`) is the journal's stable, tested wire format."""
    return json.dumps(
        {
            "turn_id": entry.turn_id,
            "task_id": entry.task_id,
            "path": entry.path,
            "before": entry.before,
            "after": entry.after,
        },
        ensure_ascii=False,
    )


def _entry_from_json(line: str) -> UndoEntry:
    """Parse one JSONL line back into an `UndoEntry`."""
    data: dict[str, Any] = json.loads(line)
    return UndoEntry(
        turn_id=data["turn_id"],
        task_id=data["task_id"],
        path=data["path"],
        before=data["before"],
        after=data["after"],
    )


class UndoManager:
    """Journals file mutations under a repo and reverts them on request.

    Every mutation this manager is told about via `record` is appended
    to `journal_path` (JSONL, one `UndoEntry` per line) and kept in an
    in-memory list for this instance's lifetime. Reverting reads and
    writes real files under `repo_root`; nothing here re-injects file
    content anywhere else, so it carries none of the untrusted-data
    framing concerns a model-facing tool would.
    """

    def __init__(self, *, repo_root: Path, journal_path: Path | None = None) -> None:
        """Point this manager at `repo_root`, loading whatever entries
        already exist at `journal_path` (default: `repo_root /
        ".kestrel" / "undo.jsonl"`) into memory. Neither the journal's
        parent directory nor the file itself is created here -- that
        happens lazily, on the first call to `record`, so constructing
        a manager is never itself a filesystem-mutating action."""
        self._repo_root = repo_root
        self._journal_path = (
            journal_path
            if journal_path is not None
            else repo_root / ".kestrel" / "undo.jsonl"
        )
        self._entries: list[UndoEntry] = self._load_existing_entries()

    @property
    def journal_path(self) -> Path:
        """The journal file this manager reads from and appends to."""
        return self._journal_path

    def _load_existing_entries(self) -> list[UndoEntry]:
        """Read every entry already on disk at `journal_path`, in
        append order; returns an empty list when the file does not
        exist yet."""
        if not self._journal_path.exists():
            return []
        text = self._journal_path.read_bytes().decode("utf-8")
        lines = [line for line in text.splitlines() if line]
        entries: list[UndoEntry] = []
        for i, line in enumerate(lines):
            try:
                entries.append(_entry_from_json(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if i == len(lines) - 1:
                    break
                raise exc
        return entries

    def record(self, entry: UndoEntry) -> None:
        """Append `entry` to the journal, creating its parent
        directory and the file itself if this is the first write, and
        add it to this instance's in-memory list.

        The caller is responsible for `entry.before` matching what was
        actually on disk immediately before its own mutation --
        `record` trusts it unconditionally. Conflict detection only
        ever happens at revert time, against `after`.

        Appends in binary mode with an explicit `\\n`, never text
        mode's platform-dependent newline translation -- the journal
        is a byte-format contract (a pinned golden file tests it), so
        its line endings must be identical on every platform, not
        whatever the local OS would otherwise substitute.
        """
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{_entry_to_json(entry)}\n".encode("utf-8")
        with self._journal_path.open("ab") as handle:
            handle.write(line)
        self._entries.append(entry)

    def _resolve_within_repo_root(self, path: str) -> Path:
        """Resolve `path` against `_repo_root`, following any symlink, and
        raise ValueError if the resolved location falls outside `_repo_root`
        or if `path` is absolute."""
        if path.startswith(("/", "\\")) or Path(path).is_absolute():
            raise ValueError(f"{path}: absolute path not allowed")
        repo_root = self._repo_root.resolve()
        candidate = self._repo_root / path
        if candidate.is_symlink():
            resolved = (candidate.parent / os.readlink(candidate)).resolve(strict=False)
        else:
            resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            raise ValueError(f"{path}: escapes the repository root")
        return resolved

    def _current_content(self, path: str) -> str | None:
        """The current on-disk content at `path` (resolved under
        `repo_root`), or `None` if no file exists there.

        Reads raw bytes and decodes them explicitly rather than using
        text-mode reading, which would otherwise translate `\\r\\n`
        sequences on some platforms -- a translation that would make
        the equality check against `entry.after` spuriously fail (or
        spuriously pass) depending on the OS this happens to run on.
        """
        resolved = self._resolve_within_repo_root(path)
        if not resolved.exists():
            return None
        return resolved.read_bytes().decode("utf-8")

    def _restore(self, entry: UndoEntry) -> None:
        """Write `entry.before` back to `entry.path` (or delete the
        file, if `entry.before is None`), after confirming the file's
        current content matches `entry.after` exactly.

        Raises:
            UndoConflictError: the current content does not match
                `entry.after` -- naming `entry.path`, and leaving the
                file untouched.
        """
        resolved = self._resolve_within_repo_root(entry.path)
        current = self._current_content(entry.path)
        if current != entry.after:
            raise UndoConflictError(
                f"{entry.path}: current content does not match the journal's "
                "recorded state; refusing to revert over an out-of-band change"
            )
        if entry.before is None:
            resolved.unlink(missing_ok=True)
        else:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(entry.before.encode("utf-8"))

    def _revert_one(self, entry: UndoEntry) -> UndoEntry:
        """Restore `entry`'s pre-mutation content and record the
        compensating entry for the revert itself; returns `entry`
        unchanged, naming what was undone."""
        self._restore(entry)
        self.record(
            UndoEntry(
                turn_id=entry.turn_id,
                task_id=entry.task_id,
                path=entry.path,
                before=entry.after,
                after=entry.before,
            )
        )
        return entry

    def revert_last(self) -> UndoEntry:
        """Undo the single most recent entry across the whole journal
        (including a previous revert's own compensating entry, so
        calling this twice in a row toggles rather than raises) and
        return it.

        Raises:
            IndexError: the journal holds no entries at all.
            UndoConflictError: see `_restore`.
        """
        if not self._entries:
            raise IndexError("undo journal is empty")
        return self._revert_one(self._entries[-1])

    def revert_turn(self, turn_id: int) -> list[UndoEntry]:
        """Undo every entry recorded with this `turn_id`, most-recent-
        first, and return them in the order they were reverted.

        Entries from other turns interleaved in the journal are left
        untouched. An out-of-band (or not-yet-reverted, later) change
        to a path this turn also touched surfaces as `UndoConflictError`
        on whichever entry hits it, exactly as `revert_last` would.
        """
        matching = [entry for entry in self._entries if entry.turn_id == turn_id]
        reverted: list[UndoEntry] = []
        try:
            for entry in reversed(matching):
                reverted.append(self._revert_one(entry))
        except UndoConflictError as exc:
            raise UndoConflictError(str(exc), reverted=reverted) from exc
        return reverted

    def revert_task(self, task_id: str) -> list[UndoEntry]:
        """Undo every entry recorded with this `task_id`, most-recent-
        first, and return them in the order they were reverted.

        Mirrors `revert_turn` at task granularity.
        """
        matching = [entry for entry in self._entries if entry.task_id == task_id]
        reverted: list[UndoEntry] = []
        try:
            for entry in reversed(matching):
                reverted.append(self._revert_one(entry))
        except UndoConflictError as exc:
            raise UndoConflictError(str(exc), reverted=reverted) from exc
        return reverted
