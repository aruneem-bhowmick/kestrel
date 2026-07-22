"""A per-repo (or global) `sqlite-vec`-backed knowledge-base store.

`KnowledgeStore` is this codebase's first user of both SQLite and a
vector index. `kestrel.managers.undo.UndoManager` and
`kestrel.managers.session.SessionManager` are its closest structural
precedent -- a `.kestrel/`-scoped store, lazily created on first write,
read back into a typed value on open -- but both of those are
append-only JSONL, not a query engine, so this module owns its own
on-disk shape (two SQLite tables) rather than reusing either one.

A note's text and its embedding are split across two tables sharing one
rowid: `notes` (an ordinary table holding `text`/`repo`/`tags`/
`source_task`/`timestamp`) and `note_embeddings` (a `vec0` virtual table,
provided by the `sqlite-vec` loadable extension, holding only the vector
itself). `sqlite-vec`'s own public Python API is `sqlite_vec.load(conn)`
(registers the extension onto an already-`enable_load_extension(True)`-
flagged connection) and `sqlite_vec.serialize_float32(...)` (the raw
little-endian float32 byte layout its `vec0` table expects); this module
never touches `sqlite-vec`'s SQL surface beyond `CREATE VIRTUAL TABLE`,
`INSERT`, and a `MATCH`/`k` nearest-neighbor `SELECT`.

`add_note` always L2-normalizes an embedding before storing it, which is
what lets `search` turn a `vec0` nearest-neighbor query's own L2 distance
`d` into a cosine similarity via `1 - d**2 / 2` -- an identity that only
holds exactly when both vectors compared are unit length -- without a
second query to recompute anything. `search` normalizes its own query
vector the same way before running the KNN query, so the identity holds
regardless of whether a caller's embedding model happens to emit
unit-length vectors on its own.

This module deliberately does not decide *which* on-disk path (per-repo
or global) a caller should open, or how many notes it should return by
default -- a caller always names `db_path` and `top_k` explicitly. It
also does nothing with a real embedding model: every test exercising it
builds its own hand-picked vectors.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import platformdirs
import sqlite_vec

_KB_FILENAME: Final[str] = "kb.sqlite3"
_GLOBAL_APP_NAME: Final[str] = "kestrel"


@dataclass(frozen=True, slots=True)
class KnowledgeNote:
    """One knowledge-base note (KES-KBS-001's own field list).

    Attributes:
        id: The store-assigned primary key, or `None` before
            `KnowledgeStore.add_note` has persisted it.
        text: The note's own text, verbatim.
        embedding: `text`'s own embedding vector.
        repo: A stable identifier for the repo this note originated
            from (`str(repo_root.resolve())`, by convention -- this
            module does not itself resolve a path, callers always pass
            the identifier already resolved).
        tags: Free-form labels (D4 -- unrelated to
            `kestrel.registry.model.Tag`), e.g. `("testing", "jetson")`.
        source_task: The task id this note was proposed from.
        timestamp: `time.time()`-style UTC epoch seconds this note was
            added, matching `kestrel.managers.session.TurnRecord`'s own
            convention.
    """

    id: int | None
    text: str
    embedding: tuple[float, ...]
    repo: str
    tags: tuple[str, ...]
    source_task: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class ScoredNote:
    """One `KnowledgeStore.search` result.

    Attributes:
        note: The matched note.
        score: Cosine similarity in `[-1, 1]` between the query vector
            and `note.embedding` -- higher is more similar, matching
            this codebase's own "higher is better" ranking convention
            (`SearchHit` order, `cache_hit_ratio`'s own alert
            direction).
    """

    note: KnowledgeNote
    score: float


class KnowledgeStoreError(Exception):
    """A store operation could not complete: a dimension mismatch
    between a note's own embedding and the store's configured
    dimension, or a malformed on-disk database. `str(self)` names the
    remedy."""


def resolve_kb_path(repo_root: Path, *, global_: bool) -> Path:
    """`repo_root / ".kestrel" / "kb.sqlite3"` when `global_` is
    `False`; `<platformdirs.user_data_dir("kestrel")> / "kb.sqlite3"`
    otherwise -- mirrors `kestrel.config._user_config_path`'s own
    `platformdirs`-based per-user path convention (D6), applied to
    `user_data_dir` instead of `user_config_dir` since a KB database is
    data, not configuration."""
    if global_:
        return Path(platformdirs.user_data_dir(_GLOBAL_APP_NAME)) / _KB_FILENAME
    return repo_root / ".kestrel" / _KB_FILENAME


def _l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize `vector` to unit length.

    Raises:
        KnowledgeStoreError: `vector` is the zero vector -- normalizing
            it would require dividing by zero.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        raise KnowledgeStoreError("cannot normalize a zero vector")
    return tuple(component / magnitude for component in vector)


def _deserialize_float32(blob: bytes) -> tuple[float, ...]:
    """The inverse of `sqlite_vec.serialize_float32`: unpack `blob`'s
    raw little-endian float32 bytes back into a plain tuple of Python
    floats, the shape `note_embeddings` itself always stores a vector
    as."""
    count = len(blob) // 4
    return struct.unpack(f"<{count}f", blob)


class KnowledgeStore:
    """One open `sqlite-vec`-backed database, at a fixed embedding
    dimension for its own lifetime."""

    def __init__(self, *, db_path: Path, embedding_dim: int) -> None:
        """Open (creating parent directories and the schema if this is
        a fresh file) a connection at `db_path`, load the `sqlite_vec`
        extension, and ensure the `notes` table and the `note_embeddings`
        `vec0` virtual table (dimension `embedding_dim`) both exist.
        Never deletes or migrates an existing file -- opening a
        database previously created at a different `embedding_dim`
        leaves its schema as-is; a dimension mismatch surfaces at
        `add_note`/`search` time instead (see below), not here.
        """
        self._embedding_dim = embedding_dim
        self._closed = False
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS notes ("
                "id INTEGER PRIMARY KEY, "
                "text TEXT NOT NULL, "
                "repo TEXT NOT NULL, "
                "tags TEXT NOT NULL, "
                "source_task TEXT NOT NULL, "
                "timestamp REAL NOT NULL)"
            )
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS note_embeddings USING "
                f"vec0(embedding float[{embedding_dim}])"
            )

    def add_note(self, note: KnowledgeNote) -> KnowledgeNote:
        """Insert `note` (`note.id` must be `None`) inside one
        transaction spanning both `notes` and `note_embeddings`, and
        return it with `id` populated from the row's own `rowid`.

        Raises:
            KnowledgeStoreError: `note.id` is not `None`; `len(note.
                embedding) != embedding_dim`; the underlying insert
                fails for any other reason (wrapped, never a raw
                `sqlite3.Error`).
        """
        if note.id is not None:
            raise KnowledgeStoreError("add_note: note.id must be None before insertion")
        if len(note.embedding) != self._embedding_dim:
            raise KnowledgeStoreError(
                f"add_note: embedding has {len(note.embedding)} dimensions, "
                f"store is configured for {self._embedding_dim}"
            )
        normalized = _l2_normalize(note.embedding)
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO notes (text, repo, tags, source_task, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        note.text,
                        note.repo,
                        json.dumps(list(note.tags)),
                        note.source_task,
                        note.timestamp,
                    ),
                )
                row_id = cursor.lastrowid
                self._conn.execute(
                    "INSERT INTO note_embeddings (rowid, embedding) VALUES (?, ?)",
                    (row_id, sqlite_vec.serialize_float32(list(normalized))),
                )
        except sqlite3.Error as exc:
            raise KnowledgeStoreError(f"add_note: insert failed: {exc}") from exc
        return KnowledgeNote(
            id=row_id,
            text=note.text,
            embedding=normalized,
            repo=note.repo,
            tags=note.tags,
            source_task=note.source_task,
            timestamp=note.timestamp,
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        tags: frozenset[str] | None = None,
    ) -> tuple[ScoredNote, ...]:
        """The `top_k` stored notes whose own embedding is nearest
        `query_embedding` by cosine similarity, most similar first,
        each optionally required to carry at least one tag in `tags`
        (an empty result, never an error, when nothing matches; `tags=
        None` applies no filter). Queries `note_embeddings` via
        `sqlite-vec`'s own `MATCH`/`k` KNN syntax for L2 distance, then
        converts each result's own distance `d` to a cosine similarity
        via `1 - d**2 / 2` -- exact for unit-normalized vectors, which
        `add_note` always stores (see below) -- rather than issuing a
        second query.

        `query_embedding` is itself L2-normalized before the KNN query
        runs, so the same identity holds regardless of whether a
        caller's own embedding model happens to emit unit-length
        vectors already. Every stored note is ranked (the KNN query's
        own `k` is the store's total note count, not `top_k`) before
        `tags` is applied, so a tag filter never causes a note that
        would otherwise have made the top `top_k` to be dropped in
        favor of a nearer but untagged one; `top_k` itself is applied
        last, once filtering is done.

        Raises:
            KnowledgeStoreError: `len(query_embedding) != embedding_dim`;
                `query_embedding` is the zero vector.
        """
        if len(query_embedding) != self._embedding_dim:
            raise KnowledgeStoreError(
                f"search: query embedding has {len(query_embedding)} dimensions, "
                f"store is configured for {self._embedding_dim}"
            )
        normalized_query = _l2_normalize(query_embedding)
        if top_k <= 0:
            return ()
        (total,) = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()
        if total == 0:
            return ()
        try:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM note_embeddings "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(list(normalized_query)), total),
            ).fetchall()
        except sqlite3.Error as exc:
            raise KnowledgeStoreError(f"search: query failed: {exc}") from exc

        scored: list[ScoredNote] = []
        for row_id, distance in rows:
            text, repo, tags_json, source_task, timestamp, embedding_blob = (
                self._conn.execute(
                    "SELECT n.text, n.repo, n.tags, n.source_task, n.timestamp, "
                    "ne.embedding "
                    "FROM notes AS n JOIN note_embeddings AS ne ON ne.rowid = n.id "
                    "WHERE n.id = ?",
                    (row_id,),
                ).fetchone()
            )
            note_tags = tuple(json.loads(tags_json))
            if tags is not None and not (set(note_tags) & tags):
                continue
            score = 1 - distance**2 / 2
            scored.append(
                ScoredNote(
                    note=KnowledgeNote(
                        id=row_id,
                        text=text,
                        embedding=_deserialize_float32(embedding_blob),
                        repo=repo,
                        tags=note_tags,
                        source_task=source_task,
                        timestamp=timestamp,
                    ),
                    score=score,
                )
            )
            if len(scored) == top_k:
                break
        return tuple(scored)

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if not self._closed:
            self._conn.close()
            self._closed = True
