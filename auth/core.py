"""sqlite connection, schema, and User CRUD for auth.

This module is sectioned for readability:
  - Constants
  - DB connection
  - Schema setup + verification
  - User dataclass + CRUD

Section anchors are intentional — if this module exceeds 250 LoC
during implementation, split at the section boundary into a
separate `db.py` (connect + schema) and keep users CRUD here.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# --- Constants ---

DEFAULT_DB_PATH = Path.home() / ".telemac-viewer" / "auth.db"
DEFAULT_DIR = DEFAULT_DB_PATH.parent

# --- DB connection ---


@contextmanager
def connect(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 connection in WAL mode, autocommit mode.

    `isolation_level=None` puts sqlite3 in *autocommit* mode: no implicit
    BEGIN/COMMIT around DML. This is REQUIRED so callers can issue
    explicit `BEGIN IMMEDIATE` (e.g. `delete_user_atomic`) without
    tripping `OperationalError: cannot start a transaction within a
    transaction`. The trade-off is that each DML statement commits
    immediately; the auth code's CRUD operations are independent
    single-statement writes, so this is the right default.

    The DB file (and its parent dir) are created with 0o600/0o700 on
    first creation. The file is pre-created atomically via os.open with
    O_CREAT|O_EXCL|0o600 BEFORE sqlite3.connect to avoid a TOCTOU race
    where sqlite3 would otherwise create the file with umask-default
    mode (typically 0o644) and a chmod-after-the-fact would leave a
    brief world-readable window.

    `PRAGMA busy_timeout=5000` makes concurrent writers wait up to 5s
    for the WRITE lock instead of failing immediately with SQLITE_BUSY.
    """
    is_new = not db_path.exists()
    if is_new:
        db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The mkdir call's `mode` arg is masked by umask. Force 0o700
        # explicitly so the directory is owner-only regardless of umask.
        os.chmod(db_path.parent, 0o700)
        # Pre-create the DB file at 0o600 atomically; sqlite3 will
        # then open the existing file rather than creating it with
        # umask-default permissions.
        fd = os.open(db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)

    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        yield conn
    finally:
        conn.close()


# --- Schema ---

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT NOT NULL UNIQUE COLLATE NOCASE
                  CHECK(length(username) BETWEEN 1 AND 64),
  password_hash TEXT NOT NULL,
  display_name  TEXT CHECK(display_name IS NULL OR length(display_name) <= 64),
  is_admin      INTEGER NOT NULL DEFAULT 0
                  CHECK(is_admin IN (0, 1)),
  preferences   TEXT NOT NULL DEFAULT '{}'
                  CHECK(length(preferences) <= 8192),
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

UPDATED_AT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS users_updated_at
AFTER UPDATE ON users
FOR EACH ROW BEGIN
  UPDATE users SET updated_at = datetime('now') WHERE id = NEW.id;
END;
"""

# (name, type-uppercased, notnull, pk) — order-insensitive comparison.
# Note: sqlite reports notnull=0 for INTEGER PRIMARY KEY because it is
# technically a rowid alias and accepts NULL on INSERT (which the engine
# auto-replaces with the next rowid). This is sqlite's documented quirk,
# not a schema bug.
EXPECTED_COLUMNS: frozenset[tuple[str, str, int, int]] = frozenset(
    {
        ("id", "INTEGER", 0, 1),
        ("username", "TEXT", 1, 0),
        ("password_hash", "TEXT", 1, 0),
        ("display_name", "TEXT", 0, 0),
        ("is_admin", "INTEGER", 1, 0),
        ("preferences", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    }
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the users table + updated_at trigger, then verify shape."""
    conn.executescript(SCHEMA_DDL + UPDATED_AT_TRIGGER)
    _verify_schema(conn)


def _verify_schema(conn: sqlite3.Connection) -> None:
    """Compare PRAGMA table_info(users) against EXPECTED_COLUMNS.

    CHECK constraint expression text is NOT verified (sqlite's pragma
    doesn't expose it). Documented limitation in spec §5.
    """
    rows = list(conn.execute("PRAGMA table_info(users)"))
    actual = frozenset(
        (row[1], row[2].upper().split("(")[0], row[3], row[5]) for row in rows
    )
    if actual != EXPECTED_COLUMNS:
        msg = (
            "auth.db schema does not match v1 expectations.\n"
            f"       Found columns: {sorted(c[0] for c in actual)}\n"
            f"       Expected:      {sorted(c[0] for c in EXPECTED_COLUMNS)}\n"
            "       v1 has no migration tool. Recovery options:\n"
            "         (a) back up auth.db, remove it, and re-run "
            "`python -m auth.cli create-admin`, or\n"
            "         (b) downgrade telemac-viewer to the previous version.\n"
            '       See README "Multi-user setup → Schema mismatch recovery".'
        )
        raise RuntimeError(msg)
