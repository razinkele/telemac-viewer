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

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("auth")

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


# --- User dataclass + CRUD ---


@dataclass(frozen=True)
class User:
    id: int
    username: str
    password_hash: str
    display_name: str | None
    is_admin: bool
    preferences: dict
    created_at: str
    updated_at: str


def _row_to_user(row: tuple) -> User:
    prefs_str = row[5] or "{}"
    try:
        prefs = json.loads(prefs_str)
        if not isinstance(prefs, dict):
            raise ValueError("not a dict")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "Malformed preferences for user_id=%s, falling back to {}: %s",
            row[0],
            e,
        )
        prefs = {}
    return User(
        id=row[0],
        username=row[1],
        password_hash=row[2],
        display_name=row[3],
        is_admin=bool(row[4]),
        preferences=prefs,
        created_at=row[6],
        updated_at=row[7],
    )


_SELECT_COLS = (
    "id, username, password_hash, display_name, is_admin, "
    "preferences, created_at, updated_at"
)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> User | None:
    row = conn.execute(
        f"SELECT {_SELECT_COLS} FROM users WHERE username = ? COLLATE NOCASE",
        (username,),
    ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> User | None:
    row = conn.execute(
        f"SELECT {_SELECT_COLS} FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return _row_to_user(row) if row else None


def create_user(
    conn: sqlite3.Connection,
    *,
    username: str,
    password_hash: str,
    display_name: str | None = None,
    is_admin: bool = False,
) -> int:
    """Insert a new user. Raises sqlite3.IntegrityError on duplicate username."""
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, display_name, is_admin) "
        "VALUES (?, ?, ?, ?)",
        (username, password_hash, display_name, 1 if is_admin else 0),
    )
    return cur.lastrowid


def list_users(conn: sqlite3.Connection) -> list[User]:
    rows = conn.execute(
        f"SELECT {_SELECT_COLS} FROM users ORDER BY username COLLATE NOCASE"
    ).fetchall()
    return [_row_to_user(r) for r in rows]


def update_preferences(conn: sqlite3.Connection, *, user_id: int, prefs: dict) -> int:
    """Persist preferences. Returns rowcount (0 if user no longer exists)."""
    cur = conn.execute(
        "UPDATE users SET preferences = ? WHERE id = ?",
        (json.dumps(prefs), user_id),
    )
    return cur.rowcount


def update_user(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    display_name: str | None = None,
    is_admin: bool | None = None,
) -> int:
    """Update display_name and/or is_admin. Returns rowcount."""
    sets = []
    args: list = []
    if display_name is not None:
        sets.append("display_name = ?")
        args.append(display_name)
    if is_admin is not None:
        sets.append("is_admin = ?")
        args.append(1 if is_admin else 0)
    if not sets:
        return 0
    args.append(user_id)
    cur = conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", args)
    return cur.rowcount


def update_password_hash(
    conn: sqlite3.Connection, *, user_id: int, password_hash: str
) -> int:
    cur = conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    return cur.rowcount


def delete_user_atomic(conn: sqlite3.Connection, *, user_id: int) -> bool:
    """Delete user. Refuses if doing so would leave zero admins.

    Returns True if a row was deleted, False if blocked by the
    last-admin guard (or the user didn't exist).

    SAFETY NOTE: the predicate `count(*) > 1` is evaluated within the
    transaction's read snapshot. Without explicit serialization, two
    concurrent admin sessions could each see count=2 and both delete,
    leaving zero admins. The function therefore acquires an IMMEDIATE
    (reserved-lock) transaction so only one writer at a time evaluates
    the count + delete.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            """
            DELETE FROM users
             WHERE id = ?
               AND (
                 is_admin = 0
                 OR (SELECT count(*) FROM users WHERE is_admin = 1) > 1
               )
            """,
            (user_id,),
        )
        deleted = cur.rowcount > 0
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return deleted
