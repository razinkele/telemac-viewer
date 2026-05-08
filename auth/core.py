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
    first creation; a `chmod` failure on creation refuses with the
    error propagated. Existing files are not re-chmod'd.

    `PRAGMA busy_timeout=5000` makes concurrent writers wait up to 5s
    for the WRITE lock instead of failing immediately with SQLITE_BUSY.
    """
    is_new = not db_path.exists()
    if is_new:
        db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The mkdir call's `mode` arg is masked by umask. Force 0o700
        # explicitly so the directory is owner-only regardless of umask.
        os.chmod(db_path.parent, 0o700)

    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        if is_new:
            os.chmod(db_path, 0o600)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        yield conn
    finally:
        conn.close()
