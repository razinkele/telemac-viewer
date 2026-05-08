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
