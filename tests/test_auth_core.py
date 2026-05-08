"""Tests for auth.core — sqlite + users CRUD."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


def test_connect_creates_db_with_wal_and_0600(tmp_path: Path) -> None:
    from auth.core import connect

    db_path = tmp_path / "auth.db"
    with connect(db_path) as conn:
        # WAL mode active
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    # File created with 0o600
    assert db_path.exists()
    assert oct(db_path.stat().st_mode & 0o777) == "0o600"
