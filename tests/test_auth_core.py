"""Tests for auth.core — sqlite + users CRUD."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


def test_connect_creates_db_with_wal_and_0600(tmp_path: Path) -> None:
    from auth.core import connect

    # Use a nested path so the parent dir is genuinely created by connect()
    db_path = tmp_path / "telemac-viewer-test" / "auth.db"
    with connect(db_path) as conn:
        # WAL mode active
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    # File created with 0o600
    assert db_path.exists()
    assert oct(db_path.stat().st_mode & 0o777) == "0o600"
    # Parent dir created with 0o700
    assert oct(db_path.parent.stat().st_mode & 0o777) == "0o700"


def test_ensure_schema_creates_users_table_with_expected_columns(
    tmp_path: Path,
) -> None:
    from auth.core import connect, ensure_schema

    db_path = tmp_path / "auth.db"
    with connect(db_path) as conn:
        ensure_schema(conn)
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(users)")}

    expected = {
        "id",
        "username",
        "password_hash",
        "display_name",
        "is_admin",
        "preferences",
        "created_at",
        "updated_at",
    }
    assert set(cols.keys()) == expected


def test_updated_at_trigger_fires_on_update(tmp_path: Path) -> None:
    from auth.core import connect, ensure_schema

    db_path = tmp_path / "auth.db"
    with connect(db_path) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at, updated_at) "
            "VALUES (?, ?, '2000-01-01 00:00:00', '2000-01-01 00:00:00')",
            ("alice", "x"),
        )
        conn.execute("UPDATE users SET display_name='Alice' WHERE username='alice'")
        after = conn.execute(
            "SELECT updated_at FROM users WHERE username='alice'"
        ).fetchone()[0]
    assert after != "2000-01-01 00:00:00", (
        "updated_at trigger did not fire — value is still the literal "
        f"set on INSERT: {after!r}"
    )


def test_is_admin_check_constraint_rejects_invalid(tmp_path: Path) -> None:
    from auth.core import connect, ensure_schema
    import sqlite3

    db_path = tmp_path / "auth.db"
    with connect(db_path) as conn:
        ensure_schema(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) "
                "VALUES (?, ?, ?)",
                ("bob", "x", 2),  # invalid; CHECK rejects
            )


def test_verify_schema_detects_extra_column(tmp_path: Path) -> None:
    from auth.core import connect, ensure_schema, _verify_schema

    db_path = tmp_path / "auth.db"
    with connect(db_path) as conn:
        ensure_schema(conn)
        conn.execute("ALTER TABLE users ADD COLUMN extra TEXT")
        with pytest.raises(RuntimeError) as excinfo:
            _verify_schema(conn)
    msg = str(excinfo.value)
    assert "schema does not match" in msg
    assert "Recovery options" in msg
    assert "create-admin" in msg
    assert "README" in msg


def test_create_and_get_user(tmp_path: Path) -> None:
    from auth.core import connect, create_user, ensure_schema, get_user_by_username

    db = tmp_path / "auth.db"
    with connect(db) as conn:
        ensure_schema(conn)
        uid = create_user(
            conn,
            username="alice",
            password_hash="h",
            display_name="Alice",
            is_admin=True,
        )
        u = get_user_by_username(conn, "alice")
    assert u is not None
    assert u.id == uid
    assert u.username == "alice"
    assert u.display_name == "Alice"
    assert u.is_admin is True
    assert u.preferences == {}


def test_get_user_by_username_case_insensitive(tmp_path: Path) -> None:
    from auth.core import connect, create_user, ensure_schema, get_user_by_username

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        create_user(conn, username="Alice", password_hash="h")
        assert get_user_by_username(conn, "alice") is not None
        assert get_user_by_username(conn, "ALICE") is not None
        assert get_user_by_username(conn, "bob") is None


def test_get_user_by_id(tmp_path: Path) -> None:
    from auth.core import connect, create_user, ensure_schema, get_user_by_id

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        uid = create_user(conn, username="alice", password_hash="h")
        u = get_user_by_id(conn, uid)
        assert u is not None
        assert u.username == "alice"
        assert get_user_by_id(conn, 99999) is None


def test_create_user_rejects_duplicate(tmp_path: Path) -> None:
    import sqlite3
    from auth.core import connect, create_user, ensure_schema

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        create_user(conn, username="alice", password_hash="h")
        with pytest.raises(sqlite3.IntegrityError):
            create_user(
                conn, username="ALICE", password_hash="h"
            )  # case-insensitive UNIQUE


def test_list_users_orders_by_username(tmp_path: Path) -> None:
    from auth.core import connect, create_user, ensure_schema, list_users

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        for name in ("charlie", "alice", "bob"):
            create_user(conn, username=name, password_hash="h")
        users = list_users(conn)
    assert [u.username for u in users] == ["alice", "bob", "charlie"]


def test_user_preferences_round_trip(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        get_user_by_username,
        update_preferences,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        create_user(conn, username="alice", password_hash="h")
        update_preferences(
            conn, user_id=1, prefs={"variable": "WATER DEPTH", "palette": "Viridis"}
        )
        u = get_user_by_username(conn, "alice")
    assert u.preferences == {"variable": "WATER DEPTH", "palette": "Viridis"}


def test_preferences_malformed_json_falls_back_to_empty(tmp_path: Path, caplog) -> None:
    import logging

    from auth.core import connect, create_user, ensure_schema, get_user_by_username

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        create_user(conn, username="alice", password_hash="h")
        conn.execute("UPDATE users SET preferences='not-json' WHERE username='alice'")
        with caplog.at_level(logging.WARNING, logger="auth"):
            u = get_user_by_username(conn, "alice")
    assert u.preferences == {}
    assert any("Malformed preferences" in r.message for r in caplog.records), (
        f"Expected WARNING log; got {[r.message for r in caplog.records]}"
    )


def test_update_preferences_returns_zero_rowcount_when_user_deleted(
    tmp_path: Path,
) -> None:
    from auth.core import connect, create_user, ensure_schema, update_preferences

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        uid = create_user(conn, username="alice", password_hash="h")
        conn.execute("DELETE FROM users WHERE id=?", (uid,))
        rowcount = update_preferences(conn, user_id=uid, prefs={"variable": "X"})
    assert rowcount == 0
