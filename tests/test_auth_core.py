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


def test_update_user_modifies_display_name_and_is_admin(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        get_user_by_id,
        update_user,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        uid = create_user(
            conn, username="alice", password_hash="h", display_name="A", is_admin=False
        )
        update_user(conn, user_id=uid, display_name="Alice", is_admin=True)
        u = get_user_by_id(conn, uid)
    assert u.display_name == "Alice"
    assert u.is_admin is True


def test_update_password_hash(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        get_user_by_id,
        update_password_hash,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        uid = create_user(conn, username="alice", password_hash="old")
        update_password_hash(conn, user_id=uid, password_hash="new")
        u = get_user_by_id(conn, uid)
    assert u.password_hash == "new"


def test_delete_user_atomic_succeeds_when_other_admins_exist(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        delete_user_atomic,
        get_user_by_id,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        a = create_user(conn, username="alice", password_hash="h", is_admin=True)
        b = create_user(conn, username="bob", password_hash="h", is_admin=True)
        deleted = delete_user_atomic(conn, user_id=a)
        assert deleted is True
        assert get_user_by_id(conn, a) is None
        assert get_user_by_id(conn, b) is not None


def test_delete_user_atomic_refuses_last_admin(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        delete_user_atomic,
        get_user_by_id,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        a = create_user(conn, username="alice", password_hash="h", is_admin=True)
        deleted = delete_user_atomic(conn, user_id=a)
        assert deleted is False
        assert get_user_by_id(conn, a) is not None  # still there


def test_delete_user_atomic_allows_non_admin_when_only_admin(tmp_path: Path) -> None:
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        delete_user_atomic,
        get_user_by_id,
    )

    with connect(tmp_path / "auth.db") as conn:
        ensure_schema(conn)
        admin = create_user(conn, username="admin", password_hash="h", is_admin=True)
        bob = create_user(conn, username="bob", password_hash="h", is_admin=False)
        assert delete_user_atomic(conn, user_id=bob) is True
        assert get_user_by_id(conn, admin) is not None


def test_delete_user_atomic_serializes_concurrent_admin_deletes(tmp_path: Path) -> None:
    """Race regression: two admin sessions deleting each other simultaneously
    must NOT both succeed. BEGIN IMMEDIATE serializes them.
    """
    import threading
    from auth.core import (
        connect,
        create_user,
        ensure_schema,
        delete_user_atomic,
        list_users,
    )

    db = tmp_path / "auth.db"
    with connect(db) as conn:
        ensure_schema(conn)
        a = create_user(conn, username="alpha", password_hash="h", is_admin=True)
        b = create_user(conn, username="bravo", password_hash="h", is_admin=True)

    results: list[bool] = []
    barrier = threading.Barrier(2)

    def attempt_delete(target_id: int) -> None:
        with connect(db) as conn:
            barrier.wait()
            results.append(delete_user_atomic(conn, user_id=target_id))

    t1 = threading.Thread(target=attempt_delete, args=(b,))
    t2 = threading.Thread(target=attempt_delete, args=(a,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == [False, True], (
        f"Expected exactly one delete to succeed; got {results}. "
        "Both succeeding means the last-admin guard race-condition is back."
    )
    with connect(db) as conn:
        remaining = list_users(conn)
    admin_count = sum(1 for u in remaining if u.is_admin)
    assert admin_count == 1, f"Expected exactly 1 admin remaining; got {admin_count}"


# --- CLI smoke tests (Task 10) ---


def _run_cli(*args, env_extra=None, input_text=None):
    """Run `python -m auth.cli ...` as a subprocess. Returns CompletedProcess."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    # Prepend, don't overwrite — preserve any inherited PYTHONPATH the
    # project's pytest config may rely on for stub modules.
    env["PYTHONPATH"] = (
        "/home/razinka/telemac/telemac-viewer" + os.pathsep + env.get("PYTHONPATH", "")
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "auth.cli", *args],
        capture_output=True,
        text=True,
        input=input_text,
        env=env,
    )


def test_cli_create_admin_with_password_file(tmp_path: Path) -> None:
    pwfile = tmp_path / "pw"
    pwfile.write_text("hunter2longenough\n")  # trailing newline must be stripped
    pwfile.chmod(0o600)
    db = tmp_path / "auth.db"

    r = _run_cli(
        "create-admin",
        "--username",
        "alice",
        "--display-name",
        "A",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r.returncode == 0, r.stderr
    # Verify the user was created and the password works (i.e. trailing \n was stripped)
    from auth.core import connect, get_user_by_username
    from auth.crypto import verify_password

    with connect(db) as conn:
        u = get_user_by_username(conn, "alice")
    assert u is not None
    assert u.is_admin is True
    assert verify_password("hunter2longenough", u.password_hash), (
        "trailing newline was not stripped — password mismatch"
    )


def test_cli_create_admin_refuses_when_admin_exists(tmp_path: Path) -> None:
    pwfile = tmp_path / "pw"
    pwfile.write_text("hunter2longenough")
    pwfile.chmod(0o600)
    db = tmp_path / "auth.db"

    r1 = _run_cli(
        "create-admin",
        "--username",
        "alice",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r1.returncode == 0

    r2 = _run_cli(
        "create-admin",
        "--username",
        "bob",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r2.returncode == 2, r2.stderr
    assert "admin already exists" in r2.stderr.lower()


def test_cli_create_admin_refuses_loose_password_file_mode(tmp_path: Path) -> None:
    pwfile = tmp_path / "pw"
    pwfile.write_text("hunter2longenough")
    pwfile.chmod(0o644)  # too loose
    db = tmp_path / "auth.db"

    r = _run_cli(
        "create-admin",
        "--username",
        "alice",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r.returncode == 5, r.stderr
    assert "0600" in r.stderr or "mode" in r.stderr.lower()


def test_cli_reset_password_user_not_found(tmp_path: Path) -> None:
    pwfile = tmp_path / "pw"
    pwfile.write_text("longenough")
    pwfile.chmod(0o600)
    db = tmp_path / "auth.db"
    # ensure schema exists but no users
    from auth.core import connect, ensure_schema

    with connect(db) as conn:
        ensure_schema(conn)

    r = _run_cli(
        "reset-password",
        "--username",
        "ghost",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r.returncode == 4
    assert "not found" in r.stderr.lower()


def test_cli_create_admin_short_password_rejected(tmp_path: Path) -> None:
    pwfile = tmp_path / "pw"
    pwfile.write_text("short")
    pwfile.chmod(0o600)
    db = tmp_path / "auth.db"

    r = _run_cli(
        "create-admin",
        "--username",
        "alice",
        "--password-file",
        str(pwfile),
        env_extra={"TELEMAC_VIEWER_DB": str(db)},
    )
    assert r.returncode != 0
    assert "8" in r.stderr


def test_cli_create_admin_refuses_non_tty_stdin(tmp_path: Path) -> None:
    """Spec §8: non-tty stdin without --password-file refuses with exit 5."""
    import subprocess
    import sys

    db = tmp_path / "auth.db"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        "/home/razinka/telemac/telemac-viewer" + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["TELEMAC_VIEWER_DB"] = str(db)
    r = subprocess.run(
        [sys.executable, "-m", "auth.cli", "create-admin", "--username", "x"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert r.returncode == 5, r.stderr
    assert "non-interactive" in r.stderr.lower() or "tty" in r.stderr.lower()


def test_save_user_prefs_outcome_ok_gone_error(tmp_path: Path, caplog) -> None:
    import logging
    from unittest.mock import patch

    from auth.core import connect, create_user, ensure_schema, save_user_prefs_outcome

    db = tmp_path / "auth.db"
    with connect(db) as conn:
        ensure_schema(conn)
        uid = create_user(conn, username="alice", password_hash="h")

        # 'ok' path
        assert save_user_prefs_outcome(conn, user_id=uid, prefs={"x": 1}) == "ok"

        # 'gone' path — user deleted out from under us
        conn.execute("DELETE FROM users WHERE id=?", (uid,))
        assert save_user_prefs_outcome(conn, user_id=uid, prefs={"x": 1}) == "gone"

        # 'error' path — DB write fails
        with caplog.at_level(logging.ERROR, logger="auth"):
            with patch(
                "auth.core.update_preferences",
                side_effect=__import__("sqlite3").OperationalError("locked"),
            ):
                assert (
                    save_user_prefs_outcome(conn, user_id=uid, prefs={"x": 1})
                    == "error"
                )
        assert any("Failed to save preferences" in r.message for r in caplog.records)
