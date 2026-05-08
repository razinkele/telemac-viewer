"""Tests for auth.routes — login/logout/admin HTTP behavior."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient


def _wrap(routes, db_path, secret=b"x" * 32):
    """Build a wrapped Starlette+middleware app for tests.

    CRITICAL: set `app.state.auth_secret` and `app.state.auth_db_path`
    so route handlers don't fall back to `load_or_create_secret` —
    which would create `~/.telemac-viewer/auth_secret` as a test
    side-effect AND would mean the cookie signed by the test's
    in-memory `secret` arg wouldn't be the same secret the route
    handler reads on response.
    """
    from auth.middleware import auth_middleware

    inner = Starlette(routes=routes)
    inner.state.auth_secret = secret
    inner.state.auth_db_path = db_path
    return auth_middleware(inner, db_path=db_path, secret=secret)


def _seed_user(db, username="alice", password="hunter2", is_admin=False):
    """Seed a user with a REAL bcrypt hash so route tests exercise the
    full verify path. (Earlier drafts used `password_hash="x"` which made
    the tests lie about end-to-end auth — flagged in the loop-#6
    test-quality review.)
    """
    from auth.core import connect, create_user, ensure_schema
    from auth.crypto import hash_password

    h = hash_password(password)
    assert h.startswith("$2b$"), f"expected real bcrypt hash, got {h!r}"
    with connect(db) as conn:
        ensure_schema(conn)
        return create_user(
            conn,
            username=username,
            password_hash=h,
            is_admin=is_admin,
        )


def test_login_get_renders_form(tmp_path: Path) -> None:
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r = client.get("/login")
        assert r.status_code == 200
        assert "<form" in r.text
        assert 'name="username"' in r.text
        assert 'name="password"' in r.text


def test_login_post_success_sets_cookie_and_redirects(tmp_path: Path) -> None:
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r = client.post(
            "/login",
            data={"username": "alice", "password": "hunter2"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert r.headers["location"] == "/"
        assert "tv_session=" in r.headers.get("set-cookie", "")
        assert "Path=/" in r.headers["set-cookie"]
        assert "HttpOnly" in r.headers["set-cookie"]
        assert "samesite=lax" in r.headers["set-cookie"].lower()


def test_login_post_failure_redirects_with_error(tmp_path: Path) -> None:
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r = client.post(
            "/login",
            data={"username": "alice", "password": "wrong"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "/login?error=invalid" in r.headers["location"]


def test_login_post_unknown_user_does_not_distinguish(tmp_path: Path) -> None:
    """User-not-found and wrong-password produce the same response."""
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r1 = client.post(
            "/login",
            data={"username": "ghost", "password": "anything"},
            follow_redirects=False,
        )
        r2 = client.post(
            "/login",
            data={"username": "alice", "password": "wrong"},
            follow_redirects=False,
        )
        assert r1.status_code == r2.status_code == 302
        assert r1.headers["location"] == r2.headers["location"]


def test_login_next_param_relative_path_honored(tmp_path: Path) -> None:
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r = client.post(
            "/login?next=/admin/users",
            data={"username": "alice", "password": "hunter2"},
            follow_redirects=False,
        )
        assert r.headers["location"] == "/admin/users"


def test_login_next_param_absolute_url_replaced_with_root(tmp_path: Path) -> None:
    """Spec §7.1 step 4: open-redirect guard.

    Includes the backslash-bypass case (``/\\evil`` is normalized by browsers
    to ``//evil``) found by the loop-#6 adversarial review — the regex must
    reject both ``/\\`` and ``//``.
    """
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        hostile_inputs = (
            "https://evil.example/",
            "//evil.example/",
            "/\\evil.example/",  # browser normalizes \ → /
            "javascript:alert(1)",
            "%2F%2Fevil.example/",  # URL-encoded //
            "/%0D%0ALocation:%20https://evil",  # CRLF URL-encoded (control chars rejected after decode)
            "/" + "a" * 300,  # over the 256-char length cap
        )
        for hostile in hostile_inputs:
            r = client.post(
                f"/login?next={hostile}",
                data={"username": "alice", "password": "hunter2"},
                follow_redirects=False,
            )
            assert r.headers["location"] == "/", f"failed for next={hostile!r}"


def test_login_get_with_malformed_cookie_clears_it(tmp_path: Path) -> None:
    """Spec §7.1 step 3."""
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    app = _wrap(auth_routes, db)
    with TestClient(app) as client:
        r = client.get("/login", cookies={"tv_session": "garbage"})
        # The set-cookie header on the response should clear the bad cookie
        cookie_hdr = r.headers.get("set-cookie", "")
        assert "tv_session=" in cookie_hdr
        assert "Max-Age=0" in cookie_hdr or "max-age=0" in cookie_hdr.lower()
        assert "Path=/" in cookie_hdr


def test_logout_clears_cookie_and_redirects(tmp_path: Path) -> None:
    from auth.crypto import sign_session_cookie
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    _seed_user(db)
    secret = b"x" * 32
    app = _wrap(auth_routes, db, secret=secret)
    token = sign_session_cookie(user_id=1, secret=secret)
    with TestClient(app) as client:
        r = client.post(
            "/logout", cookies={"tv_session": token}, follow_redirects=False
        )
        assert r.status_code == 302
        assert "/login" in r.headers["location"]
        assert (
            "Max-Age=0" in r.headers["set-cookie"]
            or "max-age=0" in r.headers["set-cookie"].lower()
        )


def _admin_client(tmp_path: Path):
    """Return TestClient + secret + admin's session cookie + admin id."""
    from auth.core import connect, create_user, ensure_schema
    from auth.crypto import hash_password, sign_session_cookie
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    secret = b"x" * 32
    with connect(db) as conn:
        ensure_schema(conn)
        admin_id = create_user(
            conn,
            username="admin",
            password_hash=hash_password("p"),
            is_admin=True,
        )
    app = _wrap(auth_routes, db, secret=secret)
    token = sign_session_cookie(user_id=admin_id, secret=secret)
    return TestClient(app), token, admin_id, db


def test_admin_users_lists_users(tmp_path: Path) -> None:
    client, token, admin_id, db = _admin_client(tmp_path)
    _seed_user(db, username="alice", password="x")
    with client:
        r = client.get("/admin/users", cookies={"tv_session": token})
        assert r.status_code == 200
        assert "alice" in r.text
        assert "admin" in r.text


def test_admin_users_forbidden_for_non_admin(tmp_path: Path) -> None:
    from auth.crypto import sign_session_cookie
    from auth.routes import auth_routes

    db = tmp_path / "auth.db"
    secret = b"x" * 32
    bob_id = _seed_user(db, username="bob", password="x", is_admin=False)
    app = _wrap(auth_routes, db, secret=secret)
    token = sign_session_cookie(user_id=bob_id, secret=secret)
    with TestClient(app) as client:
        r = client.get("/admin/users", cookies={"tv_session": token})
        assert r.status_code == 403


def test_admin_create_user(tmp_path: Path) -> None:
    client, token, _admin_id, db = _admin_client(tmp_path)
    with client:
        r = client.post(
            "/admin/users/create",
            data={"username": "alice", "password": "longenough", "display_name": "A"},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 302
    from auth.core import connect, get_user_by_username

    with connect(db) as conn:
        assert get_user_by_username(conn, "alice") is not None


def test_admin_create_rejects_short_password(tmp_path: Path) -> None:
    client, token, _admin_id, _db = _admin_client(tmp_path)
    with client:
        r = client.post(
            "/admin/users/create",
            data={"username": "alice", "password": "short", "display_name": "A"},
            cookies={"tv_session": token},
        )
        assert r.status_code == 400
        assert "8" in r.text  # mention min length


def test_admin_create_rejects_long_password_utf8(tmp_path: Path) -> None:
    """Spec §5: ≤ 72 bytes UTF-8, NOT 72 chars."""
    client, token, _admin_id, _db = _admin_client(tmp_path)
    # 37 two-byte chars = 74 bytes (over limit) but 37 chars < 72 char-count
    long_pw = "ñ" * 37
    assert len(long_pw.encode("utf-8")) == 74
    with client:
        r = client.post(
            "/admin/users/create",
            data={"username": "alice", "password": long_pw, "display_name": "A"},
            cookies={"tv_session": token},
        )
        assert r.status_code == 400
        assert "72" in r.text


def test_admin_create_rejects_duplicate_username(tmp_path: Path) -> None:
    client, token, _admin_id, db = _admin_client(tmp_path)
    _seed_user(db, username="alice", password="x")
    with client:
        r = client.post(
            "/admin/users/create",
            data={"username": "alice", "password": "longenough", "display_name": ""},
            cookies={"tv_session": token},
        )
        assert r.status_code == 400
        assert "already exists" in r.text.lower()


def test_admin_create_rejects_invalid_username_chars(tmp_path: Path) -> None:
    client, token, _admin_id, _db = _admin_client(tmp_path)
    with client:
        for bad in ("alice space", "alice/", "<script>"):
            r = client.post(
                "/admin/users/create",
                data={"username": bad, "password": "longenough", "display_name": ""},
                cookies={"tv_session": token},
            )
            assert r.status_code == 400, f"failed for {bad!r}"


def test_admin_html_autoescapes_username_and_display_name(tmp_path: Path) -> None:
    """Spec §10 risk row: stored XSS via username or display_name."""
    from auth.core import connect, create_user, ensure_schema
    from auth.crypto import hash_password

    client, token, _admin_id, db = _admin_client(tmp_path)
    # Force-insert a hostile display_name (bypassing the regex on POST)
    with connect(db) as conn:
        create_user(
            conn,
            username="bob",
            password_hash=hash_password("x"),
            display_name="<script>alert('xss')</script>",
        )
    with client:
        r = client.get("/admin/users", cookies={"tv_session": token})
        assert "<script>alert('xss')</script>" not in r.text
        assert "&lt;script&gt;" in r.text


def test_admin_edit_user_updates_display_name(tmp_path: Path) -> None:
    client, token, _admin_id, db = _admin_client(tmp_path)
    uid = _seed_user(db, username="alice", password="x")
    with client:
        r = client.post(
            f"/admin/users/{uid}/edit",
            data={"display_name": "New Name"},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 302
    from auth.core import connect, get_user_by_id

    with connect(db) as conn:
        u = get_user_by_id(conn, uid)
    assert u.display_name == "New Name"


def test_admin_reset_password(tmp_path: Path) -> None:
    from auth.crypto import verify_password

    client, token, _admin_id, db = _admin_client(tmp_path)
    uid = _seed_user(db, username="alice", password="oldpass99")
    with client:
        r = client.post(
            f"/admin/users/{uid}/reset-password",
            data={"password": "newpassword99"},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 302
    from auth.core import connect, get_user_by_id

    with connect(db) as conn:
        u = get_user_by_id(conn, uid)
    assert verify_password("newpassword99", u.password_hash)


def test_admin_delete_user_succeeds_when_other_admins_exist(tmp_path: Path) -> None:
    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="x")
    with client:
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 302
    from auth.core import connect, get_user_by_id

    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is None


def test_admin_delete_refuses_last_admin(tmp_path: Path) -> None:
    client, token, admin_id, db = _admin_client(tmp_path)
    # admin is the only admin
    with client:
        r = client.post(
            f"/admin/users/{admin_id}/delete",
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        # Refusal returns 400 with explicit message
        assert r.status_code == 400
        assert "last admin" in r.text.lower()
    from auth.core import connect, get_user_by_id

    with connect(db) as conn:
        assert get_user_by_id(conn, admin_id) is not None


def test_admin_edit_returns_400_when_user_gone(tmp_path: Path) -> None:
    """Two admins racing: A edits user X while B has just deleted X.

    The atomic last-admin guard handles delete itself; the edit handler
    must surface "no longer exists" rather than redirect with a misleading
    success log.
    """
    from auth.core import connect

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="x")
    # Delete bob out from under the edit
    with connect(db) as conn:
        conn.execute("DELETE FROM users WHERE id=?", (bob_id,))
    with client:
        r = client.post(
            f"/admin/users/{bob_id}/edit",
            data={"display_name": "Bobbie"},
            cookies={"tv_session": token},
        )
        assert r.status_code == 400
        assert "no longer exists" in r.text


def test_admin_reset_password_returns_400_when_user_gone(tmp_path: Path) -> None:
    from auth.core import connect

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="x")
    with connect(db) as conn:
        conn.execute("DELETE FROM users WHERE id=?", (bob_id,))
    with client:
        r = client.post(
            f"/admin/users/{bob_id}/reset-password",
            data={"password": "longenough"},
            cookies={"tv_session": token},
        )
        assert r.status_code == 400
        assert "no longer exists" in r.text
