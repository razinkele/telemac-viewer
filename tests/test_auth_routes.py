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
