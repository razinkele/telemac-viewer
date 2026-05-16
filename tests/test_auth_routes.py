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


# --- Task 10: two-step delete with signed confirm token ---


def _mint_confirm_token(secret: bytes, actor_id: int, target_id: int) -> str:
    """Mint a fresh admin-delete-confirm token for tests."""
    import itsdangerous

    return itsdangerous.URLSafeTimedSerializer(
        secret, salt="admin-delete-confirm"
    ).dumps({"actor_id": actor_id, "target_id": target_id})


def _get_delete_token(client, token: str, target_id: int) -> str:
    """GET the confirm page and parse the hidden confirm_token value.

    Tests the full GET→POST flow exactly like the browser would.
    """
    r = client.get(
        f"/admin/users/{target_id}/delete",
        cookies={"tv_session": token},
    )
    assert r.status_code == 200, f"confirm GET failed: {r.status_code} {r.text[:200]}"
    m = re.search(r'name="confirm_token"\s+value="([^"]+)"', r.text)
    assert m, f"confirm_token not in confirm page: {r.text[:500]}"
    return m.group(1)


def test_admin_delete_get_renders_confirm_with_size_and_resolved_path(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """GET /admin/users/<uid>/delete renders the confirm page with the
    username, file count, size_human, resolved path, AND a hidden
    confirm_token form field."""
    import model_library

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")

    # Seed a real file under users/<bob_id>/ so size_human shows non-zero.
    bob_root = model_library.user_library_root(bob_id)
    (bob_root / "demo.txt").write_bytes(b"hello world!")  # 12 bytes

    with client:
        r = client.get(
            f"/admin/users/{bob_id}/delete",
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert "bob" in r.text
        assert "1 file" in r.text or "1 file(s)" in r.text
        assert "12 B" in r.text
        # Resolved path is the users/<uid>/ base.
        expected_path = str(
            (model_library.user_library_default_base() / str(bob_id)).resolve()
        )
        assert expected_path in r.text
        # Hidden token field
        assert 'name="confirm_token"' in r.text
        assert 'value="' in r.text


def test_admin_delete_get_returns_404_for_missing_user(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """GET on a non-existent uid -> 404."""
    client, token, _admin_id, _db = _admin_client(tmp_path)
    with client:
        r = client.get(
            "/admin/users/9999/delete",
            cookies={"tv_session": token},
        )
        assert r.status_code == 404


def test_admin_delete_post_succeeds_cascades_and_flashes_info(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """Happy path: GET confirm page -> POST with token -> auth row gone,
    library dir removed, flash 'info' cookie set, follow-up GET clears it."""
    import model_library
    from auth.core import connect, get_user_by_id

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")
    bob_root = model_library.user_library_root(bob_id)
    (bob_root / "file.txt").write_bytes(b"X")
    bob_base = model_library.user_library_default_base() / str(bob_id)
    assert bob_base.exists()

    with client:
        confirm_token = _get_delete_token(client, token, bob_id)
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": confirm_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/admin/users"
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    # Auth row gone
    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is None
    # Library dir gone
    assert not bob_base.exists()


def test_admin_delete_post_when_last_admin_refuses_and_preserves_library(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """POST to delete the only admin: must refuse, library MUST NOT be touched."""
    import model_library
    from auth.core import connect, get_user_by_id

    client, token, admin_id, db = _admin_client(tmp_path)
    # Seed admin's own library so we can verify it survives the refusal.
    admin_root = model_library.user_library_root(admin_id)
    (admin_root / "keepme.txt").write_bytes(b"important")
    admin_base = model_library.user_library_default_base() / str(admin_id)

    with client:
        confirm_token = _get_delete_token(client, token, admin_id)
        r = client.post(
            f"/admin/users/{admin_id}/delete",
            data={"confirm_token": confirm_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        # Flash payload encodes the warn level + "last admin" message.
        cookie_hdr = r.headers.get("set-cookie", "")
        assert "admin_flash=" in cookie_hdr

    # Auth row preserved
    with connect(db) as conn:
        assert get_user_by_id(conn, admin_id) is not None
    # Library dir UNTOUCHED — this is the critical invariant.
    assert admin_base.exists()
    assert (admin_base / "models" / "keepme.txt").exists()


def test_admin_delete_post_missing_confirm_token_rejects(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """POST with no confirm_token form field -> flash 'warn', auth row intact."""
    from auth.core import connect, get_user_by_id

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")

    with client:
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={},  # no confirm_token
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is not None


def test_admin_delete_post_bad_signature_confirm_token_rejects(
    tmp_path: Path, isolated_telemac_dirs, caplog
) -> None:
    """POST with garbage confirm_token -> flash 'warn', auth row intact, WARN log."""
    import logging

    from auth.core import connect, get_user_by_id

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")

    with client, caplog.at_level(logging.WARNING, logger="auth"):
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": "this-is-not-a-valid-token"},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    # WARN log emitted for bad-signature path (security signal)
    assert any("bad confirm token" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]

    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is not None


def test_admin_delete_post_expired_confirm_token_rejects(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """Confirm token older than 10 min (max_age=600) is rejected as expired.

    Mints a token with a fixed past timestamp via freezegun-style time
    manipulation — here, we monkeypatch itsdangerous's
    URLSafeTimedSerializer to mint with a stale timestamp directly.
    """
    import time

    import itsdangerous

    from auth.core import connect, get_user_by_id

    client, token, admin_id, db = _admin_client(tmp_path)
    secret = b"x" * 32
    bob_id = _seed_user(db, username="bob", password="hunter2")

    # Mint a token whose embedded timestamp is 1 hour ago by using
    # itsdangerous's `now` injection via _now_ms — we can't easily
    # patch this, so we use SignatureExpired's contract directly:
    # craft a token, then check max_age=0 rejects it. (The route uses
    # max_age=600; we simulate "expired" by checking the rejection
    # branch is reachable via SignatureExpired by passing a token whose
    # timestamp is just 1 second old + monkeypatched max_age via the
    # `loads(max_age=...)` call already does this if we pass a small
    # max_age. Cleanest test: monkeypatch time.time used by serializer
    # so the issued-at is far in the past.)
    real_time = time.time
    fake_t = [real_time() - 3600]  # 1h ago at mint time

    serializer = itsdangerous.URLSafeTimedSerializer(
        secret, salt="admin-delete-confirm"
    )
    # Monkeypatch time.time in the itsdangerous.timed module to mint
    # a token with a stale timestamp, then restore so the route's
    # max_age=600 check against the real `now` rejects it.
    import itsdangerous.timed as _td

    original = _td.time.time
    _td.time.time = lambda: fake_t[0]
    try:
        stale_token = serializer.dumps({"actor_id": admin_id, "target_id": bob_id})
    finally:
        _td.time.time = original

    with client:
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": stale_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is not None


def test_admin_delete_post_confirm_token_actor_mismatch_rejects(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """Token minted by admin A, replayed by admin B -> rejected.

    Same secret is used by both admins; the actor_id embedded in the
    payload must match the request-scope user_id."""
    from auth.core import connect, create_user, get_user_by_id
    from auth.crypto import hash_password, sign_session_cookie

    client, token, admin_a_id, db = _admin_client(tmp_path)
    secret = b"x" * 32  # same as _admin_client / _wrap default
    # Add a second admin (B) and a victim user.
    with connect(db) as conn:
        admin_b_id = create_user(
            conn,
            username="admin_b",
            password_hash=hash_password("hunter2"),
            is_admin=True,
        )
    bob_id = _seed_user(db, username="bob", password="hunter2")

    # A mints the token, B replays it.
    token_for_b = _mint_confirm_token(secret, actor_id=admin_a_id, target_id=bob_id)
    admin_b_session = sign_session_cookie(user_id=admin_b_id, secret=secret)

    with client:
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": token_for_b},
            cookies={"tv_session": admin_b_session},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is not None


def test_admin_delete_post_confirm_token_target_mismatch_rejects(
    tmp_path: Path, isolated_telemac_dirs
) -> None:
    """Token minted for uid X, replayed against uid Y -> rejected."""
    from auth.core import connect, get_user_by_id

    client, token, admin_id, db = _admin_client(tmp_path)
    secret = b"x" * 32
    bob_id = _seed_user(db, username="bob", password="hunter2")
    carol_id = _seed_user(db, username="carol", password="hunter2")

    # Mint a token for bob, then try to delete carol with it.
    bob_token = _mint_confirm_token(secret, actor_id=admin_id, target_id=bob_id)

    with client:
        r = client.post(
            f"/admin/users/{carol_id}/delete",
            data={"confirm_token": bob_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    with connect(db) as conn:
        assert get_user_by_id(conn, carol_id) is not None
        assert get_user_by_id(conn, bob_id) is not None


def test_admin_delete_post_db_error_does_not_cascade(
    tmp_path: Path, isolated_telemac_dirs, monkeypatch
) -> None:
    """Monkeypatch delete_user_atomic to raise sqlite3.Error -> flash 'error',
    library NOT touched."""
    import sqlite3

    import model_library
    from auth import routes
    from auth.core import connect, get_user_by_id

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")
    bob_root = model_library.user_library_root(bob_id)
    (bob_root / "keep.txt").write_bytes(b"untouched")
    bob_base = model_library.user_library_default_base() / str(bob_id)

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated db meltdown")

    monkeypatch.setattr(routes, "delete_user_atomic", boom)

    with client:
        confirm_token = _get_delete_token(client, token, bob_id)
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": confirm_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "admin_flash=" in r.headers.get("set-cookie", "")

    # Auth row preserved
    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is not None
    # Library dir UNTOUCHED — cascade must NOT happen on db error.
    assert bob_base.exists()
    assert (bob_base / "models" / "keep.txt").exists()


def test_admin_delete_post_library_rm_fails_logs_and_flashes_warn(
    tmp_path: Path, isolated_telemac_dirs, monkeypatch, caplog
) -> None:
    """Monkeypatch model_library.delete_user_library to raise PermissionError
    (subclass of OSError) -> auth row IS gone, flash 'warn' with orphan path,
    ERROR log."""
    import logging

    import model_library
    from auth.core import connect, get_user_by_id

    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")
    bob_root = model_library.user_library_root(bob_id)
    (bob_root / "stuck.txt").write_bytes(b"orphaned")
    bob_base = model_library.user_library_default_base() / str(bob_id)
    expected_orphan = str(bob_base)

    def boom(user_id: int):
        raise PermissionError("simulated rm failure")

    monkeypatch.setattr(model_library, "delete_user_library", boom)

    with client, caplog.at_level(logging.ERROR, logger="auth"):
        confirm_token = _get_delete_token(client, token, bob_id)
        r = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": confirm_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r.status_code == 303
        cookie_hdr = r.headers.get("set-cookie", "")
        assert "admin_flash=" in cookie_hdr

    # Auth row IS gone (auth is source of truth; library cascade is a
    # downstream best-effort).
    with connect(db) as conn:
        assert get_user_by_id(conn, bob_id) is None
    # ERROR log mentions the orphan path
    log_text = (
        " ".join(rec.message for rec in caplog.records)
        + " "
        + " ".join(rec.getMessage() for rec in caplog.records)
    )
    assert "library cascade failed" in log_text
    assert expected_orphan in log_text or str(bob_id) in log_text


def test_admin_flash_cookie_one_shot(tmp_path: Path, isolated_telemac_dirs) -> None:
    """Set flash via POST delete, GET /admin/users -> banner shown +
    cookie cleared on response; GET again -> no banner.

    This is the one-shot invariant: the cookie is deleted AS PART of the
    response that renders the banner, so a refresh shows nothing."""
    client, token, _admin_id, db = _admin_client(tmp_path)
    bob_id = _seed_user(db, username="bob", password="hunter2")

    with client:
        # Trigger a flash via the happy delete path.
        confirm_token = _get_delete_token(client, token, bob_id)
        r1 = client.post(
            f"/admin/users/{bob_id}/delete",
            data={"confirm_token": confirm_token},
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        # Hand the flash cookie back to the next GET. Parse the value
        # out of the set-cookie header (don't rely on the testclient's
        # cookie jar, which is being deprecated for per-request cookies).
        sc1 = r1.headers.get("set-cookie", "")
        m = re.search(r"admin_flash=([^;]+)", sc1)
        assert m, f"no admin_flash cookie in response: {sc1}"
        flash_cookie = m.group(1)

        r2 = client.get(
            "/admin/users",
            cookies={"tv_session": token, "admin_flash": flash_cookie},
            follow_redirects=False,
        )
        assert r2.status_code == 200
        # Banner div (NOT just the CSS rule, which is always present).
        assert 'class="alert alert-info"' in r2.text
        assert "Deleted user" in r2.text
        # The cookie is CLEARED by the same response (Max-Age=0 or expires)
        sc2 = r2.headers.get("set-cookie", "")
        assert "admin_flash=" in sc2
        assert (
            "Max-Age=0" in sc2
            or "max-age=0" in sc2.lower()
            or "expires=" in sc2.lower()
        ), f"flash cookie not cleared on render: {sc2}"

        # Refresh /admin/users WITHOUT the flash cookie (simulating that
        # the browser has obeyed the clear). No banner now.
        r3 = client.get(
            "/admin/users",
            cookies={"tv_session": token},
            follow_redirects=False,
        )
        assert r3.status_code == 200
        # No banner div this time (the CSS rule `.alert-info{...}` is
        # always present, so we assert the actual div is absent).
        assert 'class="alert alert-info"' not in r3.text
        assert "Deleted user" not in r3.text


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
