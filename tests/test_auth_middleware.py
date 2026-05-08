"""Tests for auth.middleware — ASGI wrapping, accessors, decorators."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _build_app(tmp_db: Path, secret: bytes = b"x" * 32):
    """Build a Starlette app wrapped in auth_middleware for testing."""
    from auth.core import connect, create_user, ensure_schema
    from auth.crypto import hash_password
    from auth.middleware import auth_middleware

    # Seed a user so we have someone to authenticate as
    with connect(tmp_db) as conn:
        ensure_schema(conn)
        create_user(conn, username="alice", password_hash=hash_password("hunter2"))

    async def public(_r):
        return PlainTextResponse("public")

    async def gated(request):
        uid = request.scope.get("user_id")
        return PlainTextResponse(f"hi {uid}")

    inner = Starlette(
        routes=[
            Route("/login", public),  # public route per spec §7.1
            Route("/static/x.css", public),
            Route("/", gated),
        ]
    )
    return auth_middleware(inner, db_path=tmp_db, secret=secret)


def test_unauthenticated_request_redirects_to_login(tmp_path: Path) -> None:
    app = _build_app(tmp_path / "auth.db")
    with TestClient(app) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].startswith("/login")


def test_public_route_passes_through_with_malformed_cookie(tmp_path: Path) -> None:
    app = _build_app(tmp_path / "auth.db")
    with TestClient(app) as client:
        # Public-route check MUST happen before cookie decode (spec §7.1 step 2)
        r = client.get("/login", cookies={"tv_session": "garbage"})
        assert r.status_code == 200
        assert r.text == "public"


def test_authenticated_request_passes_through(tmp_path: Path) -> None:
    from auth.crypto import sign_session_cookie

    db = tmp_path / "auth.db"
    secret = b"x" * 32
    app = _build_app(db, secret=secret)
    token = sign_session_cookie(user_id=1, secret=secret)
    with TestClient(app) as client:
        r = client.get("/", cookies={"tv_session": token}, follow_redirects=False)
        assert r.status_code == 200
        assert r.text == "hi 1"


def test_db_unavailable_returns_503(tmp_path: Path, caplog) -> None:
    """Spec §3.5: fail-closed on DB error — never pass-through.

    Two patch points cover both real failure modes the middleware can
    encounter: get_user_by_id raising (sqlite read fails after connect)
    AND connect() itself raising (sqlite file lock, disk full, missing
    parent dir). Without testing both, a refactor that moves the failure
    earlier in the middleware would silently break fail-closed.
    """
    from auth.crypto import sign_session_cookie

    db = tmp_path / "auth.db"
    secret = b"x" * 32
    app = _build_app(db, secret=secret)
    token = sign_session_cookie(user_id=1, secret=secret)

    # Variant 1: get_user_by_id raises mid-request
    with patch(
        "auth.middleware.get_user_by_id", side_effect=sqlite3.OperationalError("locked")
    ):
        with TestClient(app) as client:
            r = client.get("/", cookies={"tv_session": token}, follow_redirects=False)
            assert r.status_code == 503, (
                "Middleware did not fail closed on DB read error"
            )

    # Variant 2: connect() itself raises before any read
    with patch(
        "auth.middleware.connect", side_effect=sqlite3.OperationalError("disk I/O")
    ):
        with TestClient(app) as client:
            r = client.get("/", cookies={"tv_session": token}, follow_redirects=False)
            assert r.status_code == 503, (
                "Middleware did not fail closed on connect() error"
            )


def test_warn_if_public_bind_logs_warning_for_0_0_0_0(caplog) -> None:
    from auth.middleware import warn_if_public_bind

    with caplog.at_level(logging.WARNING, logger="auth"):
        warn_if_public_bind("0.0.0.0")
    assert any("publicly reachable" in r.message for r in caplog.records)


def test_warn_if_public_bind_silent_for_loopback(caplog) -> None:
    from auth.middleware import warn_if_public_bind

    with caplog.at_level(logging.WARNING, logger="auth"):
        warn_if_public_bind("127.0.0.1")
    assert not any("publicly reachable" in r.message for r in caplog.records)


def test_warn_if_public_bind_silent_for_rfc1918(caplog) -> None:
    from auth.middleware import warn_if_public_bind

    with caplog.at_level(logging.WARNING, logger="auth"):
        warn_if_public_bind("192.168.1.10")
        warn_if_public_bind("10.0.0.5")
    assert not any("publicly reachable" in r.message for r in caplog.records)


def test_websocket_upgrade_with_valid_cookie(tmp_path: Path) -> None:
    """Spec §3.1: WebSocket handshake goes through the same auth path."""
    from starlette.routing import WebSocketRoute
    from auth.core import connect, create_user, ensure_schema
    from auth.crypto import hash_password, sign_session_cookie
    from auth.middleware import auth_middleware

    db = tmp_path / "auth.db"
    with connect(db) as conn:
        ensure_schema(conn)
        create_user(conn, username="alice", password_hash=hash_password("p"))

    async def ws_endpoint(websocket):
        await websocket.accept()
        await websocket.send_text(f"uid:{websocket.scope.get('user_id')}")
        await websocket.close()

    inner = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    secret = b"x" * 32
    app = auth_middleware(inner, db_path=db, secret=secret)
    token = sign_session_cookie(user_id=1, secret=secret)

    with TestClient(app) as client:
        with client.websocket_connect("/ws", cookies={"tv_session": token}) as ws:
            assert ws.receive_text() == "uid:1"


def test_websocket_upgrade_without_cookie_rejected(tmp_path: Path) -> None:
    """Spec §3.1: unauth'd WebSocket gets close code 4401.

    Asserts the SPECIFIC code so a future change that drops to e.g.
    code=1000 (normal close) wouldn't quietly let the test pass.
    """
    from starlette.routing import WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    from auth.core import connect, ensure_schema
    from auth.middleware import auth_middleware

    db = tmp_path / "auth.db"
    with connect(db) as conn:
        ensure_schema(conn)

    async def ws_endpoint(websocket):
        await websocket.accept()

    inner = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    app = auth_middleware(inner, db_path=db, secret=b"x" * 32)

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws"):
                pass
        assert excinfo.value.code == 4401, (
            f"Expected close code 4401 (unauthorized) but got {excinfo.value.code}"
        )
