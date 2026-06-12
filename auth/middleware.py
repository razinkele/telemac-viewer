"""ASGI middleware: cookie validation, scope writes, fail-closed on DB error.

Spec sections: 3.1, 3.2, 3.3, 3.4, 3.5, 7.1, 2.1.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import re
import sqlite3
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from auth.core import DEFAULT_DB_PATH, User, connect, get_user_by_id
from auth.crypto import decode_session_cookie

logger = logging.getLogger("auth")

# Public routes that ALWAYS short-circuit auth (spec §7.1 step 2)
_PUBLIC_PREFIXES = ("/login", "/logout", "/static", "/favicon.ico")


def _is_public(path: str) -> bool:
    return any(
        path == p or path.startswith(p + "/") or path == p for p in _PUBLIC_PREFIXES
    ) or path.startswith("/static/")


# Browser-facing mount prefix (e.g. "/telemac" behind nginx+Shiny Server).
# Strict charset: the header is attacker-controllable when nginx does NOT
# set it, and an unvalidated value would turn the login redirect into an
# open redirect (X-Forwarded-Prefix: https://evil → Location: https://evil/login).
_PREFIX_RE = re.compile(r"^/[A-Za-z0-9_\-./]{0,128}$")


def get_url_prefix(scope: dict) -> str:
    """Return the validated X-Forwarded-Prefix header value, or "".

    Shiny Server strips the mount path before the ASGI app sees the
    request and sets no root_path, so the public prefix can only come
    from the reverse proxy. Empty string when absent (direct/dev runs).
    """
    raw = ""
    for k, v in scope.get("headers", []):
        if k == b"x-forwarded-prefix":
            raw = v.decode("latin-1").strip()
            break
    raw = raw.rstrip("/")
    if not raw:
        return ""
    if not _PREFIX_RE.fullmatch(raw) or "//" in raw or ".." in raw:
        logger.warning("Rejected malformed X-Forwarded-Prefix: %r", raw[:64])
        return ""
    return raw


def auth_middleware(
    app,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    secret: bytes,
):
    """ASGI function middleware. Wraps `app` with cookie auth.

    `secret` is captured at construction. The middleware runs BEFORE
    Starlette sets `scope["app"]`, so it can NOT read the secret from
    `scope["app"].state` — that field is only populated when Starlette's
    inner __call__ fires, which happens after the middleware decoded the
    cookie. (Earlier revisions of this plan tried that pattern and broke
    auth on every restart.) Caller (app.py) loads the secret eagerly at
    module import and passes it here.

    Sets `scope["user_id"]: int | None` on the way through. On HTTP /
    WebSocket scope types, decodes the tv_session cookie and validates
    the user_id in DB. Public routes short-circuit before DB access.
    """
    # Once-per-process flag for the deployment-context warning (§2.1).
    _bind_warned = [False]

    async def middleware(scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            # lifespan etc. — pass through
            await app(scope, receive, send)
            return

        # Deployment-context warning, fired once per process on the first
        # http/websocket request when scope["server"] reveals the bind.
        if not _bind_warned[0]:
            server = scope.get("server")
            if server:
                warn_if_public_bind(server[0])
            _bind_warned[0] = True

        # Browser-facing mount prefix — set for ALL http/ws requests
        # (public login/logout routes need it to build form actions).
        prefix = get_url_prefix(scope)
        scope["url_prefix"] = prefix

        path = scope.get("path", "")
        if scope["type"] == "http" and _is_public(path):
            await app(scope, receive, send)
            return

        # Decode cookie if present
        cookie_header = ""
        for k, v in scope.get("headers", []):
            if k == b"cookie":
                cookie_header = v.decode("latin-1")
                break

        token = _extract_cookie(cookie_header, "tv_session")
        user_id: int | None = None
        if token:
            payload = decode_session_cookie(token, secret=secret)
            # Validate payload shape BEFORE touching the DB. A signed cookie
            # with the wrong shape (forged after secret compromise, or an
            # old-format cookie after a future schema change) must not be
            # treated as authenticated AND must not crash with KeyError /
            # sqlite3.InterfaceError leaking as a 500.
            uid_candidate = (
                payload.get("user_id") if isinstance(payload, dict) else None
            )
            if isinstance(uid_candidate, int):
                try:
                    with connect(db_path) as conn:
                        user = get_user_by_id(conn, uid_candidate)
                except sqlite3.Error as e:
                    # Broad sqlite3.Error (parent of OperationalError,
                    # InterfaceError, etc.) — fail-closed.
                    logger.error("DB unreachable in auth_middleware: %s", e)
                    if scope["type"] == "http":
                        await PlainTextResponse("Service Unavailable", status_code=503)(
                            scope, receive, send
                        )
                    else:
                        # WebSocket: close with 1011 (internal error)
                        await send({"type": "websocket.close", "code": 1011})
                    return
                if user is not None:
                    user_id = user.id

        scope["user_id"] = user_id

        if user_id is None and scope["type"] == "http":
            # Unauthenticated → redirect to <prefix>/login with next param.
            # `next` carries the BROWSER-facing path (prefix included) so
            # login_post can redirect to it verbatim after auth. quote()
            # keeps `?` / `&` in the original query string from being
            # parsed as separators of the /login URL itself.
            next_url = prefix + path
            if scope.get("query_string"):
                next_url += "?" + scope["query_string"].decode("latin-1")
            await RedirectResponse(
                url=f"{prefix}/login?next={quote(next_url, safe='/')}",
                status_code=302,
            )(scope, receive, send)
            return

        if user_id is None and scope["type"] == "websocket":
            await send(
                {"type": "websocket.close", "code": 4401}
            )  # custom 4401 = unauthorized
            return

        await app(scope, receive, send)

    return middleware


def _extract_cookie(header: str, name: str) -> str | None:
    """Tiny cookie parser — avoids pulling in http.cookies for one value."""
    for chunk in header.split(";"):
        chunk = chunk.strip()
        if chunk.startswith(name + "="):
            return chunk[len(name) + 1 :]
    return None


# --- Accessors ---


def get_current_user_from_request(request: Request) -> User | None:
    uid = request.scope.get("user_id")
    if uid is None:
        return None
    # Read from app.state (set by lifespan or by tests' `_wrap` helper),
    # not from scope. Middleware writes only `user_id` into scope; the
    # DB path lives on app.state.auth_db_path. Sibling helper
    # `_get_db_path` in routes.py reads the same attribute.
    db_path = getattr(request.app.state, "auth_db_path", DEFAULT_DB_PATH)
    with connect(db_path) as conn:
        return get_user_by_id(conn, uid)


def get_current_user_id_from_scope(scope: dict) -> int | None:
    return scope.get("user_id")


def require_admin(request: Request) -> User:
    user = get_current_user_from_request(request)
    if user is None or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# --- Route-error decorator ---


def handle_route_errors(
    handler: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Wrap an auth route handler. Logs uncaught exceptions and returns
    a generic 500 response. MUST NOT log request body or form fields
    (passwords would leak).
    """

    @functools.wraps(handler)
    async def wrapper(request: Request) -> Response:
        try:
            return await handler(request)
        except HTTPException:
            raise  # pass through (Starlette handles the response)
        except Exception:
            uid = request.scope.get("user_id")
            logger.error(
                "Unhandled exception in route %s %s (user_id=%s)",
                request.method,
                request.url.path,
                uid,
                exc_info=True,
            )
            return PlainTextResponse(
                "An internal error occurred — see server logs.",
                status_code=500,
            )

    return wrapper


# --- Deployment-context guard (§2.1) ---


def warn_if_public_bind(host: str) -> None:
    """Emit a WARNING if the app is bound to a non-private address.

    Called from app.py at startup, BEFORE wrapping the Shiny app, with
    the host string from app.py's own argv parsing. The middleware
    itself doesn't see uvicorn's --host.
    """
    if host == "0.0.0.0":
        _emit_bind_warning(host)
        return
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return  # hostname, not address — skip
    if addr.is_loopback or addr.is_private:
        return
    _emit_bind_warning(host)


def _emit_bind_warning(host: str) -> None:
    logger.warning(
        "Auth waivers (CSRF, rate-limit) assume private deploy. "
        "Detected bind=%s; verify the host is not publicly reachable, "
        "or add a reverse-proxy ACL.",
        host,
    )
