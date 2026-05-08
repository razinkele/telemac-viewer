"""HTTP routes: login, logout, admin user management.

Spec §7. Inline Jinja2 templates with autoescape=True — no templates/ dir.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from jinja2 import Environment, BaseLoader, select_autoescape
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

from auth.core import (
    DEFAULT_DB_PATH,
    connect,
    create_user,
    delete_user_atomic,
    ensure_schema,
    get_user_by_id,
    get_user_by_username,
    list_users,
    update_password_hash,
    update_user,
)
from auth.crypto import (
    NULL_HASH,
    hash_password,
    load_or_create_secret,
    sign_session_cookie,
    verify_password,
)
from auth.middleware import handle_route_errors, require_admin

logger = logging.getLogger("auth")

# --- Jinja2 setup with autoescape ---

_jinja = Environment(
    loader=BaseLoader(),
    autoescape=select_autoescape(default=True),
)

# --- Templates ---

LOGIN_HTML = _jinja.from_string("""
<!doctype html>
<html><head><title>Sign in — TELEMAC Viewer</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f4f8;
  display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{background:#fff;padding:2rem;border-radius:.5rem;box-shadow:0 2px 8px rgba(0,0,0,.1);
  width:320px}
h1{margin-top:0;color:#0a3d62;font-size:1.25rem}
input{display:block;width:100%;padding:.5rem;margin:.5rem 0;border:1px solid #ccc;
  border-radius:.25rem;box-sizing:border-box}
button{background:#0a3d62;color:#fff;border:0;padding:.6rem 1rem;border-radius:.25rem;
  cursor:pointer;width:100%}
.err{color:#c0392b;margin:.5rem 0;font-size:.9rem}
.info{color:#2c7;margin:.5rem 0;font-size:.9rem}
</style></head><body>
<form class="card" method="post" action="/login{% if next %}?next={{ next }}{% endif %}">
  <h1>TELEMAC Viewer</h1>
  {% if error %}<div class="err">Invalid username or password.</div>{% endif %}
  {% if logged_out %}<div class="info">Logged out.</div>{% endif %}
  <input name="username" placeholder="Username" autofocus required>
  <input name="password" type="password" placeholder="Password" required>
  <button type="submit">Sign in</button>
</form></body></html>
""")


# --- Helpers ---

# Defense-in-depth: only printable, no control chars (rejects `\r\n` for
# header injection), only relative paths starting with `/`, no `//` or
# `/\` (browsers normalize `\` to `/`), and a length cap so an attacker
# can't stuff a 10MB next-value through the validator.
_SAFE_NEXT_RE = re.compile(r"^/(?![/\\])[^\x00-\x1f\x7f]*$")
_NEXT_MAX_LEN = 256


def _safe_next(raw: str | None) -> str:
    """Validate the `next` query param — only relative paths are allowed.

    Blocks:
      next=https://evil/      (absolute URL with scheme)
      next=//evil/            (protocol-relative URL)
      next=/\evil/            (browser-normalized to //evil/)
      next=javascript:alert() (no leading slash)
      next=/\r\nLocation:...  (CRLF / control chars)
      next=/<10MB junk>       (length cap)
    URL-encoded variants are decoded by Starlette before this check so
    `next=%2F%2Fevil` is decoded to `//evil` and rejected; `%5C` decodes
    to `\` and is rejected; `%0D%0A` decodes to CRLF and is rejected.
    """
    if raw and len(raw) <= _NEXT_MAX_LEN and _SAFE_NEXT_RE.fullmatch(raw):
        return raw
    return "/"


def _set_session_cookie(response, user_id: int, secret: bytes, secure: bool) -> None:
    token = sign_session_cookie(user_id=user_id, secret=secret)
    response.set_cookie(
        "tv_session",
        token,
        max_age=30 * 86400,
        path="/",
        httponly=True,
        samesite="lax",
        secure=secure,
    )


def _clear_session_cookie(response, secure: bool) -> None:
    response.delete_cookie(
        "tv_session",
        path="/",
        httponly=True,
        samesite="lax",
        secure=secure,
    )


def _get_secret(request: Request) -> bytes:
    """Stash the secret on the app state at startup; falls back to disk read."""
    secret = getattr(request.app.state, "auth_secret", None)
    if secret is None:
        secret_path = getattr(
            request.app.state,
            "auth_secret_path",
            DEFAULT_DB_PATH.parent / "auth_secret",
        )
        secret = load_or_create_secret(secret_path)
        request.app.state.auth_secret = secret
    return secret


def _get_db_path(request: Request) -> Path:
    return getattr(request.app.state, "auth_db_path", DEFAULT_DB_PATH)


# --- Routes ---


@handle_route_errors
async def login_get(request: Request) -> HTMLResponse:
    body = LOGIN_HTML.render(
        next=request.query_params.get("next", ""),
        error=request.query_params.get("error") == "invalid",
        logged_out=request.query_params.get("logged_out") == "1",
    )
    response = HTMLResponse(body)
    # If the visitor brought a malformed cookie, clear it (spec §7.1 step 3)
    if request.cookies.get("tv_session"):
        from auth.crypto import decode_session_cookie

        if (
            decode_session_cookie(
                request.cookies["tv_session"],
                secret=_get_secret(request),
            )
            is None
        ):
            _clear_session_cookie(response, secure=request.url.scheme == "https")
    return response


@handle_route_errors
async def login_post(request: Request) -> RedirectResponse:
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""

    db_path = _get_db_path(request)
    with connect(db_path) as conn:
        user = get_user_by_username(conn, username)

    # Always run bcrypt verify (against NULL_HASH if user not found) for
    # constant-time response. Spec §7.1 step 4.
    if user is None:
        verify_password(password, NULL_HASH)
        ok = False
    else:
        ok = verify_password(password, user.password_hash)

    if not ok:
        # Truncate the username in logs to 32 chars: a user who types their
        # password into the username field by accident otherwise leaks the
        # password into the journal.
        logger.warning("Failed login for username=%r", username[:32])
        return RedirectResponse("/login?error=invalid", status_code=302)

    logger.info("Login success: user_id=%s username=%s", user.id, user.username)
    target = _safe_next(request.query_params.get("next"))
    response = RedirectResponse(target, status_code=302)
    _set_session_cookie(
        response,
        user_id=user.id,
        secret=_get_secret(request),
        secure=request.url.scheme == "https",
    )
    return response


@handle_route_errors
async def logout_post(request: Request) -> RedirectResponse:
    uid = request.scope.get("user_id")
    logger.info("Logout: user_id=%s", uid)
    response = RedirectResponse("/login?logged_out=1", status_code=302)
    _clear_session_cookie(response, secure=request.url.scheme == "https")
    return response


# --- Route table (admin routes added in Task 9) ---

auth_routes = [
    Route("/login", login_get, methods=["GET"]),
    Route("/login", login_post, methods=["POST"]),
    Route("/logout", logout_post, methods=["POST"]),
]
