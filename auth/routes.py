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


# --- Admin templates ---

ADMIN_USERS_HTML = _jinja.from_string("""
<!doctype html><html><head><title>Admin — Users</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
table{width:100%;border-collapse:collapse}
th,td{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}
.actions form{display:inline}
button{background:#0a3d62;color:#fff;border:0;padding:.3rem .6rem;border-radius:.2rem;
  cursor:pointer;margin:.1rem}
.danger{background:#c0392b}
.create{margin:1rem 0;padding:1rem;background:#f6f8fa;border-radius:.3rem}
.create input,.create label{margin:.25rem .5rem .25rem 0}
.err{color:#c0392b;background:#fdecea;padding:.5rem;border-radius:.3rem;margin:.5rem 0}
</style></head><body>
<h1>Users</h1>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<form class="create" method="post" action="/admin/users/create">
  <h3>Create user</h3>
  <input name="username" placeholder="username" required>
  <input name="display_name" placeholder="display name (optional)">
  <input name="password" type="password" placeholder="password (≥8 chars, ≤72 bytes)" required>
  <label><input type="checkbox" name="is_admin"> admin</label>
  <button type="submit">Create</button>
</form>
<table><tr><th>id</th><th>username</th><th>display name</th><th>admin?</th>
  <th>created</th><th>actions</th></tr>
{% for u in users %}
<tr>
  <td>{{ u.id }}</td>
  <td>{{ u.username }}</td>
  <td>{{ u.display_name or "" }}</td>
  <td>{{ "yes" if u.is_admin else "" }}</td>
  <td>{{ u.created_at }}</td>
  <td class="actions">
    <form method="post" action="/admin/users/{{ u.id }}/edit" style="display:inline">
      <input name="display_name" placeholder="new display name" value="{{ u.display_name or '' }}" style="width:8rem">
      <input type="checkbox" name="is_admin"{% if u.is_admin %} checked{% endif %}>admin
      <button type="submit">Save</button>
    </form>
    <form method="post" action="/admin/users/{{ u.id }}/reset-password" style="display:inline">
      <input name="password" type="password" placeholder="new password" style="width:8rem">
      <button type="submit">Reset PW</button>
    </form>
    <form method="post" action="/admin/users/{{ u.id }}/delete" style="display:inline"
          onsubmit="return confirm('Delete {{ u.username }}?')">
      <button type="submit" class="danger">Delete</button>
    </form>
  </td>
</tr>
{% endfor %}
</table>
<form method="post" action="/logout"><button type="submit">Log out</button></form>
</body></html>
""")


# --- Validation helpers ---

_USERNAME_RE = re.compile(r"^[\w.\-]{1,64}$")
_DISPLAY_RE = re.compile(r"^[\w .\-]{1,64}$")


def _validate_password(pw: str) -> str | None:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if len(pw.encode("utf-8")) > 72:
        return "Password must be at most 72 UTF-8 bytes."
    return None


def _validate_username(name: str) -> str | None:
    if not _USERNAME_RE.match(name):
        return "Username must match [a-zA-Z0-9_.-], 1–64 chars."
    return None


def _render_users_page(
    db_path: Path, error: str | None = None, status: int = 200
) -> HTMLResponse:
    with connect(db_path) as conn:
        users = list_users(conn)
    return HTMLResponse(
        ADMIN_USERS_HTML.render(users=users, error=error),
        status_code=status,
    )


# --- Admin handlers ---


@handle_route_errors
async def admin_users_get(request: Request) -> HTMLResponse:
    require_admin(request)
    return _render_users_page(_get_db_path(request))


@handle_route_errors
async def admin_users_create(request: Request) -> HTMLResponse | RedirectResponse:
    require_admin(request)
    form = await request.form()
    username = (form.get("username") or "").strip()
    display_name = (form.get("display_name") or "").strip() or None
    password = form.get("password") or ""
    is_admin = form.get("is_admin") in ("on", "true", "1")

    if err := _validate_username(username):
        return _render_users_page(_get_db_path(request), error=err, status=400)
    if display_name and not _DISPLAY_RE.match(display_name):
        return _render_users_page(
            _get_db_path(request),
            error="Display name must match [a-zA-Z0-9_ .-], 1–64 chars.",
            status=400,
        )
    if err := _validate_password(password):
        return _render_users_page(_get_db_path(request), error=err, status=400)

    db = _get_db_path(request)
    with connect(db) as conn:
        # Pre-check (defense in depth; the INSERT is wrapped too)
        if get_user_by_username(conn, username) is not None:
            return _render_users_page(
                db,
                error=f"Username {username!r} already exists.",
                status=400,
            )
        try:
            create_user(
                conn,
                username=username,
                password_hash=hash_password(password),
                display_name=display_name,
                is_admin=is_admin,
            )
        except sqlite3.IntegrityError:
            return _render_users_page(
                db,
                error=f"Username {username!r} already exists.",
                status=400,
            )
    logger.info("Admin created user: username=%s is_admin=%s", username, is_admin)
    return RedirectResponse("/admin/users", status_code=302)


@handle_route_errors
async def admin_users_edit(request: Request) -> RedirectResponse | HTMLResponse:
    require_admin(request)
    user_id = int(request.path_params["user_id"])
    form = await request.form()
    display_name = (form.get("display_name") or "").strip() or None
    is_admin = form.get("is_admin") in ("on", "true", "1")
    if display_name and not _DISPLAY_RE.match(display_name):
        return _render_users_page(
            _get_db_path(request),
            error="Display name must match [a-zA-Z0-9_ .-], 1–64 chars.",
            status=400,
        )
    with connect(_get_db_path(request)) as conn:
        rc = update_user(
            conn, user_id=user_id, display_name=display_name, is_admin=is_admin
        )
    if rc == 0:
        # User row was deleted by another admin between page load and form submit.
        # Surface this rather than redirecting with a misleading "saved" log line.
        return _render_users_page(
            _get_db_path(request),
            error=f"User id={user_id} no longer exists; refresh the page.",
            status=400,
        )
    logger.info("Admin edited user_id=%s", user_id)
    return RedirectResponse("/admin/users", status_code=302)


@handle_route_errors
async def admin_users_reset_password(
    request: Request,
) -> RedirectResponse | HTMLResponse:
    require_admin(request)
    user_id = int(request.path_params["user_id"])
    form = await request.form()
    password = form.get("password") or ""
    if err := _validate_password(password):
        return _render_users_page(_get_db_path(request), error=err, status=400)
    with connect(_get_db_path(request)) as conn:
        rc = update_password_hash(
            conn, user_id=user_id, password_hash=hash_password(password)
        )
    if rc == 0:
        # Worse-than-edit race: silently "succeeding" here would let the
        # admin believe the user can now log in with the new password, but
        # the user row is gone. Surface explicitly.
        return _render_users_page(
            _get_db_path(request),
            error=f"User id={user_id} no longer exists; cannot reset password.",
            status=400,
        )
    logger.info("Admin reset password for user_id=%s", user_id)
    return RedirectResponse("/admin/users", status_code=302)


@handle_route_errors
async def admin_users_delete(request: Request) -> RedirectResponse | HTMLResponse:
    require_admin(request)
    user_id = int(request.path_params["user_id"])
    db_path = _get_db_path(request)
    with connect(db_path) as conn:
        # Distinguish "last admin" from "already deleted" so the error
        # message is honest instead of always saying "last admin".
        target = get_user_by_id(conn, user_id)
        if target is None:
            return _render_users_page(
                db_path,
                error=f"User id={user_id} no longer exists.",
                status=400,
            )
        deleted = delete_user_atomic(conn, user_id=user_id)
    if not deleted:
        # Atomic guard rejected the delete because target was the last admin
        # (target existed in the SELECT above; only path to deleted=False).
        return _render_users_page(
            db_path,
            error="Refusing to delete the last admin.",
            status=400,
        )
    logger.info("Admin deleted user_id=%s", user_id)
    return RedirectResponse("/admin/users", status_code=302)


# --- Route table ---

auth_routes = [
    Route("/login", login_get, methods=["GET"]),
    Route("/login", login_post, methods=["POST"]),
    Route("/logout", logout_post, methods=["POST"]),
    Route("/admin/users", admin_users_get, methods=["GET"]),
    Route("/admin/users/create", admin_users_create, methods=["POST"]),
    Route("/admin/users/{user_id:int}/edit", admin_users_edit, methods=["POST"]),
    Route(
        "/admin/users/{user_id:int}/reset-password",
        admin_users_reset_password,
        methods=["POST"],
    ),
    Route("/admin/users/{user_id:int}/delete", admin_users_delete, methods=["POST"]),
]
