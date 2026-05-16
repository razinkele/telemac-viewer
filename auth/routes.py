"""HTTP routes: login, logout, admin user management.

Spec §7. Inline Jinja2 templates with autoescape=True — no templates/ dir.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Literal

import itsdangerous
from jinja2 import Environment, BaseLoader, select_autoescape
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse
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


# --- Admin flash + confirm-token helpers (Task 10) ---


def _admin_actor_id(request: Request) -> int:
    """Return the authenticated admin's user_id from the request scope.

    Safe to call only after require_admin has run on the request.
    """
    uid = request.scope.get("user_id")
    if not isinstance(uid, int) or uid <= 0:
        raise RuntimeError("admin route reached without scope user_id")
    return uid


def _flash(
    request: Request,
    response,
    level: Literal["info", "warn", "error"],
    message: str,
) -> None:
    """Set a one-shot signed flash cookie. Read + cleared by the next
    /admin/users render via _read_flash(request, response).

    Takes ``request`` because the secret is loaded per-request via
    ``_get_secret(request)`` — there is no module-level secret constant.
    """
    payload = itsdangerous.URLSafeTimedSerializer(
        _get_secret(request),
        salt="admin-flash",
    ).dumps({"level": level, "message": message})
    response.set_cookie(
        "admin_flash",
        payload,
        max_age=60,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/admin",
    )


def _read_flash(request: Request, response) -> dict | None:
    """Read + clear the one-shot admin_flash cookie.

    Returns the decoded {"level", "message"} payload, or None if the
    cookie is missing, expired, or has a bad signature. In all
    cookie-present cases, instructs the browser to clear the cookie so
    the banner shows exactly once.
    """
    raw = request.cookies.get("admin_flash")
    if not raw:
        return None
    response.delete_cookie("admin_flash", path="/admin")
    try:
        return itsdangerous.URLSafeTimedSerializer(
            _get_secret(request),
            salt="admin-flash",
        ).loads(raw, max_age=60)
    except itsdangerous.SignatureExpired:
        logger.info("admin flash cookie expired — clearing")
        return None
    except itsdangerous.BadSignature:
        logger.warning("admin flash cookie bad signature — clearing")
        return None


def _redirect_with_flash(
    request: Request,
    level: Literal["info", "warn", "error"],
    message: str,
) -> RedirectResponse:
    """Construct the /admin/users redirect AND attach the signed flash cookie.

    Folds the response-construct + cookie-mutate pattern into one call so
    every branch in admin_users_delete_post returns one of these.
    """
    response = RedirectResponse("/admin/users", status_code=303)
    _flash(request, response, level, message)
    return response


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
.alert{padding:.5rem;border-radius:.3rem;margin:.5rem 0}
.alert-info{color:#0a3d62;background:#e8f4ff}
.alert-warn{color:#8a6d3b;background:#fcf2dc}
.alert-error{color:#c0392b;background:#fdecea}
.del-link{color:#c0392b;text-decoration:none;padding:.3rem .6rem;
  border:1px solid #c0392b;border-radius:.2rem;display:inline-block;margin:.1rem}
</style></head><body>
<h1>Users</h1>
{% if flash %}<div class="alert alert-{{ flash.level }}">{{ flash.message }}</div>{% endif %}
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
    <a class="del-link" href="/admin/users/{{ u.id }}/delete">Delete</a>
  </td>
</tr>
{% endfor %}
</table>
<form method="post" action="/logout"><button type="submit">Log out</button></form>
</body></html>
""")


ADMIN_DELETE_CONFIRM_HTML = _jinja.from_string("""
<!doctype html>
<html><head><title>Delete user — TELEMAC Viewer</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}
h1{color:#c0392b}
pre{background:#f6f8fa;padding:.5rem;border-radius:.3rem;overflow-x:auto}
button{background:#c0392b;color:#fff;border:0;padding:.5rem 1rem;
  border-radius:.2rem;cursor:pointer}
a.cancel{margin-left:1rem;color:#0a3d62}
</style></head><body>
<h1>Delete user &lsquo;{{ user.username }}&rsquo;?</h1>
<p>This will permanently delete the user account AND
  {{ usage.files }} file(s) / {{ usage.size_human }} from:</p>
<pre><code>{{ resolved_path }}</code></pre>
<p><strong>This action cannot be undone.</strong></p>
<form method="post" action="/admin/users/{{ user.id }}/delete">
  <input type="hidden" name="confirm_token" value="{{ token }}">
  <button type="submit">Delete user + files</button>
  <a class="cancel" href="/admin/users">Cancel</a>
</form>
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
    db_path: Path,
    error: str | None = None,
    status: int = 200,
    flash: dict | None = None,
) -> HTMLResponse:
    with connect(db_path) as conn:
        users = list_users(conn)
    return HTMLResponse(
        ADMIN_USERS_HTML.render(users=users, error=error, flash=flash),
        status_code=status,
    )


# --- Admin handlers ---


@handle_route_errors
async def admin_users_get(request: Request) -> HTMLResponse:
    require_admin(request)
    # Build the response first (mutable) so _read_flash can attach
    # the delete-cookie header for the one-shot banner semantics.
    with connect(_get_db_path(request)) as conn:
        users = list_users(conn)
    response = HTMLResponse(
        ADMIN_USERS_HTML.render(users=users, error=None, flash=None)
    )
    flash = _read_flash(request, response)
    if flash is not None:
        # Re-render with the flash payload threaded in. (delete_cookie
        # header is already attached by _read_flash; rebuilding the body
        # via HTMLResponse() would discard it, so set body via .body=.)
        response.body = ADMIN_USERS_HTML.render(
            users=users, error=None, flash=flash
        ).encode("utf-8")
    return response


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
async def admin_users_delete_get(request: Request) -> HTMLResponse:
    """Render the two-step delete-confirm page.

    Shows the username, file count, size_human, and resolved
    library path so the admin can verify what's about to be removed.
    Embeds a signed (actor_id, target_id) confirm token (salt
    'admin-delete-confirm') in a hidden form field; the POST handler
    verifies the token within a 10-minute window.
    """
    require_admin(request)
    user_id = int(request.path_params["user_id"])
    with connect(_get_db_path(request)) as conn:
        user = get_user_by_id(conn, user_id)
    if user is None:
        return PlainTextResponse("Not found", status_code=404)

    from model_library import measure_user_library, user_library_default_base

    usage = measure_user_library(user_id)
    # Use the base (no mkdir side-effect) so a GET on the confirm page
    # never creates a `users/<uid>/` dir for a user that has none yet.
    resolved_path = str((user_library_default_base() / str(user_id)).resolve())
    token = itsdangerous.URLSafeTimedSerializer(
        _get_secret(request),
        salt="admin-delete-confirm",
    ).dumps({"actor_id": _admin_actor_id(request), "target_id": user_id})
    body = ADMIN_DELETE_CONFIRM_HTML.render(
        user=user, usage=usage, resolved_path=resolved_path, token=token
    )
    return HTMLResponse(body)


@handle_route_errors
async def admin_users_delete_post(
    request: Request,
) -> RedirectResponse | HTMLResponse:
    """POST handler — verify confirm token, then cascade delete.

    Cascade order: capture username + orphan path BEFORE delete; on
    delete_user_atomic returning False (last-admin guard), library MUST
    NOT be cascaded; on OSError/ValueError during library rm, auth row
    is NOT rolled back (auth is source of truth).

    All branches return _redirect_with_flash() — a fresh
    /admin/users 303 redirect with a signed-cookie flash banner.
    """
    require_admin(request)
    actor_id = _admin_actor_id(request)
    target_id = int(request.path_params["user_id"])
    form = await request.form()

    db_path = _get_db_path(request)
    with connect(db_path) as conn:
        user_to_delete = get_user_by_id(conn, target_id)
        if user_to_delete is None:
            return _redirect_with_flash(
                request,
                "warn",
                f"User uid={target_id} already gone — nothing to do.",
            )
        username = user_to_delete.username

        raw_token = form.get("confirm_token", "")
        try:
            payload = itsdangerous.URLSafeTimedSerializer(
                _get_secret(request),
                salt="admin-delete-confirm",
            ).loads(raw_token, max_age=600)
        except itsdangerous.SignatureExpired:
            return _redirect_with_flash(
                request,
                "warn",
                "Confirm page expired — please re-open the delete page.",
            )
        except itsdangerous.BadSignature:
            logger.warning(
                "admin delete POST with bad confirm token actor=%d target=%d",
                actor_id,
                target_id,
            )
            return _redirect_with_flash(
                request,
                "warn",
                "Invalid confirmation token — please use the Delete link.",
            )

        if payload.get("actor_id") != actor_id or payload.get("target_id") != target_id:
            logger.warning(
                "admin delete confirm token actor/target mismatch "
                "(actor=%d target=%d payload=%r)",
                actor_id,
                target_id,
                payload,
            )
            return _redirect_with_flash(
                request,
                "warn",
                "Confirmation mismatch — please re-open the delete page.",
            )

        from model_library import user_library_default_base

        orphan_path = user_library_default_base() / str(target_id)

        try:
            deleted = delete_user_atomic(conn, user_id=target_id)
        except sqlite3.Error:
            logger.exception("auth db error during cascade delete uid=%d", target_id)
            return _redirect_with_flash(
                request,
                "error",
                "Database error — see server log.",
            )

        if not deleted:
            # Atomic guard rejected: target was the last admin. Library
            # MUST NOT be cascaded — the account is still active.
            return _redirect_with_flash(
                request,
                "warn",
                "Refused — would remove the last admin.",
            )
        # Auth row is gone (committed when `with connect` exits, below).

    # Cascade library cleanup AFTER the auth transaction commits.
    from model_library import delete_user_library

    try:
        usage = delete_user_library(target_id)
    except (OSError, ValueError):
        logger.exception(
            "library cascade failed uid=%d path=%s", target_id, orphan_path
        )
        return _redirect_with_flash(
            request,
            "warn",
            f"User deleted but files remain at {orphan_path} — see log.",
        )

    logger.info(
        "admin uid=%d deleted user uid=%d files=%d bytes=%d",
        actor_id,
        target_id,
        usage.files,
        usage.size_bytes,
    )
    return _redirect_with_flash(
        request,
        "info",
        f"Deleted user '{username}' ({usage.files} files / {usage.size_human}).",
    )


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
    Route(
        "/admin/users/{user_id:int}/delete",
        admin_users_delete_get,
        methods=["GET"],
    ),
    Route(
        "/admin/users/{user_id:int}/delete",
        admin_users_delete_post,
        methods=["POST"],
    ),
]
