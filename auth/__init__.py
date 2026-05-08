"""Multi-user auth for the TELEMAC viewer.

Spec: docs/superpowers/specs/2026-05-08-user-accounts-design.md
"""

from auth.core import (  # noqa: F401  (public re-export)
    DEFAULT_DB_PATH,
    User,
    create_user,
    delete_user_atomic,
    ensure_schema,
    get_user_by_id,
    get_user_by_username,
    list_users,
    update_password_hash,
    update_preferences,
    update_user,
)
from auth.crypto import (  # noqa: F401
    BCRYPT_ROUNDS,
    NULL_HASH,
    decode_session_cookie,
    hash_password,
    load_or_create_secret,
    sign_session_cookie,
    verify_password,
)
from auth.middleware import (  # noqa: F401
    auth_middleware,
    get_current_user_from_request,
    get_current_user_id_from_scope,
    handle_route_errors,
    require_admin,
    warn_if_public_bind,
)
from auth.routes import auth_routes  # noqa: F401
