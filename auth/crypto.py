"""bcrypt password hashing + signed-cookie session helpers.

Spec §6, §7.1. Isolated from DB for unit testability — this module
has no sqlite imports.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.hash import bcrypt

# --- Constants ---

# bcrypt cost factor. Default 12 -> ~100ms per verify on modern hardware.
# CRITICAL: NULL_HASH below uses this same constant so timing-equalized
# verify against "user not found" runs identical rounds.
BCRYPT_ROUNDS = 12

# --- Passwords ---


def hash_password(plain: str) -> str:
    return bcrypt.using(rounds=BCRYPT_ROUNDS).hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verify. Returns False on any malformed hash too."""
    try:
        return bcrypt.verify(plain, hashed)
    except (ValueError, TypeError):
        return False


# Pre-computed null hash at the SAME cost as new user passwords so that
# verify-when-user-not-found runs the same number of bcrypt rounds.
# Computed once at module import.
NULL_HASH = hash_password("")


# --- Session cookies ---

DEFAULT_MAX_AGE = 30 * 86400  # 30 days, fixed (no rolling refresh)


def sign_session_cookie(
    *, user_id: int, secret: bytes, salt: str = "tv-session"
) -> str:
    serializer = URLSafeTimedSerializer(secret, salt=salt)
    return serializer.dumps({"user_id": user_id})


def decode_session_cookie(
    token: str,
    *,
    secret: bytes,
    salt: str = "tv-session",
    max_age: int = DEFAULT_MAX_AGE,
) -> dict | None:
    """Returns the payload dict on success, None on bad-sig or expired.

    The returned dict includes an ``iat`` (issued-at, unix seconds) field
    extracted from the signed timestamp itsdangerous embeds in the token.
    A non-positive ``max_age`` rejects all cookies (callers occasionally
    use ``max_age=0`` to force re-auth without trusting clock skew).
    """
    if max_age <= 0:
        return None
    serializer = URLSafeTimedSerializer(secret, salt=salt)
    try:
        payload, ts = serializer.loads(token, max_age=max_age, return_timestamp=True)
    except (BadSignature, SignatureExpired):
        return None
    if isinstance(payload, dict):
        payload = {**payload, "iat": int(ts.timestamp())}
    return payload


# --- Signing-secret persistence ---


def load_or_create_secret(path: Path) -> bytes:
    """Read the signing secret from disk, creating it on FileNotFoundError.

    Any other read failure (PermissionError, IsADirectoryError, empty file)
    is propagated — the caller (process startup) refuses to continue rather
    than silently regenerating, which would mass-log-out every user.
    """
    try:
        secret = path.read_bytes()
        if not secret:
            raise ValueError(f"signing-secret at {path} is empty")
        return secret
    except FileNotFoundError:
        pass

    # Generate atomically: tmp + rename, mode 0o600 via O_CREAT|O_EXCL.
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    new_secret = secrets.token_hex(32).encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new_secret)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return new_secret
