"""bcrypt password hashing + signed-cookie session helpers.

Spec §6, §7.1. Isolated from DB for unit testability — this module
has no sqlite imports.
"""

from __future__ import annotations

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
