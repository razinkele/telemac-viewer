"""Tests for auth.crypto — password hashing + cookie sign/verify."""

from __future__ import annotations

import time

import pytest


def test_hash_and_verify_password_round_trip() -> None:
    from auth.crypto import hash_password, verify_password

    h = hash_password("hunter2")
    assert verify_password("hunter2", h) is True
    assert verify_password("wrong", h) is False


def test_null_hash_uses_same_cost_factor_as_real_hashes() -> None:
    """Timing equalization requires NULL_HASH to be at the same cost
    factor as new passwords; otherwise verify-against-NULL_HASH runs
    fewer rounds, leaking user-not-found via timing."""
    from auth.crypto import BCRYPT_ROUNDS, NULL_HASH, hash_password

    assert BCRYPT_ROUNDS >= 10, (
        f"BCRYPT_ROUNDS={BCRYPT_ROUNDS} too low; "
        "minimum cost for production-grade bcrypt is 10."
    )
    real = hash_password("anything")
    real_cost = int(real.split("$")[2])
    null_cost = int(NULL_HASH.split("$")[2])
    assert real_cost == BCRYPT_ROUNDS == null_cost


def test_verify_against_null_hash_returns_false_quickly_but_not_too_quickly() -> None:
    """Sanity: verify against NULL_HASH should take roughly the same
    time as verify against a real hash (both run bcrypt's full cost)."""
    from auth.crypto import NULL_HASH, hash_password, verify_password

    real = hash_password("p")

    t0 = time.perf_counter()
    verify_password("anything", NULL_HASH)
    null_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    verify_password("anything", real)
    real_t = time.perf_counter() - t0

    ratio = max(null_t, real_t) / max(min(null_t, real_t), 1e-6)
    assert ratio < 3, f"timing diverged: null={null_t:.3f}s real={real_t:.3f}s"


def test_sign_and_decode_session_round_trip() -> None:
    from auth.crypto import decode_session_cookie, sign_session_cookie

    secret = b"x" * 32
    token = sign_session_cookie(user_id=42, secret=secret)
    payload = decode_session_cookie(token, secret=secret)
    assert payload["user_id"] == 42
    assert "iat" in payload


def test_decode_session_rejects_bad_signature() -> None:
    from auth.crypto import decode_session_cookie, sign_session_cookie

    token = sign_session_cookie(user_id=1, secret=b"x" * 32)
    payload = decode_session_cookie(token, secret=b"y" * 32)
    assert payload is None


def test_decode_session_rejects_expired() -> None:
    """Cookies older than max_age are rejected."""
    from auth.crypto import decode_session_cookie, sign_session_cookie

    token = sign_session_cookie(user_id=1, secret=b"x" * 32)
    payload = decode_session_cookie(token, secret=b"x" * 32, max_age=0)
    assert payload is None


def test_load_or_create_secret_creates_with_0600(tmp_path) -> None:
    from auth.crypto import load_or_create_secret
    import os

    path = tmp_path / "auth_secret"
    secret1 = load_or_create_secret(path)
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert len(secret1) >= 32

    secret2 = load_or_create_secret(path)
    assert secret1 == secret2  # idempotent read


def test_load_or_create_secret_refuses_unreadable(tmp_path) -> None:
    """Read-fail other than FileNotFoundError must NOT silently regenerate."""
    from auth.crypto import load_or_create_secret
    import os

    path = tmp_path / "auth_secret"
    path.write_bytes(b"x" * 32)
    os.chmod(path, 0o000)
    try:
        with pytest.raises(PermissionError):
            load_or_create_secret(path)
    finally:
        os.chmod(path, 0o600)
