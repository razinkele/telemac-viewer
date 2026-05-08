"""CLI: `python -m auth.cli create-admin | reset-password`.

Spec §8. Two subcommands; rotate-secret is a documented runbook, not a CLI.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from auth.core import (
    DEFAULT_DB_PATH,
    connect,
    create_user,
    ensure_schema,
    get_user_by_username,
    update_password_hash,
)
from auth.crypto import hash_password


# --- Exit codes (spec §8) ---
EXIT_OK = 0
EXIT_ADMIN_EXISTS = 2
EXIT_FS_ERROR = 3
EXIT_USER_NOT_FOUND = 4
EXIT_INPUT_ERROR = 5
EXIT_CHMOD_FAIL = 6


def _db_path() -> Path:
    """Honor TELEMAC_VIEWER_DB env var for tests, else default."""
    env = os.environ.get("TELEMAC_VIEWER_DB")
    return Path(env) if env else DEFAULT_DB_PATH


def _read_password(args: argparse.Namespace) -> str:
    """Resolve a password from --password-file or interactive prompt.

    Spec §8: non-tty refuses unless --password-file is supplied; password
    files MUST be mode 0600 and have a single trailing CR/LF stripped.
    """
    if args.password_file:
        p = Path(args.password_file)
        try:
            mode = p.stat().st_mode & 0o777
        except OSError as e:
            print(f"ERROR: cannot stat {p}: {e}", file=sys.stderr)
            sys.exit(EXIT_INPUT_ERROR)
        if mode != 0o600:
            print(
                f"ERROR: password file {p} has mode {oct(mode)}; "
                f"refusing to read with mode != 0600.",
                file=sys.stderr,
            )
            sys.exit(EXIT_INPUT_ERROR)
        # Read and strip exactly one trailing CR/LF.
        data = p.read_bytes().rstrip(b"\r\n")
        return data.decode("utf-8")

    if not sys.stdin.isatty():
        print(
            "ERROR: refusing to read password from non-interactive stdin; "
            "run from a terminal or pass --password-file <path> "
            "(mode 0600 enforced).",
            file=sys.stderr,
        )
        sys.exit(EXIT_INPUT_ERROR)

    pw1 = getpass.getpass("Password: ")
    pw2 = getpass.getpass("Confirm:  ")
    if pw1 != pw2:
        print("ERROR: passwords don't match.", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    return pw1


def _validate(pw: str) -> None:
    if len(pw) < 8:
        print("ERROR: password must be at least 8 characters.", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    if len(pw.encode("utf-8")) > 72:
        print("ERROR: password must be at most 72 UTF-8 bytes.", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)


def cmd_create_admin(args: argparse.Namespace) -> int:
    # Open the DB ONCE just to check whether an admin already exists, then
    # close it before prompting. Two reasons mirroring cmd_reset_password:
    #   (a) don't hold a sqlite handle through interactive getpass();
    #   (b) sys.exit() inside a `with connect()` block would commit a
    #       half-baked transaction.
    db = _db_path()
    try:
        with connect(db) as conn:
            ensure_schema(conn)
            existing_admin = conn.execute(
                "SELECT id FROM users WHERE is_admin = 1 LIMIT 1"
            ).fetchone()
        if existing_admin:
            print(
                "ERROR: an admin already exists; use the /admin UI to add more users.",
                file=sys.stderr,
            )
            return EXIT_ADMIN_EXISTS
    except PermissionError as e:
        print(f"ERROR: filesystem permission: {e}", file=sys.stderr)
        return EXIT_CHMOD_FAIL
    except OSError as e:
        print(f"ERROR: filesystem: {e}", file=sys.stderr)
        return EXIT_FS_ERROR

    # Now prompt for password (no DB connection held during prompt).
    password = _read_password(args)
    _validate(password)

    try:
        with connect(db) as conn:
            uid = create_user(
                conn,
                username=args.username,
                password_hash=hash_password(password),
                display_name=args.display_name,
                is_admin=True,
            )
        print(f"Created admin {args.username!r} (id={uid}).")
        return EXIT_OK
    except PermissionError as e:
        print(f"ERROR: filesystem permission: {e}", file=sys.stderr)
        return EXIT_CHMOD_FAIL
    except OSError as e:
        print(f"ERROR: filesystem: {e}", file=sys.stderr)
        return EXIT_FS_ERROR


def cmd_reset_password(args: argparse.Namespace) -> int:
    # Read and validate the password BEFORE opening the DB connection so
    # we don't hold a sqlite handle during interactive prompts and don't
    # call sys.exit() while inside `with connect(...)`.
    password = _read_password(args)
    _validate(password)

    db = _db_path()
    try:
        with connect(db) as conn:
            ensure_schema(conn)
            u = get_user_by_username(conn, args.username)
            if u is None:
                print(f"ERROR: user {args.username!r} not found.", file=sys.stderr)
                return EXIT_USER_NOT_FOUND
            update_password_hash(
                conn, user_id=u.id, password_hash=hash_password(password)
            )
    except PermissionError as e:
        print(f"ERROR: filesystem permission: {e}", file=sys.stderr)
        return EXIT_CHMOD_FAIL
    except OSError as e:
        print(f"ERROR: filesystem: {e}", file=sys.stderr)
        return EXIT_FS_ERROR
    print(f"Password reset for {args.username!r}.")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m auth.cli",
        description="TELEMAC viewer auth CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create-admin", help="Create the first admin.")
    p_create.add_argument("--username", required=True)
    p_create.add_argument("--display-name", default=None)
    p_create.add_argument(
        "--password-file",
        default=None,
        help="File containing the password (mode 0600 required).",
    )
    p_create.set_defaults(func=cmd_create_admin)

    p_reset = sub.add_parser("reset-password", help="Reset a user's password.")
    p_reset.add_argument("--username", required=True)
    p_reset.add_argument("--password-file", default=None)
    p_reset.set_defaults(func=cmd_reset_password)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
