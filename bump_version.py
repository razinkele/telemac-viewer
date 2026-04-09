#!/usr/bin/env python3
"""Bump version, update CHANGELOG.md, and create a git tag.

Usage:
    python bump_version.py patch   # 3.1.0 -> 3.1.1
    python bump_version.py minor   # 3.1.0 -> 3.2.0
    python bump_version.py major   # 3.1.0 -> 4.0.0
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
VERSION_FILE = ROOT / "VERSION"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def read_version() -> tuple[int, int, int]:
    text = VERSION_FILE.read_text().strip()
    parts = text.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def bump(major: int, minor: int, patch: int, part: str) -> tuple[int, int, int]:
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    if part == "patch":
        return major, minor, patch + 1
    raise ValueError(f"Unknown part: {part!r}. Use major, minor, or patch.")


def update_changelog(old_ver: str, new_ver: str) -> None:
    if not CHANGELOG_FILE.exists():
        return
    text = CHANGELOG_FILE.read_text()
    today = date.today().isoformat()
    text = text.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{new_ver}] - {today}",
        1,
    )
    CHANGELOG_FILE.write_text(text)


def git_tag(version: str) -> None:
    subprocess.run(["git", "add", "VERSION", "CHANGELOG.md", "app.py"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"release: v{version}"],
        check=True,
    )
    subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"v{version}"], check=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("major", "minor", "patch"):
        print(__doc__.strip())
        sys.exit(1)

    part = sys.argv[1]
    major, minor, patch = read_version()
    old_ver = f"{major}.{minor}.{patch}"
    major, minor, patch = bump(major, minor, patch, part)
    new_ver = f"{major}.{minor}.{patch}"

    VERSION_FILE.write_text(new_ver + "\n")
    update_changelog(old_ver, new_ver)

    print(f"{old_ver} -> {new_ver}")
    git_tag(new_ver)
    print(f"Tagged v{new_ver}. Push with: git push && git push --tags")


if __name__ == "__main__":
    main()
