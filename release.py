#!/usr/bin/env python3
"""Release automation CLI — version bumping, commit parsing, and git tagging.

Usage:
    release.py prep <major|minor|patch> [--since <ref>]
    release.py bump <major|minor|patch>
    release.py tag <version> [files...]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VERSION_FILE = ROOT / "VERSION"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
DEFAULT_TAG_FILES = ["VERSION", "CHANGELOG.md", "docs/API.md", "README.md"]

USER_FACING_MODULES = [
    "analysis.py",
    "geometry.py",
    "layers.py",
    "crs.py",
    "validation.py",
    "constants.py",
    "telemac_defaults.py",
    "viewer_types.py",
]

COMMIT_TYPE_MAP = {
    "feat": "Added",
    "fix": "Fixed",
    "refactor": "Changed",
    "perf": "Changed",
    "style": "Changed",
    "test": "Tests",
    "docs": "Documentation",
}

OMITTED_TYPES = {"chore", "ci", "build", "release"}

COMMIT_RE = re.compile(r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?:\s*(?P<message>.+)$")


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command, raising RuntimeError with stderr on non-zero exit.

    ``subprocess.run(..., check=True)`` raises ``CalledProcessError`` whose
    ``.stderr`` is ``None`` unless ``capture_output=True`` is set. Using this
    helper ensures the actual git error (e.g. pre-commit hook rejection)
    surfaces in the raised message instead of a bare traceback.
    """
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git command failed: {' '.join(args)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def read_version(version_file: Path = VERSION_FILE) -> tuple[int, int, int]:
    """Read VERSION file and return (major, minor, patch) tuple."""
    text = Path(version_file).read_text().strip()
    parts = text.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def bump(major: int, minor: int, patch: int, part: str) -> tuple[int, int, int]:
    """Bump version by part (major/minor/patch)."""
    if part == "patch":
        return (major, minor, patch + 1)
    elif part == "minor":
        return (major, minor + 1, 0)
    elif part == "major":
        return (major + 1, 0, 0)
    else:
        raise ValueError(
            f"Invalid bump part: {part!r}. Must be major, minor, or patch."
        )


def write_version(version_file: Path, version: tuple[int, int, int]) -> None:
    """Write version string + newline to file."""
    Path(version_file).write_text(f"{version[0]}.{version[1]}.{version[2]}\n")


def parse_commit_message(subject: str) -> tuple[str | None, str | None, str]:
    """Parse 'type(scope): message' format. Returns (type, scope, message)."""
    m = COMMIT_RE.match(subject)
    if m:
        return m.group("type"), m.group("scope"), m.group("message")
    return None, None, subject


def map_commit_type(commit_type: str | None) -> str | None:
    """Map commit type to changelog section. Returns None for omitted types."""
    if commit_type is None:
        return "Other"
    if commit_type in OMITTED_TYPES:
        return None
    return COMMIT_TYPE_MAP.get(commit_type, "Other")


def strip_trailers(body: str) -> str:
    """Strip git trailers from end of commit body."""
    lines = body.strip().splitlines()
    known_trailers = {"Co-Authored-By", "Signed-off-by", "Reviewed-by", "Acked-by"}
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue
        m = re.match(r"^([\w][\w-]*[\w]):\s", line)
        if m and ("-" in m.group(1) or m.group(1) in known_trailers):
            lines.pop()
        else:
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _find_latest_tag(cwd: Path | None = None) -> str | None:
    """Find latest v* tag using git describe."""
    try:
        result = _run_git(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            cwd=cwd,
        )
        return result.stdout.strip()
    except RuntimeError:
        return None


def _get_files_changed(commit_hash: str, cwd: Path | None = None) -> list[str]:
    """Get list of files changed in a commit."""
    result = _run_git(
        [
            "git",
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--name-only",
            commit_hash,
        ],
        cwd=cwd,
    )
    return [f for f in result.stdout.strip().splitlines() if f]


def gather_commits(cwd: Path | None = None, since: str | None = None) -> list[dict]:
    """Gather commits since tag/ref. Returns list of commit dicts."""
    if since is None:
        since = _find_latest_tag(cwd)

    delimiter = "--commit-boundary-a1b2c3--"
    fmt = f"%H%n%s%n%b%n{delimiter}"

    cmd = ["git", "log", f"--format={fmt}"]
    if since:
        cmd.append(f"{since}..HEAD")
    else:
        cmd.append("HEAD")

    result = _run_git(cmd, cwd=cwd)

    raw = result.stdout.strip()
    if not raw:
        return []

    blocks = raw.split(delimiter)
    commits = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 2)
        if len(lines) < 2:
            continue
        commit_hash = lines[0].strip()
        subject = lines[1].strip()
        body = lines[2].strip() if len(lines) > 2 else ""
        body = strip_trailers(body)

        ctype, scope, message = parse_commit_message(subject)
        changelog_section = map_commit_type(ctype)
        files_changed = _get_files_changed(commit_hash, cwd)

        commits.append(
            {
                "hash": commit_hash[:7],
                "type": ctype,
                "scope": scope,
                "message": message,
                "body": body,
                "files_changed": files_changed,
                "changelog_section": changelog_section,
            }
        )

    return commits


def prep_json(
    bump_type: str,
    version_file: Path = VERSION_FILE,
    cwd: Path | None = None,
    since: str | None = None,
) -> dict:
    """Return full JSON structure for release prep."""
    old = read_version(version_file)
    new = bump(old[0], old[1], old[2], bump_type)
    commits = gather_commits(cwd, since)

    all_changed = set()
    for c in commits:
        all_changed.update(c["files_changed"])

    changed_modules = sorted(f for f in all_changed if f.endswith(".py"))
    user_facing = [m for m in USER_FACING_MODULES if m in all_changed]

    return {
        "bump_type": bump_type,
        "old_version": f"{old[0]}.{old[1]}.{old[2]}",
        "new_version": f"{new[0]}.{new[1]}.{new[2]}",
        "commits": commits,
        "changed_modules": changed_modules,
        "user_facing_modules": user_facing,
    }


def git_tag(
    version: str,
    files: list[str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Stage files, commit as 'release: v{version}', create annotated tag."""
    if files is None:
        files = list(DEFAULT_TAG_FILES)

    cwd_path = Path(cwd) if cwd else ROOT

    # Stage files, skipping missing ones
    for f in files:
        fpath = cwd_path / f
        if fpath.exists():
            _run_git(["git", "add", f], cwd=cwd_path)

    _run_git(
        ["git", "commit", "-m", f"release: v{version}"],
        cwd=cwd_path,
    )
    _run_git(
        ["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"],
        cwd=cwd_path,
    )


def main() -> None:
    """CLI entry point dispatching prep/bump/tag commands."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "prep":
        if len(sys.argv) < 3 or sys.argv[2] not in ("major", "minor", "patch"):
            print("Usage: release.py prep <major|minor|patch> [--since <ref>]")
            sys.exit(1)
        bump_type = sys.argv[2]
        since = None
        if "--since" in sys.argv:
            idx = sys.argv.index("--since")
            if idx + 1 < len(sys.argv):
                since = sys.argv[idx + 1]
            else:
                print("Error: --since requires a ref argument")
                sys.exit(1)
        result = prep_json(bump_type, since=since)
        print(json.dumps(result, indent=2))

    elif command == "bump":
        if len(sys.argv) < 3 or sys.argv[2] not in ("major", "minor", "patch"):
            print("Usage: release.py bump <major|minor|patch>")
            sys.exit(1)
        bump_type = sys.argv[2]
        old = read_version()
        new = bump(old[0], old[1], old[2], bump_type)
        write_version(VERSION_FILE, new)
        print(f"{old[0]}.{old[1]}.{old[2]} -> {new[0]}.{new[1]}.{new[2]}")

    elif command == "tag":
        if len(sys.argv) < 3:
            print("Usage: release.py tag <version> [files...]")
            sys.exit(1)
        version = sys.argv[2]
        files = sys.argv[3:] if len(sys.argv) > 3 else None
        git_tag(version, files)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
