"""Local model library: filesystem-backed TELEMAC project folders.

Pure functions (no Shiny imports). Companion files (.cas/.cli/.liq) are
located next to the chosen .slf, mirroring the upload-companion contract
in server_core._find_uploaded_by_ext.

Library root is `~/.telemac-viewer/models/` by default, overridable via
the `TELEMAC_VIEWER_MODELS` environment variable. Auto-created on first
access; refuses paths inside the viewer source tree.
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_VIEWER_TREE = Path(__file__).resolve().parent


# --- Per-user storage foundation (v3.7.0 per-user-storage feature) ---


class LibrarySource(Enum):
    """Where a ProjectEntry lives: in the user's own dir, or in the shared overlay."""

    USER = "user"
    SHARED = "shared"

    @property
    def display_label(self) -> str:
        return {"user": "My models", "shared": "Shared"}[self.value]


@dataclass(frozen=True)
class LibraryUsage:
    """Return type of measure_user_library / delete_user_library.

    Single dataclass shared by both so the (int, int) tuple order cannot
    silently swap between caller sites. The bytes field is renamed
    `size_bytes` to avoid shadowing the bytes builtin.
    """

    files: int
    size_bytes: int

    @property
    def size_human(self) -> str:
        n = self.size_bytes
        for unit, scale in [("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)]:
            if n >= scale:
                return f"{n / scale:.1f} {unit}"
        return f"{n} B"


_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COMPANION_BASENAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_ALLOWED_COMPANION_SUFFIXES = frozenset({".slf", ".cas", ".cli", ".liq"})
_PARTIAL_DIR_RE = re.compile(r"^\.(.+)\.partial-(\d+)$")
_SWEEP_MAX_PER_STARTUP = 5
_STALE_MTIME_FALLBACK_SECONDS = (
    30 * 60
)  # 30 min — older than any plausible in-progress save


def _validate_project_name(name: str) -> str:
    """Return the validated name; raise ValueError if unsafe.

    `.` and `..` are excluded by the regex (the dot is not in the
    character class), so no separate check needed.
    """
    if not isinstance(name, str) or not _PROJECT_NAME_RE.fullmatch(name):
        raise ValueError(f"Project name {name!r} must be 1-64 chars of [A-Za-z0-9_-]")
    return name


def _validate_companion_basename(name: str) -> str:
    """Sanitize a client-supplied filename to use as a destination basename.

    Shiny's FileInfo['name'] is client-controlled. Strip path components
    via os.path.basename, then enforce the regex and an extension whitelist.
    """
    if not isinstance(name, str):
        raise ValueError(f"Companion filename must be str, got {type(name).__name__}")
    bare = os.path.basename(name)
    if not _COMPANION_BASENAME_RE.fullmatch(bare):
        raise ValueError(f"Companion filename {name!r} contains unsafe characters")
    suffix = Path(bare).suffix.lower()
    if suffix not in _ALLOWED_COMPANION_SUFFIXES:
        raise ValueError(
            f"Companion {bare!r} has disallowed extension {suffix!r}; "
            f"allowed: {sorted(_ALLOWED_COMPANION_SUFFIXES)}"
        )
    return bare


def _sanitize_for_project_name(raw: str) -> str:
    """Best-effort coerce arbitrary text to a valid project name.

    Used for the HEC-RAS auto-name. Replaces any non-[A-Za-z0-9_-] character
    with `_`, collapses runs of underscores, trims to 64 chars, strips
    leading/trailing `_`. Returns `hecras_import_<ts>` if the result would
    be empty OR is just `hecras` OR matches `hecras_\d{8}-\d{6}` (only the
    prefix + timestamp survived).
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")[:64]
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fallback = f"hecras_import_{ts}"
    if (
        not cleaned
        or cleaned == "hecras"
        or re.fullmatch(r"hecras_\d{8}-\d{6}", cleaned)
    ):
        return fallback
    return cleaned


def _validate_user_id(user_id: int) -> None:
    """Reject non-positive, bool-subtype, or out-of-range user ids.

    bool is a subclass of int in Python; without the isinstance(bool)
    guard, user_id=True would silently map to user 1's library.

    Upper bound 2**63 matches SQLite's signed INTEGER range.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise TypeError(f"user_id must be int (not bool), got {type(user_id).__name__}")
    if not (0 < user_id < 2**63):
        raise ValueError(f"user_id out of SQLite INTEGER range: {user_id}")


_initialized: set[Path] = set()
_warned: set[str] = set()
_swept_for: set[Path] = set()
_sweep_total_count: int = 0


def _default_library_root() -> Path:
    """Compute the default library root lazily so tests can monkeypatch
    Path.home without re-importing the module.
    """
    return Path.home() / ".telemac-viewer" / "models"


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[viewer] {message}", file=sys.stderr)


def _reset_for_testing() -> None:
    """Clear module-level memoization. Tests only."""
    global _sweep_total_count
    _initialized.clear()
    _warned.clear()
    _swept_for.clear()
    _sweep_total_count = 0


@dataclass(frozen=True)
class ProjectEntry:
    name: str
    path: Path
    slf_files: tuple[Path, ...]


@dataclass(frozen=True)
class ProjectFiles:
    slf: Path
    cas: Path | None
    cli: Path | None
    liq: Path | None


def library_root() -> Path:
    """Resolve the model library root, creating it on first call.

    Returns the path even when the path-safety guard fires; the guard's
    side effect is a stderr warning and `scan_library` treating the root
    as empty.
    """
    raw = os.environ.get("TELEMAC_VIEWER_MODELS")
    root = (
        Path(raw).expanduser().resolve() if raw else _default_library_root().resolve()
    )

    try:
        if _VIEWER_TREE in root.parents or root == _VIEWER_TREE:
            _warn_once(
                f"unsafe-root:{root}",
                f"library root {root} is inside the viewer source tree — "
                "treating as empty (set TELEMAC_VIEWER_MODELS to a different path)",
            )
            return root
    except (OSError, ValueError):
        pass

    if root not in _initialized:
        _initialized.add(root)
        try:
            existed = root.exists()
            root.mkdir(parents=True, exist_ok=True)
            if not existed:
                print(f"[viewer] created model library at {root}", file=sys.stderr)
        except OSError as exc:
            _warn_once(
                f"mkdir-fail:{root}",
                f"could not create library root {root}: {exc}",
            )
    return root


def scan_library(root: Path) -> list[ProjectEntry]:
    """List one-level-deep project folders containing at least one .slf.

    Detection rules (per spec §scan_library):
    - Skip non-directories, hidden names, unresolvable symlinks.
    - Skip folders with no .slf inside.
    - Sort projects and .slf files alphabetically.
    """
    if _VIEWER_TREE in root.parents or root == _VIEWER_TREE:
        return []
    if not root.is_dir():
        return []
    try:
        with os.scandir(root) as it:
            candidates = sorted(it, key=lambda e: e.name.lower())
    except OSError as exc:
        _warn_once(f"scan-fail:{root}", f"could not list {root}: {exc}")
        return []

    entries: list[ProjectEntry] = []
    for child in candidates:
        if child.name.startswith("."):
            continue
        try:
            if not child.is_dir(follow_symlinks=True):
                continue
        except OSError:
            continue
        proj_path = Path(child.path).resolve()
        try:
            slfs = tuple(
                sorted(
                    (
                        p
                        for p in proj_path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".slf"
                    ),
                    key=lambda p: p.name.lower(),
                )
            )
        except OSError:
            continue
        if not slfs:
            continue
        entries.append(ProjectEntry(name=child.name, path=proj_path, slf_files=slfs))
    return entries


def resolve_project(entry: ProjectEntry, slf_name: str) -> ProjectFiles:
    """Locate the chosen .slf and any matching .cas/.cli/.liq companions.

    Companion resolution: prefer the file whose basename matches the .slf;
    otherwise, if exactly one file of that extension lives in the folder,
    use it; otherwise return None for that companion.

    Raises FileNotFoundError if `slf_name` doesn't exist in the project
    folder (e.g., user deleted it after the dropdown was populated).
    """
    slf_path = entry.path / slf_name
    if not slf_path.is_file():
        raise FileNotFoundError(f"{slf_name} not found in {entry.path}")

    base = Path(slf_name).stem
    found: dict[str, Path | None] = {".cas": None, ".cli": None, ".liq": None}
    by_ext: dict[str, list[Path]] = {".cas": [], ".cli": [], ".liq": []}

    try:
        children = list(entry.path.iterdir())
    except OSError:
        children = []

    for p in children:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in by_ext:
            continue
        by_ext[ext].append(p)
        if p.stem == base:
            found[ext] = p

    for ext in (".cas", ".cli", ".liq"):
        if found[ext] is None and len(by_ext[ext]) == 1:
            found[ext] = by_ext[ext][0]

    return ProjectFiles(
        slf=slf_path,
        cas=found[".cas"],
        cli=found[".cli"],
        liq=found[".liq"],
    )


def find_companion(
    library_selection: tuple[str, str] | None,
    lib_root: Path,
    ext: str,
) -> Path | None:
    """Look up a companion file (.cas/.cli/.liq) for the selected library project.

    Returns None when no project is selected, when the project has been
    renamed/deleted, or when the requested companion is missing. Companions
    are optional, so we silently degrade — `tel_file()` clears the
    selection on its own when the .slf becomes unreachable.
    """
    if library_selection is None:
        return None
    project_name, slf_name = library_selection
    try:
        for entry in scan_library(lib_root):
            if entry.name == project_name:
                attr = ext.lstrip(".").lower()
                return getattr(resolve_project(entry, slf_name), attr)
    except FileNotFoundError:
        pass
    return None


def user_library_default_base() -> Path:
    """Read TELEMAC_VIEWER_USERS_ROOT or fall back to ~/.telemac-viewer/users.

    Public (no leading underscore) because auth/routes.py uses this from the
    admin-delete error handler to compute the orphan path WITHOUT triggering
    the mkdir side-effect of user_library_root().

    Pattern mirrors _default_library_root(). Tests monkeypatch the env var.
    """
    raw = os.environ.get("TELEMAC_VIEWER_USERS_ROOT")
    return (
        Path(raw).expanduser().resolve()
        if raw
        else Path.home() / ".telemac-viewer" / "users"
    )


def _sweep_stale_partials(models_dir: Path, *, user_id: int) -> int:
    """Remove .<name>.partial-<digits> dirs whose pid is no longer running
    or whose dir is older than _STALE_MTIME_FALLBACK_SECONDS.

    See spec §4.1 for the full semantics: strict regex (skips .lock);
    PID-reuse defense via mtime fallback; PermissionError treated as
    alive-skip; per-process cap (_SWEEP_MAX_PER_STARTUP) NOT per-uid;
    WARN log on overflow with the uid.
    """
    global _sweep_total_count
    if _sweep_total_count >= _SWEEP_MAX_PER_STARTUP:
        return 0

    removed = 0
    candidates = []
    if not models_dir.is_dir():
        return 0
    for entry in models_dir.iterdir():
        m = _PARTIAL_DIR_RE.match(entry.name)
        if not m or not entry.is_dir():
            continue
        candidates.append((entry, int(m.group(2))))

    available = max(0, _SWEEP_MAX_PER_STARTUP - _sweep_total_count)
    if len(candidates) > available:
        print(
            f"[viewer] _sweep_stale_partials uid={user_id} found "
            f"{len(candidates)} stale partials; will sweep up to "
            f"{available} this restart (process cap reached after {_sweep_total_count})",
            file=sys.stderr,
        )

    for partial, pid in candidates:
        if _sweep_total_count >= _SWEEP_MAX_PER_STARTUP:
            break
        try:
            os.kill(pid, 0)
            # Process alive (or pid is reused by another process). Apply
            # the mtime fallback: if the partial dir is older than the
            # cutoff, treat as stale regardless.
            mtime = partial.stat().st_mtime
            if time.time() - mtime <= _STALE_MTIME_FALLBACK_SECONDS:
                continue  # alive AND recent — skip
        except ProcessLookupError:
            pass  # dead — safe to remove
        except PermissionError:
            continue  # alive other-owned — skip

        try:
            shutil.rmtree(partial, ignore_errors=False)
            removed += 1
            _sweep_total_count += 1
        except OSError as e:
            print(
                f"[viewer] _sweep_stale_partials uid={user_id} could not "
                f"rmtree {partial}: {e}",
                file=sys.stderr,
            )

    return removed


def user_library_root(user_id: int) -> Path:
    """Return the per-user models directory; mkdir(0o700) if missing.

    Steps (see spec §4.1):
      1. validate user_id (TypeError/ValueError)
      2. compute root path under user_library_default_base()
      3. refuse if root is inside _VIEWER_TREE
      4. mkdir(mode=0o700) + explicit chmod (umask defense)
      5. chmod the user_id parent dir too
      6. _sweep_stale_partials once per process per (uid, base)
    """
    _validate_user_id(user_id)
    base = user_library_default_base()
    parent = base / str(user_id)
    root = parent / "models"

    if _VIEWER_TREE in root.parents or root == _VIEWER_TREE:
        raise ValueError(
            f"user_library_root resolves to {root} which is inside the "
            "viewer source tree; refuse to operate"
        )

    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)

    if root not in _swept_for:
        _swept_for.add(root)
        _sweep_stale_partials(root, user_id=user_id)

    return root
