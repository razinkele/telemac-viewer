"""Local model library: filesystem-backed TELEMAC project folders.

Pure functions (no Shiny imports). Companion files (.cas/.cli/.liq) are
located next to the chosen .slf, mirroring the upload-companion contract
in server_core._find_uploaded_by_ext.

Library root is `~/.telemac-viewer/models/` by default, overridable via
the `TELEMAC_VIEWER_MODELS` environment variable. Auto-created on first
access; refuses paths inside the viewer source tree.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_VIEWER_TREE = Path(__file__).resolve().parent

_initialized: set[Path] = set()
_warned: set[str] = set()


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
    _initialized.clear()
    _warned.clear()


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
