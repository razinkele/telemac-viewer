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
