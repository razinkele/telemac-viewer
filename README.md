# TELEMAC Viewer

[![Version](https://img.shields.io/badge/version-3.6.0-blue.svg)](./CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-603%20passing-brightgreen.svg)](./tests)
[![License](https://img.shields.io/badge/license-LGPL%20v2.1-orange.svg)](#license)

A web-based viewer for [TELEMAC](http://www.opentelemac.org/) simulation results,
built with [Shiny for Python](https://shiny.posit.co/py/) and
[deck.gl](https://deck.gl/) via
[shiny-deckgl](https://github.com/pbs-data-solutions/shiny-deckgl).

Visualize SELAFIN (`.slf`) results from TELEMAC-2D, TELEMAC-3D, GAIA, TOMAWAC,
ARTEMIS, and KHIONE with interactive maps, time-series analysis, and hydraulic
engineering tools.

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Multi-user setup](#multi-user-setup)
- [Project structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [License](#license)

## Features

### Visualization
- **Mesh rendering** with per-vertex coloring and 10 palettes (Viridis, Plasma, Ocean, Thermal, …)
- **3D elevation** with adjustable vertical exaggeration
- **Velocity arrows** overlay with configurable scale
- **Wireframe**, **contour isolines**, **particle tracing** with Lagrangian trails
- **Boundary nodes** color-coded from `.cli`
- **Min/max extrema** markers
- **Light/dark map canvas** + multiple basemaps (CartoDB, Satellite)
- **Difference mode** between timesteps with a diverging palette
- **Fast timestep scrub** — partial-update dispatcher sends only the mesh color buffer on palette/value changes, preserving positions and indices via deck.gl's JS-side cache (~3-4× smaller WebSocket payload for large meshes)
- **Local model library** — drop project folders into `~/.telemac-viewer/models/` (or `TELEMAC_VIEWER_MODELS=...`) — they appear in a "My models" sidebar dropdown alongside Examples and Upload. Each project folder is auto-scanned for `.slf` plus optional `.cas` / `.cli` / `.liq` companions; refresh button rescans on demand.

### Analysis
- **Time series** at clicked points, with CSV export (barycentric-interpolated for derived variables — matches the map layer)
- **Cross-section profiles** along user-drawn polylines
- **Vertical profiles** for 3D results, with layer extraction
- **Polygon zonal statistics** (min / max / mean / area / flooded fraction)
- **Flood mapping** — envelope, arrival time, duration
- **Discharge** across polylines (Q integral)
- **Volume conservation** over time (falls back to `FREE SURFACE − BOTTOM` when no depth variable is present)
- **Temporal statistics** (min / max / mean / std across all timesteps)
- **Courant number**, mesh quality, slope, element-area diagnostics
- **Custom expressions** with safe AST evaluation (`VELOCITY_U**2 + VELOCITY_V**2`)
- **Derived variables** — velocity magnitude, Froude number, vorticity
- **Mesh identity hash** — compare-overlay rejects files whose geometry differs beyond count checks

### CRS awareness
- Auto-detection from TELEMAC `.cas` files (`GEOGRAPHIC SYSTEM` keyword)
- UTM, Lambert, LKS94, and any EPSG code via manual entry
- Basemap alignment via `pyproj`
- Manual origin offset for pre-centered meshes

### HEC-RAS import
- Import HEC-RAS 1D and 2D models (`.hdf`) and convert to TELEMAC
- Mesh generation with Triangle or Gmsh backends
- Automatic Manning's roughness extraction
- Boundary-condition mapping and `.liq` file generation
- Preview map showing alignment, cross-sections, and boundaries

### Validation
- Upload observation CSV files for model-vs-observed comparison
- RMSE and Nash–Sutcliffe Efficiency (NSE)
- `.liq` liquid-boundary-file parser

## Requirements

- Python 3.11+
- A working TELEMAC v8p5r1 installation (the viewer imports `TelemacFile` from its scripts)
- Intel MPI or OpenMPI if you plan to launch simulations from within the viewer

### Python dependencies

| Package | Version |
|---|---|
| `shiny` | ≥ 1.5.1 |
| `shiny-deckgl` | ≥ 1.0.1 |
| `numpy` | any recent |
| `scipy` | any recent |
| `pyproj` | any recent |

Optional (for HEC-RAS import): `h5py`, `rasterio`, `triangle`, `gmsh`.

## Installation

```bash
# 1. Clone the viewer (or drop it beside your TELEMAC tree)
git clone https://github.com/razinkele/telemac-viewer.git
cd telemac-viewer

# 2. Create a Python 3.11+ environment (micromamba, conda, or venv)
micromamba create -n shiny python=3.13 -y
micromamba activate shiny

# 3. Install Python dependencies
pip install "shiny>=1.5.1" "shiny-deckgl>=1.0.1" numpy scipy pyproj
# Optional for HEC-RAS import:
pip install h5py rasterio triangle gmsh

# 4. Point the viewer at your TELEMAC installation
export HOMETEL=/path/to/telemac-v8p5r1
```

## Quick start

```bash
cd telemac-viewer
shiny run app.py --port 8765
```

Open <http://localhost:8765> in your browser, then pick an example from the
dropdown or upload your own `.slf`.

## Usage

1. **Select an example** (30+ across 6 TELEMAC modules) or **upload** a `.slf` file.
2. **Choose a variable** and timestep via slider or playback controls.
3. **Click the map** to probe values, start a time series, or begin a cross-section.
4. Use the **Analysis** accordion panels for advanced tools (flood mapping, discharge, particles, zonal stats).
5. **Export** time series and cross-section data as CSV.

### Comparing two result files

Use the **Compare upload** control to overlay a second `.slf` on the same mesh.
The viewer hashes `(x, y, ikle)` of both files and refuses to proceed if the
geometry differs — preventing silent rendering of file B values on file A's
triangles even when node and element counts happen to match.

### CRS setup

For geo-referenced results, the viewer auto-detects CRS from the `.cas` file.
You can also:
- Enter an EPSG code manually in the CRS panel
- Apply a manual origin offset for pre-centered meshes
- The basemap aligns automatically once a CRS is set

## Configuration

Environment variables consumed by the viewer (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `HOMETEL` | `/home/razinka/telemac/telemac-v8p5r1` | TELEMAC installation root |
| `SYSTELCFG` | `$HOMETEL/configs/systel.local.cfg` | TELEMAC compile config |
| `USETELCFG` | `gfortran.intelmpi` | Active build name |
| `I_MPI_FABRICS` | `shm:ofi` (recommended) | Intel MPI transport; required to avoid `OFI get address vector map failed` on Ubuntu 24.04 |
| `I_MPI_OFI_PROVIDER` | `tcp` (recommended) | Intel MPI OFI provider |

A sample `.env` for the whole TELEMAC project is maintained one level up from
the viewer; copy, edit, and `source` it before launching:

```bash
source ../.env
shiny run app.py --port 8765
```

Nothing in the viewer code auto-loads `.env` — wire up `python-dotenv` in
`app.py` if you want that behavior.

## Multi-user setup

The viewer ships with a built-in account system so a small lab (2–10 users)
can share one deployment without re-uploading data per person. Authentication
is mandatory — every request goes through a Starlette ASGI middleware that
redirects unauthenticated traffic to `/login`.

State lives in two files under `~/.telemac-viewer/` (mode `0o700`):

| File | Mode | Purpose |
|---|---|---|
| `auth.db` | `0o600` | sqlite (WAL) — `users` table only |
| `auth_secret` | `0o600` | 32 random bytes used to sign session cookies |

Both are created on first launch. Override the location by setting
`TELEMAC_VIEWER_AUTH_DIR` before starting the app.

### Deployment context (read this first)

CSRF tokens and per-IP rate limits are intentionally **not** implemented in
v1. The waivers depend on the viewer being deployed on a private host:

- **Not publicly bound.** Run on `127.0.0.1` or an RFC1918 LAN interface; if
  you must expose it via reverse proxy (nginx, gunicorn binding `0.0.0.0`),
  add an ACL or VPN gate at the proxy layer. The app logs a `WARNING` on
  startup if it detects a non-loopback / non-RFC1918 bind.
- **No shared registrable domain** with another web app you don't control.
  The session cookie uses `SameSite=Lax`, which protects against cross-site
  POSTs but not against same-eTLD+1 sibling apps.

Outside that envelope (public-internet deployment, multi-tenant cloud,
shared corporate domain), wrap the viewer with an external auth proxy
instead of relying on this module.

### Bootstrap the first admin

```bash
cd telemac-viewer
python -m auth.cli create-admin --username arturas --display-name "Arturas R."
# password prompted twice via getpass; refused if stdin is not a tty
```

Refuses if any admin already exists — exit code `2` ("An admin already
exists; use the /admin UI to add more users"). Once you have one admin,
all further user management happens at `/admin/users` in the browser.

#### Non-interactive (`--password-file`) for CI

```bash
echo -n 'correcthorsebatterystaple' > pwfile
chmod 0600 pwfile
python -m auth.cli create-admin --username svc --password-file pwfile
```

> **Trailing-newline trap.** `echo "secret" > pwfile` (no `-n`) writes
> `secret\n`. The CLI strips a single trailing CR/LF (so the typical
> mistake doesn't lock you out), but other trailing whitespace is treated
> as part of the password. Prefer `printf '%s' "$pw" > pwfile` or
> `echo -n` and verify with `xxd pwfile | tail -1`.

The password file is rejected if its mode isn't exactly `0o600` (exit `5`).

#### Reset a forgotten password

```bash
python -m auth.cli reset-password --username arturas
# same tty / --password-file rules
# exit 4 if user not found, 5 for stdin/file-mode errors
```

This does **not** invalidate existing sessions — see "Incident response" if
that's what you need.

### Admin UI

Visit `/admin/users` while logged in as an admin. You can:

- Create users (admin or non-admin)
- Edit display name / role
- Reset a user's password without knowing the old one
- Delete users — except the last remaining admin (refused atomically inside
  a transaction; the guard is not advisory)

Templates use Jinja2 with `autoescape=True`, so usernames and display
names containing HTML metacharacters render as text, never as markup.

### Per-user preferences

The "Account" accordion in the sidebar has a **Save current view as my
preferences** button that stores the current variable, palette, and basemap
choice on your user row. The next time you log in, those three knobs are
restored automatically. (The variable preference is silently skipped — with
an INFO log — when the saved variable isn't in the file you've loaded.)

### Secret rotation

Rotating `auth_secret` invalidates **every** active session — every user
will be redirected to `/login` on their next request. Run when you suspect
the secret has leaked, or as part of routine credential hygiene:

```bash
rm ~/.telemac-viewer/auth_secret
systemctl restart telemac-viewer   # or however you launch the app
```

A fresh 32-byte secret is generated on the next launch (mode `0o600`).
There is intentionally no `rotate-secret` CLI subcommand — the file-system
operation is the contract.

### Incident response

If you suspect an account is compromised:

1. **Reset the password.** Locks out the attacker if they only have the
   password.
   ```bash
   python -m auth.cli reset-password --username <victim>
   ```
2. **Rotate the secret.** Required to evict the attacker's *live* session
   cookie — password reset alone doesn't invalidate already-issued cookies.
   ```bash
   rm ~/.telemac-viewer/auth_secret
   systemctl restart telemac-viewer
   ```
3. **Audit `users.last_login_at`** in `auth.db` for unfamiliar timestamps.

If you suspect the attacker created an account, log in as admin and delete
it from `/admin/users` *before* step 2 (otherwise they get logged out
mid-rampage but the row remains).

### Schema mismatch recovery

The v1 schema has no migration tool. If you upgrade the viewer to a future
version that changes the `users` table, the app refuses to start with a
message like:

```
ERROR: auth.db schema does not match v1 expectations.
       Found columns: ...
       Expected:      ...
       v1 has no migration tool. Recovery options:
         (a) back up auth.db, remove it, and re-run
             `python -m auth.cli create-admin`, or
         (b) downgrade telemac-viewer to the previous version.
```

For option (a):

```bash
cp ~/.telemac-viewer/auth.db ~/.telemac-viewer/auth.db.bak.$(date +%F)
rm ~/.telemac-viewer/auth.db
python -m auth.cli create-admin --username <yours>
# every other user must be re-created from /admin/users
```

Preferences and history are lost; passwords from the backup are not
portable across schema changes.

## Per-user storage

Each authenticated user gets a private, persistent library at
`~/.telemac-viewer/users/<user_id>/models/` (mode `0o700`). Uploaded
`.slf` files and HEC-RAS conversion outputs survive across sessions
without re-uploading. The existing `~/.telemac-viewer/models/` directory
becomes a read-only **shared overlay** every user sees alongside their
own "My models" group.

### Deployment context (read this first)

If `/home` is on a small partition, set `TELEMAC_VIEWER_USERS_ROOT` to a
larger mount **before first launch**:

```bash
export TELEMAC_VIEWER_USERS_ROOT=/data/telemac-viewer/users
systemctl restart telemac-viewer
```

`TELEMAC_VIEWER_USERS_ROOT` and `TELEMAC_VIEWER_MODELS` **must be
disjoint** — neither may be a parent of the other, and they must not be
the same directory. The app refuses to start otherwise: collapsing the
two would let a non-admin write into the shared overlay, defeating the
read-only invariant. Pick distinct roots like
`/data/telemac-viewer/users/` and `/srv/telemac-viewer/shared-models/`.

### Per-user directory layout

```
~/.telemac-viewer/
├── auth.db                  (0600 — existing)
├── auth_secret              (0600 — existing)
├── models/                  (0700 — existing — now shared, read-only overlay)
│   └── <project>/
└── users/                   (0700 — new)
    └── <uid>/               (0700)
        └── models/          (0700)
            └── <project>/
                ├── case.slf
                ├── case.cas
                ├── .lock                       (fcntl per-project)
                └── .case.partial-<pid>/        (atomic-save staging)
```

- `.lock` is a per-project fcntl lockfile (`save_upload` serializes
  same-name saves so two browser tabs racing the same "Save as…" name
  can't corrupt each other).
- `.<name>.partial-<pid>/` is the staging directory used by the atomic
  rename pattern: write into staging, `os.rename` into place, then
  remove staging. A SIGKILL'd save leaves staging behind; the in-process
  sweep at startup removes up to 5 stale partials per user, per launch.

### What persists vs not

| Action | Persists? |
|---|---|
| Upload `.slf` + click **Save to my library** | Yes — moved to `users/<uid>/models/<name>/` |
| Upload `.slf` without clicking save | No — cleared on session end (Shiny tempdir) |
| HEC-RAS Convert output | Yes — auto-saved on conversion success |
| Raw HDF input to HEC-RAS Convert | No — discarded after conversion; re-upload to re-convert |
| "Save current view as my preferences" | Yes — on `auth.db` user row (existing v3.6.0 behavior) |

### Save-to-library button

The upload accordion now has a **Save as…** textbox and a **Save to my
library** button. Type a project name (letters, digits, `_`, `-`, `.`,
1–64 chars; no leading dot, no `..`), click the button, and the uploaded
`.slf` + any companion `.cas`/`.cli`/`.liq` are atomically moved into
`users/<uid>/models/<name>/`. The project then appears under **My
models** in the library dropdown. Re-using an existing name shows a
"already exists — pick a new name" toast; the save is refused
(prevents accidental clobber).

### Cascade-delete behavior

Admin `Delete` on `/admin/users/<uid>` is now a **two-step confirm**:

1. **GET `/admin/users/<uid>/delete`** renders a confirmation page that
   shows the user's saved library size (`du -sh`) and the resolved
   absolute path that will be removed. A signed `confirm_token`
   (`itsdangerous.URLSafeTimedSerializer`, distinct salt from the
   session cookie, 10-minute window) is embedded in the form.
2. **POST** with the matching token verifies the signature, captures the
   username for audit logs, removes the `auth.db` row inside the
   existing last-admin-guard transaction, then cascade-`rmtree`s
   `users/<uid>/`.

If you're uncertain, copy the dir aside before clicking Delete:

```bash
cp -r ~/.telemac-viewer/users/<uid>/ /backup/uid-<uid>-$(date +%F)/
```

A direct POST with no GET, or with a tampered token, returns 400 — the
two-step flow is not optional.

### Monitoring & log patterns

```bash
# User N hitting save errors:
journalctl -u telemac-viewer | grep -E "(save_upload|auto-save).*uid=N"
# Disk-full events:
journalctl -u telemac-viewer | grep ENOSPC
# Per-user disk usage:
du -sh ~/.telemac-viewer/users/*/
```

Every save logs a structured line including `uid=`, `project=`, and
either `bytes=` (success) or `error=` (failure). `ENOSPC` is logged at
ERROR level and surfaces a user-facing "disk full — contact admin"
toast; the partial dir is removed before the error returns.

### Manual library surgery

When a user asks to wipe one project without deleting their account:

```bash
# Safe while the app is running on Linux (inode semantics).
# User clicks ↻ in the browser to see the dropdown update.
rm -rf ~/.telemac-viewer/users/<uid>/models/<project>/
```

### Backup & restore — the sync invariant

`auth.db` is the source of truth; `users/<uid>/` is best-effort
persistent storage anchored to it. The four states:

| `auth.db` row | `users/<uid>/` dir | Meaning |
|---|---|---|
| present | present | Normal user with saved content |
| present | absent | Normal user who has never saved (created on first save) |
| absent  | present | **Orphan** — admin deleted user but cascade failed (or out-of-order restore). Safe to `rm -rf` after audit. |
| absent  | absent | User never existed or was cleanly deleted |

The dangerous state — `auth.db` rolled back **past** a uid that already
has files — would let SQLite re-issue that uid to a new user, mapping
them onto someone else's old data.

**Ordering rule:** always back up `auth.db` **BEFORE or WITH, but never
AFTER** `users/`. Never restore `auth.db` to a state older than `users/`
without first auditing for orphans.

```bash
# Consistent backup (run while service is stopped, or accept WAL semantics):
sqlite3 ~/.telemac-viewer/auth.db ".backup /backup/auth-$(date +%F).db"
tar -czf /backup/users-$(date +%F).tar.gz -C ~/.telemac-viewer users
```

**Find orphan dirs after a partial restore:**

```bash
comm -23 \
    <(ls ~/.telemac-viewer/users/ | sort) \
    <(sqlite3 ~/.telemac-viewer/auth.db "SELECT id FROM users" | sort)
# For each id printed, inspect then remove:
#   rm -rf ~/.telemac-viewer/users/<id>/
```

### Migration from the existing shared library

To move a project out of the shared overlay (`~/.telemac-viewer/models/`)
into a specific user's private library:

```bash
sudo systemctl stop telemac-viewer
# Verify the target uid exists:
sqlite3 ~/.telemac-viewer/auth.db "SELECT id, username FROM users WHERE id=<uid>"
# Create the parent dirs with correct mode:
mkdir -p ~/.telemac-viewer/users/<uid>/models/
chmod 0700 ~/.telemac-viewer/users/<uid> ~/.telemac-viewer/users/<uid>/models
# Move (or `cp -a` if you want it visible in both, but then the shared
# copy stays read-only and the user's copy is the one they edit):
mv ~/.telemac-viewer/models/<project> ~/.telemac-viewer/users/<uid>/models/<project>
sudo systemctl start telemac-viewer
# User sees the project under "My models" on next dropdown refresh.
```

Common pitfalls:

- **Typo in `<uid>`** → silently creates a folder under a non-existent
  uid (becomes an orphan). Verify the `SELECT` above first.
- **Forgetting `chmod 0700`** on `users/<uid>` and `users/<uid>/models/`
  after a manual `mkdir` → leaks the user's project names to other OS
  users. The service running as one user makes this only matter on
  multi-user hosts, but the model invariant says `0o700` everywhere.

### Recovery from a corrupted user dir

**Symptom:** the library dropdown shows `⚠ library unreadable` next to
the user's row (or scoped to one project name); the affected user's
saves return a "library unreadable" toast.

**Procedure:**

```bash
ls -la ~/.telemac-viewer/users/<uid>/models/
# Diagnose: regular file where a dir should be? bad permissions?
# broken symlink? wrong owner?
mv ~/.telemac-viewer/users/<uid>/models/ ~/.telemac-viewer/users/<uid>/models.broken-$(date +%F)/
# Restart not required — the app re-creates the dir on next save.
```

The user resumes from an empty library. Recoverable data can be pulled
out of `models.broken-*/` offline.

### Stale `.partial-*` cleanup

**Symptom:** `du -sh ~/.telemac-viewer/users/*/` shows GBs occupied by
`.*.partial-*` directories. Cause: SIGKILL or OOM during a save, before
the atomic rename completed.

The in-process sweep clears up to 5 stale partials per user per startup.
For larger backlogs (rare; usually after a crash storm):

```bash
find ~/.telemac-viewer/users -maxdepth 3 -name '.*.partial-*' -mtime +1 -exec rm -rf {} +
```

The `-mtime +1` guard is intentional: a save currently in progress
creates a `.partial-*` that is less than a day old. Don't drop the
guard.

## Project structure

```
telemac-viewer/
├── app.py                 # Main Shiny app (UI layout + server orchestration)
├── server_core.py         # Core reactive calcs (file loading, mesh, variables)
├── server_analysis.py     # Analysis panel handlers (charts, stats, exports)
├── server_playback.py     # Animation playback controls
├── server_simulation.py   # TELEMAC simulation launcher
├── server_import.py       # HEC-RAS import tab handlers
├── geometry.py            # Mesh geometry builder (binary-encoded for deck.gl)
├── layers.py              # deck.gl layer builders (mesh, velocity, contours, …)
├── analysis.py            # Spatial/temporal analysis functions
├── crs.py                 # CRS transforms and .cas detection
├── model_library.py       # Local model library — filesystem-backed project folders
├── constants.py           # Examples, palettes, environment setup
├── telemac_defaults.py    # Variable semantics, module detection, velocity pairs
├── validation.py          # Observation parsing, RMSE, NSE, .liq parser
├── viewer_types.py        # Shared types (MeshGeometry, TelemacFileProtocol)
├── auth/                  # Multi-user authentication
│   ├── core.py            #   sqlite + users CRUD + schema verification
│   ├── crypto.py          #   bcrypt + cookie sign/verify (NULL_HASH timing equalization)
│   ├── middleware.py      #   ASGI function + scope accessors + warn_if_public_bind
│   ├── routes.py          #   /login, /logout, /admin/* (Jinja2 + autoescape)
│   └── cli.py             #   create-admin, reset-password
├── telemac_tools/         # HEC-RAS import pipeline
│   ├── model.py           #   Data model (Reach, Mesh2D, BoundaryCondition)
│   ├── hecras/            #   HEC-RAS HDF5 parsers (1D, 2D, BC time series)
│   ├── domain/            #   Domain builder (DEM sampling, channel carving)
│   ├── meshing/           #   Mesh generation (Triangle, Gmsh backends)
│   └── telemac/           #   SELAFIN / CLI / CAS / LIQ file writers
├── tests/                 # pytest suite (603 tests)
├── docs/
│   ├── API.md             # Module-level API reference
│   ├── plans/             # Design documents for features
│   └── specs/             # Technical specifications
├── release.py             # Version-bumping, changelog generation, git tagging
├── CHANGELOG.md           # Release history
└── VERSION                # Single-source version file
```

## Development

```bash
# Run the app with hot reload
shiny run app.py --port 8765 --reload

# Follow the release workflow for a new version
python release.py prep minor        # dry-run: see what 3.2.0 → 3.3.0 would contain
python release.py bump minor        # bump VERSION
python release.py tag 3.3.0         # tag and create release commit
```

## Testing

```bash
cd telemac-viewer
python -m pytest tests/ -v
```

The suite currently runs **533 tests** and is expected to pass with no
`RuntimeWarning`:

```bash
python -m pytest tests/ -W error::RuntimeWarning
```

## Documentation

- [CHANGELOG.md](./CHANGELOG.md) — release history
- [docs/API.md](./docs/API.md) — module-level API reference
- [docs/plans](./docs/plans/) — design documents
- [docs/specs](./docs/specs/) — technical specifications

## Contributing

1. Fork the repository and create a branch from `master`.
2. Keep changes focused; add tests for new behavior (`tests/test_round<N>_*.py` is the conventional pattern — see recent rounds for examples).
3. Ensure `pytest tests/ -W error::RuntimeWarning` is clean.
4. Update `CHANGELOG.md` under `## [Unreleased]`.
5. Open a pull request describing the change, the tests added, and any follow-up work.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md). The current release is **v3.5.0**
(2026-04-27) — adds a local model library: drop project folders
into `~/.telemac-viewer/models/` (or set `TELEMAC_VIEWER_MODELS`)
and they appear in a "My models" sidebar dropdown alongside Examples
and Upload, with auto-scanning for `.slf` plus optional `.cas` /
`.cli` / `.liq` companions.

## License

Part of the TELEMAC system, which is released under the **LGPL v2.1**.
See the main TELEMAC license for terms:
<http://www.opentelemac.org/index.php/license>.
