# TELEMAC Viewer

A web-based viewer for [TELEMAC](http://www.opentelemac.org/) simulation results, built with [Shiny for Python](https://shiny.posit.co/py/) and [deck.gl](https://deck.gl/) via [shiny-deckgl](https://github.com/pbs-data-solutions/shiny-deckgl).

Visualize SELAFIN (.slf) result files from TELEMAC-2D, TELEMAC-3D, GAIA, TOMAWAC, ARTEMIS, and KHIONE with interactive maps, time-series analysis, and hydraulic engineering tools.

## Features

### Visualization
- **Mesh rendering** with per-vertex coloring and 10 color palettes (Viridis, Plasma, Ocean, Thermal, etc.)
- **3D elevation** with adjustable vertical exaggeration
- **Velocity arrows** overlay with configurable scale
- **Wireframe** mesh display
- **Contour isolines** with configurable levels
- **Particle tracing** with Lagrangian trajectories and animated trails
- **Boundary node** visualization with .cli file color-coding
- **Min/max extrema** markers
- **Dark mode** and multiple basemap styles (CartoDB, Satellite)
- **Difference mode** between timesteps with diverging palette

### Analysis
- **Time series** at clicked points with CSV export
- **Cross-section profiles** along user-drawn polylines
- **Vertical profiles** for 3D results with layer extraction
- **Polygon zonal statistics** (min, max, mean, std, area)
- **Flood mapping** — envelope, arrival time, duration
- **Discharge computation** across polylines (Q integral)
- **Volume conservation** tracking over time
- **Temporal statistics** (min/max/mean/std over all timesteps)
- **Courant number**, mesh quality, slope, and element area diagnostics
- **Custom expressions** with safe math evaluation (e.g., `VELOCITY_U**2 + VELOCITY_V**2`)
- **Derived variables** — velocity magnitude, Froude number, vorticity

### CRS Awareness
- Auto-detection from TELEMAC `.cas` files (GEOGRAPHIC SYSTEM keyword)
- Supports UTM, Lambert, LKS94, and any EPSG code via manual entry
- Basemap alignment with proper coordinate transforms (pyproj)
- Manual origin offset for pre-centered meshes

### HEC-RAS Import
- Import HEC-RAS 1D and 2D models (`.hdf` format) and convert to TELEMAC
- Mesh generation with Triangle or Gmsh backends
- Automatic Manning's roughness extraction
- Boundary condition mapping and `.liq` file generation
- Preview map with alignment, cross-section, and boundary layers

### Validation
- Upload observation CSV files for model-vs-observed comparison
- RMSE and Nash-Sutcliffe Efficiency (NSE) metrics
- `.liq` liquid boundary file parser

## Requirements

- Python 3.11+
- [shiny](https://pypi.org/project/shiny/) >= 1.5.1
- [shiny-deckgl](https://pypi.org/project/shiny-deckgl/) >= 1.0.1
- [numpy](https://pypi.org/project/numpy/)
- [pyproj](https://pypi.org/project/pyproj/)
- TELEMAC v8p5r1 installation (for `TelemacFile` class and example data)

Optional dependencies for HEC-RAS import:
- [h5py](https://pypi.org/project/h5py/)
- [rasterio](https://pypi.org/project/rasterio/)
- [triangle](https://pypi.org/project/triangle/)
- [gmsh](https://pypi.org/project/gmsh/) (optional mesh backend)

## Quick Start

```bash
# Set TELEMAC environment
export HOMETEL=/path/to/telemac-v8p5r1

# Run the viewer
cd telemac-viewer
shiny run app.py --port 8765
```

Open http://localhost:8765 in your browser.

## Usage

1. **Select an example** from the dropdown (30+ examples across 6 TELEMAC modules) or **upload** your own `.slf` file
2. **Choose a variable** and timestep using the slider or playback controls
3. **Click the map** to probe values, start time series, or begin a cross-section
4. Use the **Analysis** accordion panels for advanced tools (flood mapping, discharge, particles, etc.)
5. **Export** time series and cross-section data as CSV

### CRS Setup

For geo-referenced results, the viewer auto-detects CRS from the `.cas` file. You can also:
- Enter an EPSG code manually in the CRS panel
- Apply a manual origin offset for pre-centered meshes
- The basemap aligns automatically when CRS is set

## Project Structure

```
telemac-viewer/
  app.py              # Main Shiny app (UI layout + server orchestration)
  server_core.py      # Core reactive calculations (file loading, mesh, variables)
  server_analysis.py  # Analysis panel handlers (charts, stats, exports)
  server_playback.py  # Animation playback controls
  server_simulation.py # TELEMAC simulation launcher
  server_import.py    # HEC-RAS import tab handlers
  geometry.py         # Mesh geometry builder (binary encoding for deck.gl)
  layers.py           # deck.gl layer builders (mesh, velocity, contours, etc.)
  analysis.py         # Spatial/temporal analysis functions
  crs.py              # CRS transforms and .cas detection
  constants.py        # Examples, palettes, environment setup
  telemac_defaults.py # Variable semantics, module detection, velocity pairs
  validation.py       # Observation parsing, RMSE, NSE, .liq parser
  viewer_types.py     # Shared types (MeshGeometry, TelemacFileProtocol)
  telemac_tools/      # HEC-RAS import pipeline
    model.py          #   Data model (Reach, Mesh2D, BoundaryCondition, etc.)
    hecras/            #   HEC-RAS HDF5 parsers (1D, 2D, BC timeseries)
    domain/            #   Domain builder (DEM sampling, channel carving)
    meshing/           #   Mesh generation (Triangle, Gmsh backends)
    telemac/           #   SELAFIN/CLI/CAS file writers
  tests/              # pytest test suite (100+ tests)
```

## Running Tests

```bash
cd telemac-viewer
python -m pytest tests/ -v
```

## License

Part of the TELEMAC system. See the main TELEMAC license for terms.
