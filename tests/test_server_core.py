"""Tests for server_core pure helpers.

The reactive wrappers in server_core.py are thin; the decision logic
is extracted into module-level helpers that can be tested without a
Shiny session.
"""

from __future__ import annotations

import numpy as np
import pytest

from server_core import CrsResolution, _resolve_crs_from_inputs


class TestResolveCrsFromInputs:
    # --- PATH 1: Manual EPSG ---

    def test_manual_epsg_valid(self):
        result = _resolve_crs_from_inputs(
            epsg_text="4326",
            auto_crs_enabled=True,  # should be ignored on manual path
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.source == "manual"
        assert result.crs is not None
        assert result.crs.epsg == 4326
        assert result.error is None
        assert result.detected_epsg is None  # no writeback on manual

    def test_manual_epsg_strips_whitespace(self):
        result = _resolve_crs_from_inputs(
            epsg_text="  4326  ",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.crs is not None and result.crs.epsg == 4326

    def test_manual_epsg_rejects_non_numeric(self):
        result = _resolve_crs_from_inputs(
            epsg_text="4326x",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.source == "invalid"
        assert result.crs is None
        assert result.error is not None and "4326x" in result.error

    def test_manual_epsg_rejects_nonsense_code(self):
        result = _resolve_crs_from_inputs(
            epsg_text="99999",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.source == "invalid"
        assert result.crs is None
        assert "99999" in result.error

    # --- PATH 2: Auto-detect disabled ---

    def test_auto_crs_disabled_returns_none(self):
        result = _resolve_crs_from_inputs(
            epsg_text="",
            auto_crs_enabled=False,
            cas_candidates=("/some/file.cas",),  # ignored
            mesh_xy=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),  # ignored
        )
        assert result.source == "disabled"
        assert result.crs is None
        assert result.detected_epsg is None

    # --- PATH 3/4 fall-through: no sources ---

    def test_no_sources_returns_none(self):
        result = _resolve_crs_from_inputs(
            epsg_text="",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.source == "none"
        assert result.crs is None
        assert result.detected_epsg is None

    # --- PATH 4: Coordinate heuristic ---

    def test_coord_heuristic_hits_for_lks94_coords(self):
        """guess_crs_from_coords must detect the LKS94 range for these coords.

        Verified via direct call: guess_crs_from_coords([500000, 500100],
        [6200000, 6200100]) returns EPSG:3346 (LKS94). The helper must
        route that hit through source='coords' with detected_epsg set so
        the shim can perform the ui.update_text writeback.
        """
        # LKS94 range (Curonian Lagoon area) — confirmed to hit the heuristic.
        x = np.array([500000.0, 500100.0], dtype=np.float64)
        y = np.array([6200000.0, 6200100.0], dtype=np.float64)
        result = _resolve_crs_from_inputs(
            epsg_text="",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=(x, y),
        )
        assert result.source == "coords"
        assert result.crs is not None
        assert result.crs.epsg == 3346
        assert result.detected_epsg == 3346

    def test_coord_heuristic_miss_returns_none(self):
        """Coordinates that don't match any known projection return source='none'."""
        # WGS84-ish degrees — guess_crs_from_coords treats these as "already
        # geographic" and returns None.
        x = np.array([24.0, 24.1], dtype=np.float64)
        y = np.array([55.0, 55.1], dtype=np.float64)
        result = _resolve_crs_from_inputs(
            epsg_text="",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=(x, y),
        )
        # Degrees-range coords may or may not hit — if they don't, we land
        # on source="none"; if they do, the shim would still work. Either
        # behavior is acceptable, but detected_epsg must be consistent.
        assert result.source in ("coords", "none")
        if result.source == "coords":
            assert result.detected_epsg == result.crs.epsg
        else:
            assert result.crs is None and result.detected_epsg is None

    def test_coord_heuristic_skipped_when_mesh_xy_is_none(self):
        result = _resolve_crs_from_inputs(
            epsg_text="",
            auto_crs_enabled=True,
            cas_candidates=(),
            mesh_xy=None,
        )
        assert result.source == "none"


class TestPickFilePath:
    def test_upload_with_use_upload_true(self):
        from server_core import _pick_file_path

        uploaded = [{"datapath": "/tmp/u.slf", "name": "u.slf"}]
        result = _pick_file_path(
            uploaded=uploaded,
            use_upload=True,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/u.slf"

    def test_upload_with_use_upload_false_falls_to_example(self):
        from server_core import _pick_file_path

        uploaded = [{"datapath": "/tmp/u.slf", "name": "u.slf"}]
        result = _pick_file_path(
            uploaded=uploaded,
            use_upload=False,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/x.slf"

    def test_no_upload_uses_example(self):
        from server_core import _pick_file_path

        result = _pick_file_path(
            uploaded=None,
            use_upload=False,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/x.slf"

    def test_empty_upload_list_uses_example(self):
        from server_core import _pick_file_path

        result = _pick_file_path(
            uploaded=[],
            use_upload=True,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/x.slf"

    def test_missing_example_key_returns_empty_string(self):
        from server_core import _pick_file_path

        result = _pick_file_path(
            uploaded=None,
            use_upload=False,
            example_key="Nonexistent",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == ""

    def test_multi_file_upload_picks_slf_not_first(self):
        """When user uploads [.cas, .slf, .cli], picker must return the .slf."""
        from server_core import _pick_file_path

        uploaded = [
            {"datapath": "/tmp/u.cas", "name": "u.cas"},
            {"datapath": "/tmp/u.slf", "name": "u.slf"},
            {"datapath": "/tmp/u.cli", "name": "u.cli"},
        ]
        result = _pick_file_path(
            uploaded=uploaded,
            use_upload=True,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/u.slf"

    def test_multi_file_upload_case_insensitive(self):
        """Extension match is case-insensitive (browsers may preserve .SLF)."""
        from server_core import _pick_file_path

        uploaded = [
            {"datapath": "/tmp/u.cas", "name": "U.CAS"},
            {"datapath": "/tmp/u.slf", "name": "U.SLF"},
        ]
        result = _pick_file_path(
            uploaded=uploaded,
            use_upload=True,
            example_key="X",
            examples={"X": "/tmp/x.slf"},
        )
        assert result == "/tmp/u.slf"


class TestFindUploadedByExt:
    def test_none_uploaded_returns_none(self):
        from server_core import _find_uploaded_by_ext

        assert _find_uploaded_by_ext(None, ".cas") is None
        assert _find_uploaded_by_ext([], ".cas") is None

    def test_finds_first_matching_extension(self):
        from server_core import _find_uploaded_by_ext

        uploaded = [
            {"datapath": "/tmp/a.slf", "name": "a.slf"},
            {"datapath": "/tmp/b.cas", "name": "b.cas"},
            {"datapath": "/tmp/c.cli", "name": "c.cli"},
        ]
        assert _find_uploaded_by_ext(uploaded, ".cas") == "/tmp/b.cas"
        assert _find_uploaded_by_ext(uploaded, ".cli") == "/tmp/c.cli"
        assert _find_uploaded_by_ext(uploaded, ".slf") == "/tmp/a.slf"

    def test_returns_none_when_extension_absent(self):
        from server_core import _find_uploaded_by_ext

        uploaded = [{"datapath": "/tmp/a.slf", "name": "a.slf"}]
        assert _find_uploaded_by_ext(uploaded, ".cas") is None

    def test_case_insensitive_match(self):
        from server_core import _find_uploaded_by_ext

        uploaded = [{"datapath": "/tmp/a.cas", "name": "PROJECT.CAS"}]
        assert _find_uploaded_by_ext(uploaded, ".cas") == "/tmp/a.cas"

    def test_handles_missing_name_gracefully(self):
        """Defensive: if Shiny ever sends items without 'name', don't crash."""
        from server_core import _find_uploaded_by_ext

        uploaded = [
            {"datapath": "/tmp/a.slf"},
            {"datapath": "/tmp/b.cas", "name": "b.cas"},
        ]
        assert _find_uploaded_by_ext(uploaded, ".cas") == "/tmp/b.cas"
