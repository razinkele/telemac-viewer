"""Tests with real HEC-RAS files — skipped when fixtures not available.

To run: place .g##.hdf files in telemac_tools/tests/fixtures/
Download from: https://github.com/fema-ffrd/rashdf (requires git lfs)
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pytest
from telemac_tools.hecras import parse_hecras
from telemac_tools.model import HecRasModel

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
HDF_FILES = glob.glob(os.path.join(FIXTURE_DIR, "*.hdf"))

pytestmark = pytest.mark.skipif(
    not HDF_FILES,
    reason="No real HEC-RAS fixtures in telemac_tools/tests/fixtures/",
)


@pytest.fixture(params=HDF_FILES, ids=lambda p: os.path.basename(p))
def hdf_path(request):
    return request.param


class TestRealHecRasFiles:
    def test_parses_without_error(self, hdf_path):
        model = parse_hecras(hdf_path)
        assert isinstance(model, HecRasModel)
        assert model.rivers or model.areas_2d

    def test_has_geometry(self, hdf_path):
        model = parse_hecras(hdf_path)
        if model.rivers:
            for r in model.rivers:
                assert r.alignment.shape[1] == 2
                for xs in r.cross_sections:
                    assert xs.coords.shape[1] == 3
        if model.areas_2d:
            for a in model.areas_2d:
                assert a.face_points.shape[1] == 2
                assert len(a.cells) > 0

    def test_full_pipeline_2d(self, hdf_path, tmp_path):
        """Full conversion for 2D models (1D needs DEM)."""
        model = parse_hecras(hdf_path)
        if not model.areas_2d:
            pytest.skip("No 2D areas — 1D needs DEM for full pipeline")
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(hdf_path, output_dir=out)
        assert os.path.isfile(os.path.join(out, "project.slf"))
