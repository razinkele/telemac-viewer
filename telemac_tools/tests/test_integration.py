"""End-to-end integration test: HEC-RAS HDF5 → TELEMAC files."""
from __future__ import annotations
import os
import numpy as np
import pytest


class TestFullPipeline1d:
    def test_hecras_to_telemac(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(
            hecras_path=hdf_1d,
            dem_path=fake_dem,
            output_dir=out,
            floodplain_width=80.0,
        )
        assert os.path.isfile(os.path.join(out, "project.slf"))
        assert os.path.isfile(os.path.join(out, "project.cli"))
        assert os.path.isfile(os.path.join(out, "project.cas"))

    def test_slf_readable(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(hecras_path=hdf_1d, dem_path=fake_dem, output_dir=out,
                          floodplain_width=80.0)
        from data_manip.extraction.telemac_file import TelemacFile
        tf = TelemacFile(os.path.join(out, "project.slf"))
        assert tf.npoin2 > 0
        assert tf.nelem2 > 0
        assert "BOTTOM" in tf.varnames
        # SLF format truncates variable names to 16 characters
        assert any("FRICTION" in v for v in tf.varnames)
        tf.close()

    def test_cli_has_boundary_nodes(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(hecras_path=hdf_1d, dem_path=fake_dem, output_dir=out,
                          floodplain_width=80.0)
        cli_text = open(os.path.join(out, "project.cli")).read()
        lines = cli_text.strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            assert len(line.split()) == 13


class TestFullPipeline2d:
    def test_2d_to_telemac(self, hdf_2d, tmp_path):
        from telemac_tools.hecras import parse_hecras_2d, triangulate_2d_area
        from telemac_tools.domain import build_domain_2d
        from telemac_tools.telemac import write_telemac

        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        domain = build_domain_2d(model)

        out = str(tmp_path / "output_2d")
        write_telemac(mesh, domain, out, name="hecras2d")

        assert os.path.isfile(os.path.join(out, "hecras2d.slf"))
        assert os.path.isfile(os.path.join(out, "hecras2d.cli"))
        assert os.path.isfile(os.path.join(out, "hecras2d.cas"))

    def test_2d_convenience(self, hdf_2d, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output_2d_conv")
        hecras_to_telemac(hecras_path=hdf_2d, output_dir=out, name="auto2d")
        assert os.path.isfile(os.path.join(out, "auto2d.slf"))
