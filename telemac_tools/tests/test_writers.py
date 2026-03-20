"""Tests for TELEMAC file writers."""
from __future__ import annotations
import os
import numpy as np
import pytest
from telemac_tools.model import Mesh2D, TelemacDomain, BCSegment
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.telemac.writer_cli import write_cli
from telemac_tools.telemac.writer_cas import write_cas


def _simple_mesh():
    return Mesh2D(
        nodes=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        elevation=np.array([0.0, 1.0, 0.5, 1.5]),
        mannings_n=np.array([0.035, 0.035, 0.035, 0.035]),
    )


def _simple_domain():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)
    return TelemacDomain(
        boundary_polygon=poly,
        bc_segments=[
            BCSegment(node_indices=[0, 2], lihbor=5, prescribed_h=2.0),
            BCSegment(node_indices=[1, 3], lihbor=4),
        ],
    )


class TestWriteSlf:
    def test_creates_file(self, tmp_path):
        write_slf(_simple_mesh(), str(tmp_path / "test.slf"))
        assert (tmp_path / "test.slf").exists()
        assert (tmp_path / "test.slf").stat().st_size > 0

    def test_roundtrip(self, tmp_path):
        write_slf(_simple_mesh(), str(tmp_path / "test.slf"))
        from data_manip.extraction.telemac_file import TelemacFile
        tf = TelemacFile(str(tmp_path / "test.slf"))
        assert tf.npoin2 == 4
        assert tf.nelem2 == 2
        assert "BOTTOM" in tf.varnames
        tf.close()


class TestWriteCli:
    def test_creates_file(self, tmp_path):
        write_cli(_simple_mesh(), _simple_domain(), str(tmp_path / "test.cli"))
        assert (tmp_path / "test.cli").exists()

    def test_correct_columns(self, tmp_path):
        write_cli(_simple_mesh(), _simple_domain(), str(tmp_path / "test.cli"))
        lines = (tmp_path / "test.cli").read_text().strip().split("\n")
        for line in lines:
            assert len(line.split()) == 13


class TestWriteCas:
    def test_creates_file(self, tmp_path):
        write_cas(str(tmp_path / "test.cas"), name="project")
        assert (tmp_path / "test.cas").exists()

    def test_contains_geometry(self, tmp_path):
        write_cas(str(tmp_path / "test.cas"), name="project")
        text = (tmp_path / "test.cas").read_text()
        assert "GEOMETRY FILE" in text
        assert "project.slf" in text

    def test_overrides(self, tmp_path):
        write_cas(str(tmp_path / "test.cas"), name="project", overrides={"TURBULENCE MODEL": 4})
        text = (tmp_path / "test.cas").read_text()
        assert "TURBULENCE MODEL = 4" in text
