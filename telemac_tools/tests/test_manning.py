"""Tests for Manning's n extraction."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.manning import extract_mannings_1d


class TestExtractMannings1d:
    def test_returns_three_values_per_xs(self, hdf_1d):
        result = extract_mannings_1d(hdf_1d)
        assert len(result) == 3  # 3 cross-sections
        for n_vals in result:
            assert len(n_vals) == 3
            assert all(n > 0 for n in n_vals)

    def test_correct_values(self, hdf_1d):
        result = extract_mannings_1d(hdf_1d)
        assert result[0] == [0.06, 0.035, 0.06]
