"""Tests for telemac_defaults module."""
from __future__ import annotations
import pytest
from telemac_defaults import (
    suggest_palette, is_bipolar, detect_module_from_vars, find_velocity_pair,
)


class TestSuggestPalette:
    def test_water_depth(self):
        assert suggest_palette("WATER DEPTH") == "Ocean"

    def test_hauteur_d_eau(self):
        assert suggest_palette("HAUTEUR D EAU") == "Ocean"

    def test_free_surface(self):
        assert suggest_palette("FREE SURFACE") == "Ocean"

    def test_bottom(self):
        assert suggest_palette("BOTTOM") == "Plasma"

    def test_evolution_diverging(self):
        assert suggest_palette("BED EVOLUTION") == "_diverging"

    def test_temperature(self):
        assert suggest_palette("TEMPERATURE") == "Thermal"

    def test_unknown_returns_none(self):
        assert suggest_palette("CUSTOM_VARIABLE_XYZ") is None

    def test_case_insensitive(self):
        assert suggest_palette("water depth") == "Ocean"
        assert suggest_palette("Water Depth") == "Ocean"


class TestIsBipolar:
    def test_evolution_bipolar(self):
        assert is_bipolar("EVOLUTION") is True

    def test_bed_evolution_bipolar(self):
        assert is_bipolar("BED EVOLUTION") is True

    def test_water_depth_not_bipolar(self):
        assert is_bipolar("WATER DEPTH") is False

    def test_velocity_not_bipolar(self):
        assert is_bipolar("VELOCITY U") is False


class TestDetectModuleFromVars:
    def test_tomawac(self):
        assert detect_module_from_vars(["HM0", "DMOY", "WATER DEPTH"]) == "TOMAWAC"

    def test_gaia(self):
        assert detect_module_from_vars(["EVOLUTION", "TOB", "WATER DEPTH"]) == "GAIA"

    def test_artemis(self):
        assert detect_module_from_vars(["PHAS", "QB"]) == "ARTEMIS"

    def test_default_telemac2d(self):
        assert detect_module_from_vars(["WATER DEPTH", "VELOCITY U", "VELOCITY V"]) == "TELEMAC-2D"

    def test_empty_list(self):
        assert detect_module_from_vars([]) == "TELEMAC-2D"


class TestFindVelocityPair:
    def test_standard_pair(self):
        result = find_velocity_pair(["WATER DEPTH", "VELOCITY U", "VELOCITY V"])
        assert result == ("VELOCITY U", "VELOCITY V")

    def test_ux_uy_pair(self):
        result = find_velocity_pair(["WATER DEPTH", "UX", "UY"])
        assert result == ("UX", "UY")

    def test_sediment_transport_pair(self):
        result = find_velocity_pair(["QSBLX", "QSBLY", "EVOLUTION"])
        assert result == ("QSBLX", "QSBLY")

    def test_missing_returns_none(self):
        assert find_velocity_pair(["WATER DEPTH", "BOTTOM"]) is None

    def test_partial_pair_returns_none(self):
        assert find_velocity_pair(["VELOCITY U", "WATER DEPTH"]) is None
