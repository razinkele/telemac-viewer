"""Tests for constants module."""
from __future__ import annotations
import numpy as np
import pytest
from constants import format_time, cached_palette_arr, cached_gradient_colors


class TestFormatTime:
    def test_zero(self):
        assert format_time(0.0) == "0.0 s"

    def test_sub_minute(self):
        assert format_time(30.0) == "30.0 s"

    def test_minutes(self):
        result = format_time(90.0)
        assert "1" in result and "min" in result

    def test_hours(self):
        result = format_time(3661.0)
        assert "1 h" in result

    def test_exact_minute(self):
        result = format_time(60.0)
        assert "min" in result


class TestCachedPaletteArr:
    def test_standard_palette_shape(self):
        arr = cached_palette_arr("Viridis")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 4)
        assert arr.dtype == np.uint8

    def test_diverging_palette(self):
        arr = cached_palette_arr("_diverging")
        assert arr.shape == (256, 4)

    def test_reverse_differs(self):
        normal = cached_palette_arr("Viridis", reverse=False)
        rev = cached_palette_arr("Viridis", reverse=True)
        assert not np.array_equal(normal[0], rev[0])

    def test_caching_returns_same_object(self):
        a = cached_palette_arr("Viridis")
        b = cached_palette_arr("Viridis")
        assert a is b


class TestCachedGradientColors:
    def test_returns_list(self):
        colors = cached_gradient_colors("Viridis")
        assert isinstance(colors, list)
        assert len(colors) == 8

    def test_each_entry_is_rgba(self):
        colors = cached_gradient_colors("Viridis")
        for c in colors:
            assert len(c) == 4  # RGBA

    def test_reverse_differs(self):
        normal = cached_gradient_colors("Viridis", reverse=False)
        rev = cached_gradient_colors("Viridis", reverse=True)
        assert normal[0] != rev[0]
