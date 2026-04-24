"""Tests for server_simulation module-name whitelist."""

from __future__ import annotations

from server_simulation import _ALLOWED_TELEMAC_MODULES


def test_whitelist_contains_core_modules():
    assert "telemac2d" in _ALLOWED_TELEMAC_MODULES
    assert "telemac3d" in _ALLOWED_TELEMAC_MODULES


def test_whitelist_rejects_empty_and_nonsense():
    assert "" not in _ALLOWED_TELEMAC_MODULES
    assert "rm" not in _ALLOWED_TELEMAC_MODULES
    assert "../bin/sh" not in _ALLOWED_TELEMAC_MODULES
