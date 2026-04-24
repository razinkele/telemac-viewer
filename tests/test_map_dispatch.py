"""Tests for the map-update dispatcher that decides full vs partial update."""

from __future__ import annotations

import pytest
from shiny_deckgl import MapWidget, zoom_widget

from app_dispatch import decide_dispatch
from layers import build_mesh_color_patch
from tests.helpers import FakeSession


def make_sig(
    *,
    file_path: str = "/tmp/a.slf",
    wireframe: bool = False,
    boundary: bool = False,
    vectors: bool = False,
    contours: bool = False,
    extrema: bool = False,
    particles: bool = False,
    diff_mode: bool = False,
    is_3d: bool = False,
    basemap: str = "dark",
    compare_var: str = "",
    overlay_counts: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple:
    """Build a structural signature tuple (see app.py._structural_sig)."""
    return (
        file_path,
        wireframe,
        boundary,
        vectors,
        contours,
        extrema,
        particles,
        diff_mode,
        is_3d,
        basemap,
        compare_var,
        overlay_counts,
    )


def test_make_sig_shape_pinned():
    """Guard against silent drift with app.py::_structural_sig.

    If a new field is added there but not here, dispatch tests still
    pass while production sends a longer tuple — which the reactive
    calc and decide_dispatch would equate by element count, hiding
    a real change. Keep both shapes pinned together.
    """
    sig = make_sig()
    assert len(sig) == 12
    assert isinstance(sig[-1], tuple) and len(sig[-1]) == 4  # overlay_counts


def test_dispatch_returns_full_on_first_call():
    decision = decide_dispatch(prev_sig=None, curr_sig=make_sig())
    assert decision == "full"


def test_dispatch_returns_partial_when_sig_matches():
    sig = make_sig()
    decision = decide_dispatch(prev_sig=sig, curr_sig=sig)
    assert decision == "partial"


def test_dispatch_returns_full_when_file_changes():
    prev = make_sig(file_path="/tmp/a.slf")
    curr = make_sig(file_path="/tmp/b.slf")
    assert decide_dispatch(prev_sig=prev, curr_sig=curr) == "full"


def test_dispatch_returns_full_when_wireframe_toggles():
    prev = make_sig(wireframe=False)
    curr = make_sig(wireframe=True)
    assert decide_dispatch(prev_sig=prev, curr_sig=curr) == "full"


def test_dispatch_returns_full_when_3d_mode_flips():
    prev = make_sig(is_3d=False)
    curr = make_sig(is_3d=True)
    assert decide_dispatch(prev_sig=prev, curr_sig=curr) == "full"


def test_dispatch_returns_full_when_overlay_count_changes():
    prev = make_sig(overlay_counts=(0, 0, 0, 0))
    curr = make_sig(overlay_counts=(1, 0, 0, 0))  # one cross-section added
    assert decide_dispatch(prev_sig=prev, curr_sig=curr) == "full"


class TestWireFormatContract:
    """Confirm patch dicts flow through MapWidget.partial_update correctly."""

    @pytest.mark.asyncio
    async def test_mesh_color_patch_sends_deck_partial_update(self, fake_geom, fake_tf):
        session = FakeSession()
        widget = MapWidget("map")
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        patch, _, _, _ = build_mesh_color_patch(fake_geom, values, "Viridis")

        await widget.partial_update(session, [patch])

        assert len(session.messages) == 1
        msg_type, payload = session.messages[0]
        assert msg_type == "deck_partial_update"
        assert payload["id"] == "map"
        assert len(payload["layers"]) == 1
        layer = payload["layers"][0]
        assert layer["id"] == "mesh"
        assert "_meshColors" in layer
        # Geometry MUST NOT leak into the patch — the JS-side cache
        # preserves it. Keep this list aligned with the one guarded in
        # tests/test_layers.py::test_patch_contains_mesh_colors_only.
        assert "_meshPositions" not in layer
        assert "_meshIndices" not in layer
        assert "mesh" not in layer

    @pytest.mark.asyncio
    async def test_set_widgets_sends_deck_set_widgets(self):
        session = FakeSession()
        widget = MapWidget("map")
        await widget.set_widgets(session, [zoom_widget()])

        assert len(session.messages) == 1
        msg_type, payload = session.messages[0]
        assert msg_type == "deck_set_widgets"
        assert payload["id"] == "map"
        assert len(payload["widgets"]) == 1

    @pytest.mark.asyncio
    async def test_velocity_patch_sends_binary_attributes(self, fake_tf, fake_geom):
        """build_velocity_patch must flow binary attrs through partial_update."""
        import pytest as _pytest
        from layers import build_velocity_patch

        session = FakeSession()
        widget = MapWidget("map")
        patch = build_velocity_patch(fake_tf, 0, fake_geom)
        if patch is None:
            _pytest.skip("FakeTF has no velocity vars")

        await widget.partial_update(session, [patch])

        assert len(session.messages) == 1
        msg_type, payload = session.messages[0]
        assert msg_type == "deck_partial_update"
        layer = payload["layers"][0]
        assert layer["id"] == "velocity"
        # Binary-attribute mode contract
        assert isinstance(layer["data"], dict) and "length" in layer["data"]
        assert layer["getSourcePosition"].get("@@binary") is True
        assert layer["getTargetPosition"].get("@@binary") is True
        # Rule: no pre-binary shape sneaking back in
        assert not isinstance(layer["data"], list), (
            "data must be {length: N}, not a list of dicts"
        )

    @pytest.mark.asyncio
    async def test_contour_patch_sends_binary_attributes(self, fake_tf, fake_geom):
        """build_contour_patch must flow binary attrs through partial_update."""
        import pytest as _pytest
        from layers import build_contour_patch

        session = FakeSession()
        widget = MapWidget("map")
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        patch = build_contour_patch(fake_tf, values, fake_geom)
        if patch is None:
            _pytest.skip("flat field — no contours generated")

        await widget.partial_update(session, [patch])

        assert len(session.messages) == 1
        msg_type, payload = session.messages[0]
        assert msg_type == "deck_partial_update"
        layer = payload["layers"][0]
        assert layer["id"] == "contours"
        assert isinstance(layer["data"], dict) and "length" in layer["data"]
        assert layer["getSourcePosition"].get("@@binary") is True
        assert layer["getTargetPosition"].get("@@binary") is True
