"""Tests for app.py reactives that depend on Shiny's reactive system."""


def test_merged_entries_scans_once_per_invalidation():
    """Structural test: verify scan_library is called from exactly one
    reactive.calc body (merged_entries). Consumers subscribe to its
    cached result rather than calling scan_library directly.

    Catches the regression where library_choices, _pick_file_path, and
    find_companion call sites each duplicate the scan_library invocation.
    """
    import inspect
    import app

    # The server() factory contains the merged_entries reactive.calc.
    src = inspect.getsource(app.server)
    # Exactly one scan_library call should appear inside server() — inside
    # merged_entries. find_companion call sites pass merged_entries() (NOT
    # scan_library() directly).
    scan_calls = src.count("scan_library(")
    assert scan_calls == 1, (
        f"Expected exactly 1 scan_library() call inside server() (the "
        f"merged_entries reactive), found {scan_calls}. Did a consumer "
        f"re-introduce a direct scan?"
    )
    assert "user_id=" in src, "merged_entries must pass user_id to scan_library"


def test_build_project_files_strips_path_components(tmp_path):
    """Defense-in-depth: _validate_companion_basename strips path
    components from client-supplied FileInfo['name']."""
    from app import _build_project_files

    slf_path = tmp_path / "data1"
    cas_path = tmp_path / "data2"
    slf_path.touch()
    cas_path.touch()
    uploaded = [
        {"name": "../../etc/case.slf", "datapath": str(slf_path)},
        {"name": "case.cas", "datapath": str(cas_path)},
    ]
    pf = _build_project_files(uploaded)
    assert pf.slf == slf_path
    assert pf.cas == cas_path


def test_build_project_files_rejects_duplicate_suffix(tmp_path):
    from app import _build_project_files

    p1 = tmp_path / "a"
    p2 = tmp_path / "b"
    p1.touch()
    p2.touch()
    uploaded = [
        {"name": "case.slf", "datapath": str(p1)},
        {"name": "other.slf", "datapath": str(p2)},
    ]
    import pytest

    with pytest.raises(ValueError, match="Multiple uploads"):
        _build_project_files(uploaded)


def test_build_project_files_requires_slf(tmp_path):
    from app import _build_project_files

    p = tmp_path / "data"
    p.touch()
    uploaded = [{"name": "case.cas", "datapath": str(p)}]
    import pytest

    with pytest.raises(ValueError, match="No .slf"):
        _build_project_files(uploaded)


def test_build_project_files_hostile_name_with_nul_byte(tmp_path):
    from app import _build_project_files

    p = tmp_path / "data"
    p.touch()
    uploaded = [{"name": "case\x00.slf", "datapath": str(p)}]
    import pytest

    with pytest.raises(ValueError):
        _build_project_files(uploaded)


def test_build_project_files_round_trip_with_all_companions(tmp_path):
    from app import _build_project_files

    paths = {
        ".slf": tmp_path / "s",
        ".cas": tmp_path / "c",
        ".cli": tmp_path / "i",
        ".liq": tmp_path / "l",
    }
    for p in paths.values():
        p.touch()
    uploaded = [{"name": f"case{ext}", "datapath": str(p)} for ext, p in paths.items()]
    pf = _build_project_files(uploaded)
    assert pf.slf == paths[".slf"]
    assert pf.cas == paths[".cas"]
    assert pf.cli == paths[".cli"]
    assert pf.liq == paths[".liq"]
