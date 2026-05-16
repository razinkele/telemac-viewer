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
