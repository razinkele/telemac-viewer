"""Round 5 regression tests — 5 bug fixes."""
from __future__ import annotations

import inspect
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BUG 1: Import convert missing hdf_files guard
# ---------------------------------------------------------------------------
class TestImportConvertGuard:
    def test_convert_has_hdf_check(self):
        """handle_import_convert must validate hdf_files before indexing."""
        from server_import import register_import_handlers
        src = inspect.getsource(register_import_handlers)
        # Find the convert section and verify guard exists
        convert_idx = src.find("Converting")
        assert convert_idx > 0
        after_convert = src[convert_idx:]
        guard_idx = after_convert.find("not hdf_files")
        index_idx = after_convert.find('hdf_files[0]')
        assert guard_idx > 0, "Missing hdf_files guard in convert handler"
        assert guard_idx < index_idx, "Guard must come before hdf_files[0] access"


# ---------------------------------------------------------------------------
# BUG 2: Particle seed-line exception handler
# ---------------------------------------------------------------------------
class TestParticleSeedLineException:
    def test_seed_line_catches_general_exception(self):
        """Seed-line particle path must catch Exception, not just TimeoutError."""
        from server_analysis import register_analysis_handlers
        src = inspect.getsource(register_analysis_handlers)
        # Find seed-line particle section (draw_seed / polyline / seed-line)
        timeout_idx = src.find("Particle tracing timed out")
        assert timeout_idx > 0
        # After the TimeoutError handler, there should be a broader Exception catch
        after_timeout = src[timeout_idx:timeout_idx + 500]
        assert "except Exception" in after_timeout, \
            "Missing broad Exception handler after TimeoutError in seed-line path"


# ---------------------------------------------------------------------------
# BUG 3: BC node assignment silent empty warning
# ---------------------------------------------------------------------------
class TestBCEmptyMatchWarning:
    def test_assign_bc_nodes_warns_on_empty(self):
        """assign_bc_nodes should warn when no boundary nodes match a BC."""
        src = inspect.getsource(
            __import__("telemac_tools.domain.builder",
                       fromlist=["assign_bc_nodes"]).assign_bc_nodes)
        assert "warning" in src.lower() and "matched zero" in src.lower(), \
            "Missing warning for empty BC node match"


# ---------------------------------------------------------------------------
# BUG 4: inf/NaN filtering in parse_observation_csv
# ---------------------------------------------------------------------------
class TestObservationCsvInfNaN:
    def test_inf_rows_skipped(self, tmp_path):
        from validation import parse_observation_csv
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("time,value\n0.0,1.0\n1.0,inf\n2.0,3.0\n")
        times, values, varname = parse_observation_csv(str(csv_file))
        assert len(times) == 2
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(times, [0.0, 2.0])
        np.testing.assert_array_equal(values, [1.0, 3.0])

    def test_nan_rows_skipped(self, tmp_path):
        from validation import parse_observation_csv
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("time,value\n0.0,1.0\nnan,2.0\n2.0,nan\n3.0,4.0\n")
        times, values, varname = parse_observation_csv(str(csv_file))
        assert len(times) == 2
        assert np.all(np.isfinite(values))

    def test_all_inf_raises(self, tmp_path):
        from validation import parse_observation_csv
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("time,value\ninf,inf\n-inf,nan\n")
        with pytest.raises(ValueError, match="no data rows"):
            parse_observation_csv(str(csv_file))


# ---------------------------------------------------------------------------
# BUG 5: inf/NaN filtering in parse_liq_file
# ---------------------------------------------------------------------------
class TestLiqFileInfNaN:
    def test_inf_rows_skipped(self, tmp_path):
        from validation import parse_liq_file
        liq_file = tmp_path / "test.liq"
        liq_file.write_text(
            "#\n"
            "T Q(1)\n"
            "s m3/s\n"
            "0.0 10.0\n"
            "1.0 inf\n"
            "2.0 30.0\n"
        )
        result = parse_liq_file(str(liq_file))
        assert result is not None
        times = list(result.values())[0]["times"]
        assert len(times) == 2
        assert np.all(np.isfinite(times))

    def test_nan_rows_skipped(self, tmp_path):
        from validation import parse_liq_file
        liq_file = tmp_path / "test.liq"
        liq_file.write_text(
            "#\n"
            "T Q(1)\n"
            "s m3/s\n"
            "0.0 10.0\n"
            "nan 20.0\n"
            "2.0 30.0\n"
        )
        result = parse_liq_file(str(liq_file))
        assert result is not None
        vals = list(result.values())[0]["values"]
        assert np.all(np.isfinite(vals))
