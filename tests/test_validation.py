# tests/test_validation.py — Tests for validation.py
import logging
import numpy as np
import pytest

from validation import parse_observation_csv, compute_rmse, compute_nse, compute_volume_timeseries, parse_liq_file


class TestParseObservationCsv:
    def test_basic(self, tmp_path):
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text("time,WATER DEPTH\n0,1.0\n10,2.0\n20,3.0\n")
        times, values, varname = parse_observation_csv(str(csv_file))
        np.testing.assert_array_equal(times, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
        assert varname == "WATER DEPTH"

    def test_varname_from_header(self, tmp_path):
        csv_file = tmp_path / "obs2.csv"
        csv_file.write_text("t,FREE SURFACE\n5,0.5\n")
        _, _, varname = parse_observation_csv(str(csv_file))
        assert varname == "FREE SURFACE"

    def test_empty_data_raises(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("time,value\n")
        with pytest.raises(ValueError, match="no data rows"):
            parse_observation_csv(str(csv_file))

    def test_single_column_raises(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("time\n1\n2\n")
        with pytest.raises(ValueError, match="at least 2 columns"):
            parse_observation_csv(str(csv_file))


class TestComputeRmse:
    def test_perfect_match(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(a, a) == 0.0

    def test_known_value(self):
        model = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 5.0])
        # RMSE = sqrt(mean([0, 0, 4])) = sqrt(4/3)
        expected = np.sqrt(4.0 / 3.0)
        assert abs(compute_rmse(model, obs) - expected) < 1e-12

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shapes must match"):
            compute_rmse(np.array([1, 2]), np.array([1, 2, 3]))


class TestComputeNse:
    def test_perfect_match(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_nse(a, a) == 1.0

    def test_mean_model(self):
        obs = np.array([1.0, 2.0, 3.0])
        model = np.full(3, obs.mean())
        # NSE should be 0 when model = mean(obs)
        assert abs(compute_nse(model, obs)) < 1e-12

    def test_bad_model(self):
        obs = np.array([1.0, 2.0, 3.0])
        model = np.array([10.0, 20.0, 30.0])
        assert compute_nse(model, obs) < 0.0

    def test_constant_obs(self):
        obs = np.array([5.0, 5.0, 5.0])
        model = np.array([5.0, 5.0, 5.0])
        assert compute_nse(model, obs) == 1.0

    def test_constant_obs_mismatch(self):
        obs = np.array([5.0, 5.0, 5.0])
        model = np.array([1.0, 2.0, 3.0])
        assert compute_nse(model, obs) == float("-inf")

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shapes must match"):
            compute_nse(np.array([1, 2]), np.array([1, 2, 3]))


def test_compute_volume_timeseries():
    from tests.helpers import FakeTF
    from analysis import compute_mesh_integral
    tf = FakeTF()
    times, vols = compute_volume_timeseries(tf, compute_mesh_integral)
    assert len(times) == 3
    assert len(vols) == 3
    assert all(v > 0 for v in vols)


def test_parse_liq_file(tmp_path):
    liq = tmp_path / "test.liq"
    liq.write_text("# comment\nT Q(1) SL(2)\ns m3/s m\n0.0 500.0 1.5\n3600.0 450.0 1.4\n7200.0 400.0 1.3\n")
    result = parse_liq_file(str(liq))
    assert result is not None
    assert "Q(1)" in result
    assert "SL(2)" in result
    assert len(result["Q(1)"]["times"]) == 3
    assert result["Q(1)"]["values"][0] == 500.0
    assert result["Q(1)"]["unit"] == "m3/s"
    assert result["SL(2)"]["unit"] == "m"


class TestParseLiqFileErrors:
    def test_malformed_float_returns_none_and_logs(self, tmp_path, caplog):
        """Malformed data in .liq file should return None and log a warning."""
        liq = tmp_path / "bad.liq"
        liq.write_text("T  Q(1)\ns  m3/s\n0.0  abc\n")
        with caplog.at_level(logging.WARNING):
            result = parse_liq_file(str(liq))
        assert result is None
        assert any("parse" in r.message.lower() or "liq" in r.message.lower()
                    for r in caplog.records)

    def test_missing_file_returns_none(self, tmp_path):
        """Missing file should return None without raising."""
        result = parse_liq_file(str(tmp_path / "nonexistent.liq"))
        assert result is None

    def test_too_few_lines_returns_none(self, tmp_path):
        """File with fewer than 3 lines should return None."""
        liq = tmp_path / "short.liq"
        liq.write_text("T  Q(1)\ns  m3/s\n")
        result = parse_liq_file(str(liq))
        assert result is None
