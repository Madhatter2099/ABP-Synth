"""Tests for abp_synth.baseline module."""

import numpy as np
import pandas as pd
import pytest

from abp_synth.baseline import (
    BaselineResult,
    _generate_literature_fallback,
    extract_baseline,
    load_abps_data,
)


class TestLoadAbpsData:
    """Tests for load_abps_data()."""

    def test_bundled_loads_successfully(self):
        df = load_abps_data("bundled")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_bundled_has_required_columns(self):
        df = load_abps_data("bundled")
        for col in ("HGB", "RET", "OFF", "ABPS"):
            assert col in df.columns, f"Missing column: {col}"

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_abps_data("/nonexistent/path.csv")


class TestExtractBaseline:
    """Tests for extract_baseline()."""

    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return load_abps_data("bundled")

    def test_returns_baseline_result(self, df):
        bl = extract_baseline(df)
        assert isinstance(bl, BaselineResult)

    def test_mean_shape(self, df):
        bl = extract_baseline(df)
        assert bl.mean.shape == (4,)

    def test_cov_shape(self, df):
        bl = extract_baseline(df)
        assert bl.cov.shape == (4, 4)

    def test_corr_diagonal_is_one(self, df):
        bl = extract_baseline(df)
        np.testing.assert_allclose(np.diag(bl.corr), 1.0, atol=1e-6)

    def test_hgb_mean_physiologically_reasonable(self, df):
        bl = extract_baseline(df)
        # Male athlete HGB should be roughly 12-18 g/dL
        assert 12.0 < bl.mu_hgb < 18.0

    def test_ret_mean_physiologically_reasonable(self, df):
        bl = extract_baseline(df)
        # RET% should be roughly 0.2-3.0%
        assert 0.2 < bl.mu_ret < 3.0

    def test_hgb_ret_cov_is_2x2(self, df):
        bl = extract_baseline(df)
        assert bl.hgb_ret_cov.shape == (2, 2)

    def test_custom_features(self, df):
        bl = extract_baseline(df, features=["HGB", "RET"])
        assert bl.feature_names == ["HGB", "RET"]
        assert bl.mean.shape == (2,)

    def test_missing_feature_raises(self, df):
        with pytest.raises(KeyError):
            extract_baseline(df, features=["HGB", "NONEXISTENT"])

    def test_save_and_reload(self, df, tmp_path):
        bl = extract_baseline(df)
        bl.save(tmp_path)
        assert (tmp_path / "baseline_mean.npy").exists()
        assert (tmp_path / "baseline_cov.npy").exists()
        assert (tmp_path / "feature_names.npy").exists()

        loaded_mean = np.load(tmp_path / "baseline_mean.npy")
        np.testing.assert_array_equal(loaded_mean, bl.mean)


class TestLiteratureFallback:
    """Tests for _generate_literature_fallback()."""

    def test_returns_dataframe(self):
        df = _generate_literature_fallback(n=100, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_has_required_columns(self):
        df = _generate_literature_fallback(n=50, seed=0)
        for col in ("HGB", "RET", "OFF", "ABPS"):
            assert col in df.columns

    def test_deterministic_with_seed(self):
        df1 = _generate_literature_fallback(n=50, seed=123)
        df2 = _generate_literature_fallback(n=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)
