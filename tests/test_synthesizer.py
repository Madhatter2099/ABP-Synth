"""Tests for abp_synth.synthesizer module."""

import numpy as np
import pytest

from abp_synth.baseline import extract_baseline, load_abps_data
from abp_synth.synthesizer import (
    SyntheticDataset,
    generate_dataset,
    generate_normal_athlete,
    inject_epo_pattern,
)


@pytest.fixture()
def baseline():
    df = load_abps_data("bundled")
    return extract_baseline(df)


class TestGenerateNormalAthlete:
    """Tests for generate_normal_athlete()."""

    def test_output_shape(self, baseline):
        seq = generate_normal_athlete(
            10, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov
        )
        assert seq.shape == (10, 2)

    def test_hgb_in_range(self, baseline):
        seq = generate_normal_athlete(
            100, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov
        )
        assert seq[:, 0].min() >= 10.0
        assert seq[:, 0].max() <= 20.0

    def test_ret_in_range(self, baseline):
        seq = generate_normal_athlete(
            100, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov
        )
        assert seq[:, 1].min() >= 0.2
        assert seq[:, 1].max() <= 3.0

    def test_deterministic_with_rng(self, baseline):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = generate_normal_athlete(
            10, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov, rng=rng1
        )
        s2 = generate_normal_athlete(
            10, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov, rng=rng2
        )
        np.testing.assert_array_equal(s1, s2)


class TestInjectEpoPattern:
    """Tests for inject_epo_pattern()."""

    @pytest.fixture()
    def clean_seq(self, baseline):
        rng = np.random.default_rng(0)
        return generate_normal_athlete(
            12, baseline.mu_hgb, baseline.mu_ret, baseline.hgb_ret_cov, rng=rng
        )

    def test_hgb_increases_at_injection(self, clean_seq):
        original = clean_seq.copy()
        modified, labels = inject_epo_pattern(clean_seq, inject_at=3, hgb_rise=2.0)
        # HGB at inject points should be higher
        for t in range(3, 6):
            assert modified[t, 0] >= original[t, 0]

    def test_ret_decreases_after_injection(self, clean_seq):
        original = clean_seq.copy()
        modified, labels = inject_epo_pattern(clean_seq, inject_at=2, ret_drop=0.4)
        # RET at drop points (after rise_steps=3) should be lower
        drop_start = 2 + 3  # inject_at + rise_steps
        for t in range(drop_start, min(drop_start + 4, len(modified))):
            assert modified[t, 1] <= original[t, 1]

    def test_labels_mark_modified_points(self, clean_seq):
        _, labels = inject_epo_pattern(clean_seq, inject_at=2)
        assert labels.sum() > 0
        assert labels[0] == 0  # Before injection
        assert labels[1] == 0

    def test_original_not_mutated(self, clean_seq):
        original_copy = clean_seq.copy()
        inject_epo_pattern(clean_seq, inject_at=3)
        np.testing.assert_array_equal(clean_seq, original_copy)


class TestGenerateDataset:
    """Tests for generate_dataset()."""

    @pytest.fixture()
    def dataset(self, baseline):
        return generate_dataset(
            n_normal=100, n_doping=10, min_tests=8, max_tests=12,
            baseline=baseline, seed=42,
        )

    def test_total_athletes(self, dataset):
        assert dataset.n_athletes == 100

    def test_doping_count(self, dataset):
        assert dataset.n_doping == 10

    def test_clean_count(self, dataset):
        assert dataset.n_clean == 90

    def test_label_array_shape(self, dataset):
        assert dataset.labels_athlete.shape == (100,)

    def test_sequence_lengths_in_range(self, dataset):
        lengths = [len(s) for s in dataset.sequences]
        assert min(lengths) >= 8
        assert max(lengths) <= 12

    def test_inject_points_minus_one_for_clean(self, dataset):
        clean_mask = dataset.labels_athlete == 0
        assert np.all(dataset.inject_points[clean_mask] == -1)

    def test_inject_points_positive_for_doping(self, dataset):
        doping_mask = dataset.labels_athlete == 1
        assert np.all(dataset.inject_points[doping_mask] >= 0)

    def test_n_doping_exceeds_n_normal_raises(self, baseline):
        with pytest.raises(ValueError, match="must be"):
            generate_dataset(n_normal=10, n_doping=20, baseline=baseline)

    def test_deterministic_with_seed(self, baseline):
        d1 = generate_dataset(n_normal=50, n_doping=5, baseline=baseline, seed=999)
        d2 = generate_dataset(n_normal=50, n_doping=5, baseline=baseline, seed=999)
        np.testing.assert_array_equal(d1.labels_athlete, d2.labels_athlete)
        for s1, s2 in zip(d1.sequences, d2.sequences):
            np.testing.assert_array_equal(s1, s2)


class TestSyntheticDatasetIO:
    """Tests for SyntheticDataset.save() / .load()."""

    def test_roundtrip(self, baseline, tmp_path):
        ds = generate_dataset(
            n_normal=20, n_doping=2, baseline=baseline, seed=0
        )
        ds.save(tmp_path)
        loaded = SyntheticDataset.load(tmp_path)

        assert loaded.n_athletes == ds.n_athletes
        assert loaded.n_doping == ds.n_doping
        np.testing.assert_array_equal(loaded.labels_athlete, ds.labels_athlete)

    def test_to_dataframe(self, baseline):
        ds = generate_dataset(
            n_normal=10, n_doping=1, baseline=baseline, seed=0
        )
        df = ds.to_dataframe()
        assert "athlete_id" in df.columns
        assert "HGB" in df.columns
        assert "label_athlete" in df.columns
        assert len(df) == sum(len(s) for s in ds.sequences)
