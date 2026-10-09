"""
synthesizer.py — Synthetic ABP longitudinal time-series generator.

Generates realistic Hemoglobin (HGB) / Reticulocyte Percentage (RET)
time-series for clean athletes using hierarchical autoregressive physiology,
and injects validated two-phase EPO doping patterns (stimulation & washout).

References:
    - WADA (2019). Athlete Biological Passport Operating Guidelines.
    - Sottas, P.E. et al. (2008). Biostatistics, 9(2), 285-296.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from abp_synth.baseline import BaselineResult, extract_baseline, load_abps_data

# ---------------------------------------------------------------------------
# Low-level generators
# ---------------------------------------------------------------------------


def generate_normal_athlete(
    n_tests: int,
    mu_hgb: float,
    mu_ret: float,
    cov_2d: np.ndarray,
    *,
    sigma_step: float = 0.3,
    mean_reversion: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single clean athlete's HGB/RET longitudinal sequence."""
    if rng is None:
        rng = np.random.default_rng()

    mu = np.array([mu_hgb, mu_ret])
    start = rng.multivariate_normal(mu, cov_2d)
    start[0] = np.clip(start[0], 11.5, 18.5)
    start[1] = np.clip(start[1], 0.35, 2.5)

    sequence = [start]
    noise_cov = sigma_step**2 * cov_2d

    for _ in range(n_tests - 1):
        prev = sequence[-1]
        noise = rng.multivariate_normal([0.0, 0.0], noise_cov)
        revert = mean_reversion * (mu - prev)
        new_val = prev + noise + revert
        new_val[0] = np.clip(new_val[0], 11.5, 18.5)
        new_val[1] = np.clip(new_val[1], 0.35, 2.5)
        sequence.append(new_val)

    return np.array(sequence)


def inject_epo_pattern(
    sequence: np.ndarray,
    inject_at: int,
    *,
    hgb_rise: float = 2.2,
    ret_peak: float = 1.4,
    ret_supp: float | None = None,
    ret_drop: float | None = None,
    stim_steps: int = 3,
    washout_steps: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a physiologically sound two-phase EPO doping signature into a sequence.

    Phase A - Stimulation period (stim_steps):
        Bone marrow responds with burst in immature reticulocytes (RET rises),
        followed by progressive step-up in HGB.

    Phase B - Withdrawal period (washout_steps):
        Negative feedback causes RET to plunge to valley levels (0.15~0.4%),
        while HGB decays very slowly (120-day erythrocyte lifespan).
        This creates the classical scissors pattern (high HGB + low RET).
    """
    if ret_supp is None:
        ret_supp = ret_drop if ret_drop is not None else 0.75

    seq = sequence.copy().astype(float)
    n = len(seq)
    point_labels = np.zeros(n, dtype=int)

    # Phase A: stimulation
    for i in range(stim_steps):
        t = inject_at + i
        if t >= n:
            break
        progress = (i + 1) / stim_steps
        seq[t, 1] += ret_peak * np.sin(np.pi * 0.5 * progress)
        seq[t, 1] = np.clip(seq[t, 1], 0.3, 3.8)

        seq[t, 0] += hgb_rise * progress
        seq[t, 0] = np.clip(seq[t, 0], 11.5, 20.5)
        point_labels[t] = 1

    # Phase B: withdrawal (washout)
    washout_start = inject_at + stim_steps
    for j in range(washout_steps):
        t = washout_start + j
        if t >= n:
            break
        # RET deep suppression
        seq[t, 1] -= ret_supp * np.exp(-0.2 * j)
        seq[t, 1] = np.clip(seq[t, 1], 0.12, 0.45)

        # Slow HGB decay
        decay_factor = np.exp(-0.06 * (j + 1))
        seq[t, 0] = seq[inject_at + stim_steps - 1, 0] * decay_factor + sequence[t, 0] * (1 - decay_factor)
        seq[t, 0] = np.clip(seq[t, 0], 11.5, 20.5)
        point_labels[t] = 1

    return seq, point_labels


# ---------------------------------------------------------------------------
# High-level dataset container
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    sequences: list[np.ndarray]
    labels_athlete: np.ndarray
    labels_seq: list[np.ndarray]
    inject_points: np.ndarray

    def save(self, output_dir: Path | str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "synthetic_sequences.npy", np.array(self.sequences, dtype=object), allow_pickle=True)
        np.save(out / "synthetic_labels_seq.npy", np.array(self.labels_seq, dtype=object), allow_pickle=True)
        np.save(out / "synthetic_labels_athlete.npy", self.labels_athlete)
        np.save(out / "synthetic_inject_points.npy", self.inject_points)

    @classmethod
    def load(cls, input_dir: Path | str) -> SyntheticDataset:
        d = Path(input_dir)
        return cls(
            sequences=list(np.load(d / "synthetic_sequences.npy", allow_pickle=True)),
            labels_athlete=np.load(d / "synthetic_labels_athlete.npy"),
            labels_seq=list(np.load(d / "synthetic_labels_seq.npy", allow_pickle=True)),
            inject_points=np.load(d / "synthetic_inject_points.npy"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict] = []
        for i, (seq, lbl_seq) in enumerate(zip(self.sequences, self.labels_seq)):
            for t in range(len(seq)):
                rows.append({
                    "athlete_id": i,
                    "time_step": t,
                    "HGB": seq[t, 0],
                    "RET": seq[t, 1],
                    "OFF": 10.0 * seq[t, 0] - 60.0 * np.sqrt(seq[t, 1]),
                    "label_athlete": int(self.labels_athlete[i]),
                    "label_point": int(lbl_seq[t]),
                })
        return pd.DataFrame(rows)

    @property
    def n_athletes(self) -> int:
        return len(self.sequences)

    @property
    def n_doping(self) -> int:
        return int(self.labels_athlete.sum())

    @property
    def n_clean(self) -> int:
        return self.n_athletes - self.n_doping

    def summary(self) -> str:
        lengths = [len(s) for s in self.sequences]
        return (
            f"SyntheticDataset: {self.n_athletes} athletes "
            f"({self.n_clean} clean, {self.n_doping} doping)\n"
            f"  Sequence length: {min(lengths)}-{max(lengths)} "
            f"(mean {np.mean(lengths):.1f})\n"
            f"  Doping ratio: {self.n_doping / self.n_athletes * 100:.1f}%"
        )


def generate_dataset(
    n_normal: int = 10_000,
    n_doping: int = 1_000,
    min_tests: int = 8,
    max_tests: int = 16,
    baseline: BaselineResult | None = None,
    seed: int = 2024,
) -> SyntheticDataset:
    if n_doping > n_normal:
        raise ValueError(f"n_doping ({n_doping}) must be <= n_normal ({n_normal})")

    rng = np.random.default_rng(seed)

    if baseline is None:
        df = load_abps_data()
        baseline = extract_baseline(df)

    mu_hgb = baseline.mu_hgb
    mu_ret = baseline.mu_ret
    cov_2d = baseline.hgb_ret_cov

    all_sequences: list[np.ndarray] = []
    all_labels_seq: list[np.ndarray] = []
    all_labels_athlete: list[int] = []
    all_inject_at: list[int] = []

    for _ in range(n_normal):
        n_tests = int(rng.integers(min_tests, max_tests + 1))
        seq = generate_normal_athlete(n_tests, mu_hgb, mu_ret, cov_2d, rng=rng)

        # 困难负样本扰动 (脱水/高原反应)
        if rng.random() < 0.10:
            pt = int(rng.integers(1, n_tests - 1))
            seq[pt, 0] += float(rng.uniform(0.6, 1.1))
            seq[pt, 1] += float(rng.uniform(0.1, 0.25))

        all_sequences.append(seq)
        all_labels_seq.append(np.zeros(n_tests, dtype=int))
        all_labels_athlete.append(0)
        all_inject_at.append(-1)

    doping_indices = rng.choice(n_normal, n_doping, replace=False)

    for idx in doping_indices:
        seq = all_sequences[idx]
        n = len(seq)
        inject_at = int(rng.integers(2, max(3, n - 6)))
        hgb_rise = float(rng.uniform(1.8, 2.7))
        ret_peak = float(rng.uniform(1.2, 1.7))
        ret_supp = float(rng.uniform(0.6, 0.85))

        new_seq, point_labels = inject_epo_pattern(
            seq, inject_at,
            hgb_rise=hgb_rise,
            ret_peak=ret_peak,
            ret_supp=ret_supp,
            stim_steps=int(rng.integers(3, 5)),
            washout_steps=int(rng.integers(4, 6))
        )
        all_sequences[idx] = new_seq
        all_labels_seq[idx] = point_labels
        all_labels_athlete[idx] = 1
        all_inject_at[idx] = inject_at

    return SyntheticDataset(
        sequences=all_sequences,
        labels_athlete=np.array(all_labels_athlete, dtype=int),
        labels_seq=all_labels_seq,
        inject_points=np.array(all_inject_at, dtype=int),
    )
