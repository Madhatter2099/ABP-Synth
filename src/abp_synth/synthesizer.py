"""
synthesizer.py — Synthetic ABP longitudinal time-series generator.

Generates realistic Hemoglobin (HGB) / Reticulocyte Percentage (RET)
time-series for clean athletes using a mean-reverting multivariate random
walk, and optionally injects two-phase EPO doping patterns.

Mathematical basis:
    x_t = x_{t-1} + ε_t + α (μ − x_{t-1})

    where ε_t ~ N(0, σ² Σ),  α = mean reversion strength.
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
    mean_reversion: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single clean athlete's HGB/RET longitudinal sequence.

    Uses a multivariate random walk with mean reversion (AR(1) approx.)
    to simulate physiologically realistic temporal autocorrelation.

    Parameters:
        n_tests: Number of test time-points in the sequence.
        mu_hgb: Population mean of Hemoglobin (g/dL).
        mu_ret: Population mean of Reticulocyte percentage (%).
        cov_2d: 2×2 HGB-RET covariance matrix.
        sigma_step: Noise scaling factor (default 0.3).
        mean_reversion: Strength of pull towards population mean (default 0.05).
        rng: NumPy random generator instance.

    Returns:
        Array of shape ``(n_tests, 2)`` with columns ``[HGB, RET]``.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.array([mu_hgb, mu_ret])
    start = rng.multivariate_normal(mu, cov_2d)
    start[0] = np.clip(start[0], 10.0, 20.0)
    start[1] = np.clip(start[1], 0.2, 3.0)

    sequence = [start]
    noise_cov = sigma_step**2 * cov_2d

    for _ in range(n_tests - 1):
        prev = sequence[-1]
        noise = rng.multivariate_normal([0.0, 0.0], noise_cov)
        revert = mean_reversion * (mu - prev)
        new_val = prev + noise + revert
        new_val[0] = np.clip(new_val[0], 10.0, 20.0)
        new_val[1] = np.clip(new_val[1], 0.2, 3.0)
        sequence.append(new_val)

    return np.array(sequence)


def inject_epo_pattern(
    sequence: np.ndarray,
    inject_at: int,
    *,
    hgb_rise: float = 2.0,
    ret_drop: float = 0.35,
    rise_steps: int = 3,
    drop_steps: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a two-phase EPO doping signature into a sequence.

    **Phase A — Injection period** (``rise_steps`` steps):
        Bone marrow stimulation causes HGB to rise linearly by up to
        *hgb_rise* g/dL.

    **Phase B — Withdrawal period** (``drop_steps`` steps):
        Endogenous negative feedback causes RET to drop in a sinusoidal
        "inhibitory valley" pattern by up to *ret_drop* %.

    Parameters:
        sequence: Clean athlete array of shape ``(T, 2)``.
        inject_at: Time index at which the doping episode begins.
        hgb_rise: Maximum HGB elevation (g/dL).
        ret_drop: Maximum RET suppression (%).
        rise_steps: Duration of the HGB rise phase.
        drop_steps: Duration of the RET drop phase.

    Returns:
        A tuple ``(modified_sequence, point_labels)`` where
        *point_labels* is an integer array (0/1) of the same length
        marking which time-points were modified.
    """
    seq = sequence.copy().astype(float)
    n = len(seq)
    point_labels = np.zeros(n, dtype=int)

    # Phase A: HGB linear ramp-up
    for i in range(rise_steps):
        t = inject_at + i
        if t >= n:
            break
        progress = (i + 1) / rise_steps
        seq[t, 0] += hgb_rise * progress
        seq[t, 0] = np.clip(seq[t, 0], 10.0, 22.0)
        point_labels[t] = 1

    # Phase B: RET sinusoidal suppression
    drop_start = inject_at + rise_steps
    for i in range(drop_steps):
        t = drop_start + i
        if t >= n:
            break
        progress = (i + 1) / drop_steps
        drop_amount = ret_drop * np.sin(np.pi * progress)
        seq[t, 1] -= drop_amount
        seq[t, 1] = np.clip(seq[t, 1], 0.1, 3.0)
        point_labels[t] = 1

    return seq, point_labels


# ---------------------------------------------------------------------------
# High-level dataset container
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    """Container for a generated synthetic ABP dataset.

    Attributes:
        sequences: List of per-athlete arrays, each of shape ``(T_i, 2)``.
        labels_athlete: Binary array of shape ``(N,)``; ``1`` = doping.
        labels_seq: List of per-athlete per-time-point labels.
        inject_points: Array of shape ``(N,)``; the time index where EPO
            was injected (``-1`` for clean athletes).
    """

    sequences: list[np.ndarray]
    labels_athlete: np.ndarray
    labels_seq: list[np.ndarray]
    inject_points: np.ndarray

    # --- persistence -------------------------------------------------------

    def save(self, output_dir: Path | str) -> None:
        """Save the dataset to *output_dir* as ``.npy`` files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(
            out / "synthetic_sequences.npy",
            np.array(self.sequences, dtype=object),
            allow_pickle=True,
        )
        np.save(
            out / "synthetic_labels_seq.npy",
            np.array(self.labels_seq, dtype=object),
            allow_pickle=True,
        )
        np.save(out / "synthetic_labels_athlete.npy", self.labels_athlete)
        np.save(out / "synthetic_inject_points.npy", self.inject_points)

    @classmethod
    def load(cls, input_dir: Path | str) -> SyntheticDataset:
        """Load a previously saved dataset from *input_dir*."""
        d = Path(input_dir)
        return cls(
            sequences=list(
                np.load(d / "synthetic_sequences.npy", allow_pickle=True)
            ),
            labels_athlete=np.load(d / "synthetic_labels_athlete.npy"),
            labels_seq=list(
                np.load(d / "synthetic_labels_seq.npy", allow_pickle=True)
            ),
            inject_points=np.load(d / "synthetic_inject_points.npy"),
        )

    # --- convenience -------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the dataset into a long-format DataFrame.

        Columns: ``athlete_id``, ``time_step``, ``HGB``, ``RET``,
        ``label_athlete``, ``label_point``.
        """
        rows: list[dict] = []
        for i, (seq, lbl_seq) in enumerate(
            zip(self.sequences, self.labels_seq)
        ):
            for t in range(len(seq)):
                rows.append(
                    {
                        "athlete_id": i,
                        "time_step": t,
                        "HGB": seq[t, 0],
                        "RET": seq[t, 1],
                        "label_athlete": int(self.labels_athlete[i]),
                        "label_point": int(lbl_seq[t]),
                    }
                )
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
        """Return a human-readable summary string."""
        lengths = [len(s) for s in self.sequences]
        return (
            f"SyntheticDataset: {self.n_athletes} athletes "
            f"({self.n_clean} clean, {self.n_doping} doping)\n"
            f"  Sequence length: {min(lengths)}-{max(lengths)} "
            f"(mean {np.mean(lengths):.1f})\n"
            f"  Doping ratio: {self.n_doping / self.n_athletes * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# High-level generation API
# ---------------------------------------------------------------------------


def generate_dataset(
    n_normal: int = 10_000,
    n_doping: int = 1_000,
    min_tests: int = 8,
    max_tests: int = 16,
    baseline: BaselineResult | None = None,
    seed: int = 2024,
) -> SyntheticDataset:
    """Generate a complete synthetic ABP dataset in one call.

    Workflow:
        1. Generate *n_normal* clean athlete sequences via mean-reverting
           random walks parameterised from the ABPS baseline.
        2. Randomly select *n_doping* of these and inject realistic
           two-phase EPO patterns (HGB rise + RET drop).

    Parameters:
        n_normal: Number of clean athletes to generate.
        n_doping: Number of athletes to mark as doping (must be ≤ n_normal).
        min_tests: Minimum number of tests per athlete.
        max_tests: Maximum number of tests per athlete.
        baseline: Pre-computed :class:`BaselineResult`.  If ``None``, the
            bundled ABPS data is loaded and baselines are extracted
            automatically.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SyntheticDataset` instance.
    """
    if n_doping > n_normal:
        raise ValueError(
            f"n_doping ({n_doping}) must be ≤ n_normal ({n_normal})"
        )

    rng = np.random.default_rng(seed)

    # Load baseline if not provided
    if baseline is None:
        df = load_abps_data()
        baseline = extract_baseline(df)

    mu_hgb = baseline.mu_hgb
    mu_ret = baseline.mu_ret
    cov_2d = baseline.hgb_ret_cov

    # --- Step 1: generate clean sequences ---------------------------------
    all_sequences: list[np.ndarray] = []
    all_labels_seq: list[np.ndarray] = []
    all_labels_athlete: list[int] = []
    all_inject_at: list[int] = []

    for _ in range(n_normal):
        n_tests = int(rng.integers(min_tests, max_tests + 1))
        seq = generate_normal_athlete(
            n_tests, mu_hgb, mu_ret, cov_2d, rng=rng
        )
        all_sequences.append(seq)
        all_labels_seq.append(np.zeros(n_tests, dtype=int))
        all_labels_athlete.append(0)
        all_inject_at.append(-1)

    # --- Step 2: inject EPO patterns --------------------------------------
    doping_indices = rng.choice(n_normal, n_doping, replace=False)

    for idx in doping_indices:
        seq = all_sequences[idx]
        n = len(seq)
        earliest = 2
        latest = max(earliest + 1, n - 5)
        inject_at = int(rng.integers(earliest, latest + 1))
        hgb_rise = float(rng.uniform(1.5, 3.0))
        ret_drop = float(rng.uniform(0.25, 0.50))

        new_seq, point_labels = inject_epo_pattern(
            seq, inject_at, hgb_rise=hgb_rise, ret_drop=ret_drop
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
