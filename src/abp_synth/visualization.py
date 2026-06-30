"""
visualization.py — Publication-quality plots for ABP baselines and synthetic data.

All functions return a :class:`~matplotlib.figure.Figure` so they can be
displayed inline in Jupyter or saved to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from abp_synth.baseline import BaselineResult
    from abp_synth.synthesizer import SyntheticDataset

# Use non-interactive backend when saving; callers can override.
matplotlib.use("Agg")

_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]


def _maybe_save(fig: Figure, path: Path | str | None) -> None:
    if path is not None:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Baseline visualizations
# ---------------------------------------------------------------------------


def plot_distributions(
    baseline: BaselineResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot histograms with normal-fit overlays and Shapiro-Wilk tests.

    Parameters:
        baseline: A :class:`BaselineResult` from :func:`extract_baseline`.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    features = baseline.feature_names
    df = baseline.df_clean
    n_feat = len(features)

    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 4))
    if n_feat == 1:
        axes = [axes]

    fig.suptitle(
        "ABPS Dataset — Core Feature Distributions",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for i, (col, color) in enumerate(zip(features, _PALETTE)):
        ax = axes[i]
        data = df[col].dropna()

        ax.hist(data, bins=35, color=color, alpha=0.55, density=True, label="Actual")

        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color="black",
            lw=2,
            ls="--",
            label="Normal fit",
        )

        sample_size = min(4999, len(data))
        _, sw_p = stats.shapiro(data.sample(sample_size, random_state=42))

        ax.set_title(
            f"{col}\nμ={mu:.3f}, σ={sigma:.3f}\nShapiro p={sw_p:.3f}",
            fontsize=10,
        )
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_correlation(
    baseline: BaselineResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot a heatmap of the feature correlation matrix.

    Parameters:
        baseline: A :class:`BaselineResult`.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    import pandas as pd

    n = len(baseline.feature_names)
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))

    corr_df = pd.DataFrame(
        baseline.corr,
        index=baseline.feature_names,
        columns=baseline.feature_names,
    )
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Core Feature Correlation Matrix", fontsize=12)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_scatter_hgb_ret(
    baseline: BaselineResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Scatter plot of HGB vs RET with population mean cross-hairs.

    Parameters:
        baseline: A :class:`BaselineResult` containing HGB and RET columns.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    df = baseline.df_clean
    mu_hgb = baseline.mu_hgb
    mu_ret = baseline.mu_ret

    r_val = baseline.corr[
        baseline.feature_names.index("HGB"),
        baseline.feature_names.index("RET"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["HGB"], df["RET"], alpha=0.4, s=20, c="#3498db", edgecolors="none")
    ax.axvline(mu_hgb, color="red", lw=1.5, ls="--", label=f"HGB mean={mu_hgb:.2f}")
    ax.axhline(mu_ret, color="orange", lw=1.5, ls="--", label=f"RET mean={mu_ret:.2f}")

    ax.set_xlabel("HGB — Hemoglobin (g/dL)", fontsize=11)
    ax.set_ylabel("RET — Reticulocyte (%)", fontsize=11)
    ax.set_title(f"HGB vs RET (Normal Athletes Baseline)\nr = {r_val:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Synthetic dataset visualizations
# ---------------------------------------------------------------------------


def plot_comparison(
    dataset: SyntheticDataset,
    baseline: BaselineResult,
    n_samples: int = 4,
    save_path: Path | str | None = None,
    seed: int = 99,
) -> Figure:
    """Plot side-by-side normal vs. doping athlete time-series.

    Parameters:
        dataset: A :class:`SyntheticDataset`.
        baseline: A :class:`BaselineResult` (used for mean reference lines).
        n_samples: How many athletes to show per row (default 4).
        save_path: Optional file path to save the figure.
        seed: Random seed for sample selection.

    Returns:
        The matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    mu_hgb = baseline.mu_hgb
    mu_ret = baseline.mu_ret

    normal_pool = np.where(dataset.labels_athlete == 0)[0]
    doping_pool = np.where(dataset.labels_athlete == 1)[0]

    sample_normal = rng.choice(normal_pool, min(n_samples, len(normal_pool)), replace=False)
    sample_doping = rng.choice(doping_pool, min(n_samples, len(doping_pool)), replace=False)

    cols = max(len(sample_normal), len(sample_doping))
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 8))
    fig.suptitle(
        "Synthetic Time-Series Comparison\nNormal (top) vs EPO Doping (bottom)",
        fontsize=14,
        fontweight="bold",
    )

    def _plot_one(ax, idx: int, title_prefix: str) -> None:
        seq = dataset.sequences[idx]
        labels = dataset.labels_seq[idx]
        n = len(seq)
        steps = range(n)

        ax.plot(steps, seq[:, 0], "b-o", ms=5, lw=1.5, label="HGB (g/dL)")
        ax2 = ax.twinx()
        ax2.plot(steps, seq[:, 1], "r--s", ms=4, lw=1.5, label="RET (%)")

        for t in range(n):
            if labels[t] == 1:
                ax.axvspan(t - 0.5, t + 0.5, alpha=0.15, color="red")

        ax.axhline(mu_hgb, color="blue", lw=0.8, ls=":", alpha=0.5)
        ax2.axhline(mu_ret, color="red", lw=0.8, ls=":", alpha=0.5)

        inject = dataset.inject_points[idx]
        tag = f"[EPO@t={inject}]" if dataset.labels_athlete[idx] == 1 else "[CLEAN]"
        ax.set_title(f"{title_prefix} #{idx}\n{tag}", fontsize=9)
        ax.set_xlabel("Test #", fontsize=8)
        ax.set_ylabel("HGB", color="blue", fontsize=8)
        ax2.set_ylabel("RET", color="red", fontsize=8)
        ax.tick_params(axis="y", labelcolor="blue", labelsize=7)
        ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
        ax.set_xlim(-0.5, n - 0.5)
        ax.grid(alpha=0.3)

    for col, idx in enumerate(sample_normal):
        _plot_one(axes[0, col], idx, "Normal")
    for col, idx in enumerate(sample_doping):
        _plot_one(axes[1, col], idx, "Doping")

    # Hide unused axes
    for row in range(2):
        n_used = len(sample_normal) if row == 0 else len(sample_doping)
        for col in range(n_used, cols):
            axes[row, col].set_visible(False)

    red_patch = mpatches.Patch(color="red", alpha=0.3, label="Anomaly window")
    fig.legend(handles=[red_patch], loc="lower right", fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
