"""
baseline.py — ABPS baseline data loading and statistical parameter extraction.

This module provides functionality to:
1. Load the ABPS (Anti-doping Blood Profile Score) dataset from bundled CSV,
   CRAN RData (bloodcontrol/blooddoping), a local file, or remote sources.
2. Extract multivariate statistical baselines (mean vector, covariance matrix,
   correlation matrix) adhering to WADA ABP physiological guidelines.

References:
    - Sottas, P.E. et al. (2008). Biostatistics, 9(2), 285-296.
    - Sharpe, K. et al. (2006). Haematologica, 91(12), 1603-1610.
    - WADA (2019). Athlete Biological Passport Operating Guidelines.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Container for extracted baseline statistics.

    Attributes:
        mean: Mean vector of selected features. Shape ``(n_features,)``.
        cov: Covariance matrix. Shape ``(n_features, n_features)``.
        corr: Pearson correlation matrix. Shape ``(n_features, n_features)``.
        feature_names: Ordered list of feature column names.
        df_clean: Cleaned DataFrame used for computation (NaN rows dropped).
    """

    mean: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    feature_names: list[str]
    df_clean: pd.DataFrame = field(repr=False)

    # Convenience accessors ------------------------------------------------

    @property
    def hgb_ret_cov(self) -> np.ndarray:
        """Return the 2×2 HGB-RET sub-covariance matrix."""
        idx_h = self.feature_names.index("HGB")
        idx_r = self.feature_names.index("RET")
        return self.cov[np.ix_([idx_h, idx_r], [idx_h, idx_r])]

    @property
    def mu_hgb(self) -> float:
        return float(self.mean[self.feature_names.index("HGB")])

    @property
    def mu_ret(self) -> float:
        return float(self.mean[self.feature_names.index("RET")])

    def save(self, output_dir: Path | str) -> None:
        """Persist baseline arrays to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "baseline_mean.npy", self.mean)
        np.save(out / "baseline_cov.npy", self.cov)
        np.save(out / "feature_names.npy", np.array(self.feature_names))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _bundled_csv_path() -> Path:
    """Resolve the path to the CSV bundled inside the package."""
    ref = importlib.resources.files("abp_synth") / "data" / "abps_data.csv"
    return Path(str(ref))


def load_abps_data(source: str = "bundled") -> pd.DataFrame:
    """Load the ABPS baseline dataset.

    Parameters:
        source:
            ``"bundled"`` — use the calibrated CSV shipped with the package (default).
            ``"remote"``  — download ``bloodcontrol.RData`` from CRAN GitHub and parse
            with *pyreadr* (requires the ``remote`` extra).
            Any other string is treated as a path to a local CSV file.

    Returns:
        A :class:`~pandas.DataFrame` with columns
        ``HGB``, ``RET``, ``OFF``, ``ABPS``.
    """
    if source == "bundled":
        return pd.read_csv(_bundled_csv_path())

    if source == "remote":
        return _load_remote()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _load_remote() -> pd.DataFrame:
    """Download bloodcontrol.RData from CRAN GitHub and parse with pyreadr."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required for remote loading. "
            "Install it with: pip install abp-synth[remote]"
        ) from exc

    import tempfile
    import urllib.request

    url = "https://raw.githubusercontent.com/cran/ABPS/master/data/bloodcontrol.RData"
    with tempfile.NamedTemporaryFile(suffix=".RData", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        result = pyreadr.read_r(tmp.name)

    raw_df = result["bloodcontrol"]
    # 转换为标准列名
    df = pd.DataFrame({
        "HGB": raw_df["HGB"],
        "RET": raw_df["RETP"],
        "OFF": raw_df["OFFscore"] if "OFFscore" in raw_df.columns else 10.0 * raw_df["HGB"] - 60.0 * np.sqrt(raw_df["RETP"]),
        "ABPS": raw_df["ABPS"]
    })
    return df


def _generate_literature_fallback(
    n: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reference dataset based on Sottas et al. (2008) and Sharpe et al. (2006)."""
    rng = np.random.default_rng(seed)

    mu_pop = np.array([15.5, np.log(1.15)])
    cov_inter = np.array([[0.90**2, 0.90 * 0.16 * 0.25], [0.90 * 0.16 * 0.25, 0.16**2]])
    cov_intra = np.array([[0.60**2, 0.60 * 0.14 * 0.20], [0.60 * 0.14 * 0.20, 0.14**2]])

    baselines = rng.multivariate_normal(mu_pop, cov_inter, n)
    samples = baselines + rng.multivariate_normal([0, 0], cov_intra, n)

    hgb = np.clip(samples[:, 0], 11.5, 18.5)
    ret = np.clip(np.exp(samples[:, 1]), 0.3, 2.5)
    off = 10.0 * hgb - 60.0 * np.sqrt(ret)

    z_hgb = (hgb - 15.5) / 0.85
    z_ret = (ret - 1.15) / 0.20
    abps = -1.2 + 0.5 * (z_hgb**2 + z_ret**2)**0.5 + rng.normal(0, 0.25, n)

    return pd.DataFrame({"HGB": hgb, "RET": ret, "OFF": off, "ABPS": abps})


# ---------------------------------------------------------------------------
# Baseline extraction
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES: list[str] = ["HGB", "RET", "OFF", "ABPS"]


def extract_baseline(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> BaselineResult:
    """Extract multivariate baseline statistics from a DataFrame."""
    if features is None:
        features = [c for c in _DEFAULT_FEATURES if c in df.columns]
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

    df_clean = df[features].dropna()

    return BaselineResult(
        mean=df_clean.mean().values,
        cov=df_clean.cov().values,
        corr=df_clean.corr().values,
        feature_names=list(features),
        df_clean=df_clean,
    )
