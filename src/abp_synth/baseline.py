"""
baseline.py — ABPS baseline data loading and statistical parameter extraction.

This module provides functionality to:
1. Load the ABPS (Anti-doping Blood Profile Score) dataset from bundled CSV,
   a local file, or a remote source.
2. Extract multivariate statistical baselines (mean vector, covariance matrix,
   correlation matrix) used for downstream synthetic data generation.

References:
    - Sottas, P.E. et al. (2008). Biostatistics, 9(2), 285-296.
    - Sharpe, K. et al. (2006). Haematologica, 91(12), 1603-1610.
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
        mean: Mean vector of selected features.  Shape ``(n_features,)``.
        cov: Covariance matrix.  Shape ``(n_features, n_features)``.
        corr: Pearson correlation matrix.  Shape ``(n_features, n_features)``.
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
    # For editable installs the traversable *is* a Path already.
    return Path(str(ref))


def load_abps_data(source: str = "bundled") -> pd.DataFrame:
    """Load the ABPS baseline dataset.

    Parameters:
        source:
            ``"bundled"`` — use the CSV shipped with the package (default).
            ``"remote"``  — download the ``.rda`` from CRAN GitHub and parse
            with *pyreadr* (requires the ``remote`` extra).
            Any other string is treated as a path to a local CSV file.

    Returns:
        A :class:`~pandas.DataFrame` with at least columns
        ``HGB``, ``RET``, ``OFF``, ``ABPS``.
    """
    if source == "bundled":
        return pd.read_csv(_bundled_csv_path())

    if source == "remote":
        return _load_remote()

    # Treat as local path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _load_remote() -> pd.DataFrame:
    """Download the ABPS .rda from CRAN GitHub and parse with pyreadr."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required for remote loading.  "
            "Install it with:  pip install abp-synth[remote]"
        ) from exc

    import tempfile
    import urllib.request

    url = "https://github.com/cran/ABPS/raw/master/data/abps.rda"
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        result = pyreadr.read_r(tmp.name)

    key = next(iter(result))
    return result[key]


def _generate_literature_fallback(
    n: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a reference dataset from published literature parameters.

    Used as a last-resort fallback when neither bundled CSV nor remote
    sources are available.

    The parameters are drawn from:
        - Sharpe et al. (2006): HGB normality in male athletes.
        - Sottas et al. (2008): Bayesian ABP framework.
    """
    rng = np.random.default_rng(seed)

    mean = [14.5, np.log(1.0)]
    cov = [[1.44, 0.18], [0.18, 0.04]]
    samples = rng.multivariate_normal(mean, cov, n)

    hgb = samples[:, 0]
    ret = np.exp(samples[:, 1])
    off = hgb - 60 * np.sqrt(ret / 100)
    abps = np.clip(rng.beta(1.5, 15, n), 0, 1)

    return pd.DataFrame({"HGB": hgb, "RET": ret, "OFF": off, "ABPS": abps})


# ---------------------------------------------------------------------------
# Baseline extraction
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES: list[str] = ["HGB", "RET", "OFF", "ABPS"]


def extract_baseline(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> BaselineResult:
    """Extract multivariate baseline statistics from a DataFrame.

    Parameters:
        df: Raw ABPS DataFrame (e.g. from :func:`load_abps_data`).
        features: Column names to include.  Defaults to
            ``["HGB", "RET", "OFF", "ABPS"]``.

    Returns:
        A :class:`BaselineResult` containing the mean vector, covariance
        matrix, correlation matrix, and the cleaned DataFrame.
    """
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
