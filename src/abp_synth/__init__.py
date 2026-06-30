"""
abp_synth — Synthetic Athlete Biological Passport (ABP) time-series generator.

Provides tools for extracting physiological baselines from the ABPS dataset
and generating synthetic longitudinal HGB/RET profiles with optional
EPO doping pattern injection.
"""

__version__ = "0.1.0"

from abp_synth.baseline import BaselineResult, extract_baseline, load_abps_data
from abp_synth.synthesizer import SyntheticDataset, generate_dataset

__all__ = [
    "BaselineResult",
    "SyntheticDataset",
    "extract_baseline",
    "generate_dataset",
    "load_abps_data",
]
