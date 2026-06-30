"""
custom_doping_pattern.py — Inject EPO patterns with custom parameters.

Demonstrates fine-grained control over the synthesis process:
  - Custom baseline extraction
  - Manual normal athlete generation
  - EPO injection with user-specified parameters
  - Visualization

Usage:
    python examples/custom_doping_pattern.py
"""

import numpy as np

from abp_synth.baseline import extract_baseline, load_abps_data
from abp_synth.synthesizer import generate_normal_athlete, inject_epo_pattern
from abp_synth.visualization import plot_comparison

# Step 1: Load baseline
df = load_abps_data()
baseline = extract_baseline(df)
print(f"Baseline HGB: {baseline.mu_hgb:.2f} g/dL")
print(f"Baseline RET: {baseline.mu_ret:.2f} %")

# Step 2: Generate a single clean athlete with 14 tests
rng = np.random.default_rng(2024)
clean_seq = generate_normal_athlete(
    n_tests=14,
    mu_hgb=baseline.mu_hgb,
    mu_ret=baseline.mu_ret,
    cov_2d=baseline.hgb_ret_cov,
    rng=rng,
)
print(f"\nClean sequence shape: {clean_seq.shape}")

# Step 3: Inject a strong EPO pattern starting at test #4
doped_seq, labels = inject_epo_pattern(
    clean_seq,
    inject_at=4,
    hgb_rise=2.5,      # Strong HGB elevation
    ret_drop=0.45,      # Severe RET suppression
    rise_steps=3,
    drop_steps=5,
)

print(f"Modified points: {labels.sum()} / {len(labels)}")
print(f"HGB before injection (t=3): {clean_seq[3, 0]:.2f} → after: {doped_seq[3, 0]:.2f}")
print(f"HGB at peak (t=6):          {clean_seq[6, 0]:.2f} → after: {doped_seq[6, 0]:.2f}")
