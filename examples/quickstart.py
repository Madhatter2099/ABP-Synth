"""
quickstart.py — Generate a synthetic ABP dataset in 5 lines.

Usage:
    python examples/quickstart.py
"""

from abp_synth import generate_dataset

# Generate 1000 clean athletes + 100 EPO doping profiles
dataset = generate_dataset(n_normal=1000, n_doping=100, seed=42)

# Print summary
print(dataset.summary())

# Save to disk
dataset.save("./output")

# Convert to a pandas DataFrame for further analysis
df = dataset.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(df.head(10))
