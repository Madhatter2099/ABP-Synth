"""
cli.py — Command-line interface for abp-synth.

Usage::

    # Generate full dataset with defaults
    abp-synth generate --output ./output

    # Extract baseline only
    abp-synth baseline --output ./output

    # Create visualizations from saved data
    abp-synth visualize --input ./output --output ./figures
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="abp-synth")
def main() -> None:
    """abp-synth — Synthetic ABP time-series generator for anti-doping research."""


# ---- baseline sub-command -------------------------------------------------


@main.command()
@click.option(
    "--source",
    default="bundled",
    help="Data source: 'bundled' (default), 'remote', or path to a CSV.",
)
@click.option(
    "--output",
    "-o",
    default="./output",
    type=click.Path(),
    help="Directory to save baseline arrays.",
)
def baseline(source: str, output: str) -> None:
    """Extract physiological baselines from the ABPS dataset."""
    from abp_synth.baseline import extract_baseline, load_abps_data

    click.echo("Loading ABPS data...")
    df = load_abps_data(source)
    click.echo(f"  Loaded {len(df)} records.")

    bl = extract_baseline(df)
    click.echo(f"  Features: {bl.feature_names}")
    click.echo(f"  HGB mean: {bl.mu_hgb:.4f} g/dL")
    click.echo(f"  RET mean: {bl.mu_ret:.4f} %")

    out = Path(output)
    bl.save(out)
    click.echo(f"  Baseline saved to {out.resolve()}")


# ---- generate sub-command -------------------------------------------------


@main.command()
@click.option("--n-normal", default=10_000, type=int, help="Number of clean athletes.")
@click.option("--n-doping", default=1_000, type=int, help="Number of EPO doping athletes.")
@click.option("--min-tests", default=8, type=int, help="Min tests per athlete.")
@click.option("--max-tests", default=16, type=int, help="Max tests per athlete.")
@click.option("--seed", default=2024, type=int, help="Random seed.")
@click.option(
    "--output",
    "-o",
    default="./output",
    type=click.Path(),
    help="Directory to save the generated dataset.",
)
@click.option("--plot/--no-plot", default=True, help="Generate comparison plots.")
def generate(
    n_normal: int,
    n_doping: int,
    min_tests: int,
    max_tests: int,
    seed: int,
    output: str,
    plot: bool,
) -> None:
    """Generate a full synthetic ABP dataset (clean + EPO injected)."""
    from abp_synth.baseline import extract_baseline, load_abps_data
    from abp_synth.synthesizer import generate_dataset
    from abp_synth.visualization import plot_comparison

    out = Path(output)

    click.echo("Step 1/3: Loading baseline...")
    df = load_abps_data()
    bl = extract_baseline(df)
    bl.save(out)
    click.echo(f"  HGB μ={bl.mu_hgb:.4f}, RET μ={bl.mu_ret:.4f}")

    click.echo(f"Step 2/3: Generating {n_normal} clean + {n_doping} doping sequences...")
    ds = generate_dataset(
        n_normal=n_normal,
        n_doping=n_doping,
        min_tests=min_tests,
        max_tests=max_tests,
        baseline=bl,
        seed=seed,
    )
    ds.save(out)
    click.echo(f"  {ds.summary()}")
    click.echo(f"  Saved to {out.resolve()}")

    if plot:
        click.echo("Step 3/3: Generating comparison plot...")
        fig_path = out / "comparison.png"
        plot_comparison(ds, bl, save_path=fig_path)
        click.echo(f"  Plot saved to {fig_path}")
    else:
        click.echo("Step 3/3: Skipped (--no-plot).")

    click.echo("Done!")


# ---- visualize sub-command ------------------------------------------------


@main.command()
@click.option(
    "--input",
    "-i",
    "input_dir",
    default="./output",
    type=click.Path(exists=True),
    help="Directory with saved baseline / dataset files.",
)
@click.option(
    "--output",
    "-o",
    default="./figures",
    type=click.Path(),
    help="Directory to save figures.",
)
def visualize(input_dir: str, output: str) -> None:
    """Generate all visualizations from saved baseline + dataset files."""
    import numpy as np
    import pandas as pd

    from abp_synth.baseline import BaselineResult
    from abp_synth.synthesizer import SyntheticDataset
    from abp_synth.visualization import (
        plot_comparison,
        plot_correlation,
        plot_distributions,
        plot_scatter_hgb_ret,
    )

    inp = Path(input_dir)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # Reconstruct baseline
    mean = np.load(inp / "baseline_mean.npy")
    cov = np.load(inp / "baseline_cov.npy")
    feat = np.load(inp / "feature_names.npy", allow_pickle=True).tolist()

    from abp_synth.baseline import load_abps_data

    df = load_abps_data()
    df_clean = df[feat].dropna()
    corr = df_clean.corr().values

    bl = BaselineResult(
        mean=mean,
        cov=cov,
        corr=corr,
        feature_names=feat,
        df_clean=df_clean,
    )

    click.echo("Generating baseline plots...")
    plot_distributions(bl, save_path=out / "distribution.png")
    plot_correlation(bl, save_path=out / "correlation.png")

    if "HGB" in feat and "RET" in feat:
        plot_scatter_hgb_ret(bl, save_path=out / "scatter.png")

    # Try to load dataset for comparison plot
    try:
        ds = SyntheticDataset.load(inp)
        click.echo("Generating comparison plot...")
        plot_comparison(ds, bl, save_path=out / "comparison.png")
    except FileNotFoundError:
        click.echo("  No synthetic dataset found; skipping comparison plot.")

    click.echo(f"Figures saved to {out.resolve()}")
