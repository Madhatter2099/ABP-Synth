"""Tests for abp_synth.cli module (integration)."""

from click.testing import CliRunner

from abp_synth.cli import main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "abp-synth" in result.output


def test_baseline_command(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["baseline", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "baseline_mean.npy").exists()


def test_generate_command(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "--n-normal", "50",
            "--n-doping", "5",
            "--output", str(tmp_path),
            "--no-plot",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "synthetic_sequences.npy").exists()
    assert (tmp_path / "synthetic_labels_athlete.npy").exists()


def test_generate_with_plot(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "--n-normal", "50",
            "--n-doping", "5",
            "--output", str(tmp_path),
            "--plot",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "comparison.png").exists()


def test_visualize_command(tmp_path):
    runner = CliRunner()
    # First generate data
    runner.invoke(
        main,
        ["generate", "--n-normal", "30", "--n-doping", "3",
         "--output", str(tmp_path), "--no-plot"],
    )
    # Then visualize
    fig_dir = tmp_path / "figs"
    result = runner.invoke(
        main,
        ["visualize", "--input", str(tmp_path), "--output", str(fig_dir)],
    )
    assert result.exit_code == 0
    assert (fig_dir / "distribution.png").exists()
    assert (fig_dir / "correlation.png").exists()
