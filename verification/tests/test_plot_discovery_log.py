"""Tests for the per-model discovery log message in plot_density_ablation.py.

Validates:
- Requirements 7.4: When plot_density_ablation.py discovers all expected CSV files for
  a model, it SHALL log an informational message confirming successful file discovery
  for that model.
- Requirements 7.5: If a model's CSV is missing for a given N, plot_density_ablation.py
  SHALL log a warning and skip that point without aborting.
"""

import csv
import sys
from pathlib import Path
from unittest import mock

# Ensure the verification directory is on the path so we can import the script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import plot_density_ablation as pda

_ALL_MODELS = ["rf-detr", "yolo12m", "yolo12n", "lodestar", "trackpy"]
_ALL_N = [200, 600, 1000, 1446]


def _write_accuracy_csv(n_dir: Path, model_type: str) -> None:
    """Write a minimal accuracy_metrics_{model_type}.csv with the columns
    _aggregate_from_csv expects."""
    csv_path = n_dir / f"accuracy_metrics_{model_type}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "precision",
                "recall",
                "f1",
                "mean_pos_error_px",
                "n_tp",
                "n_fp",
                "n_fn",
            ],
        )
        writer.writeheader()
        for frame in range(3):
            writer.writerow(
                {
                    "frame": frame,
                    "precision": 0.9,
                    "recall": 0.8,
                    "f1": 0.85,
                    "mean_pos_error_px": 1.5,
                    "n_tp": 10,
                    "n_fp": 1,
                    "n_fn": 2,
                }
            )


def _create_full_ablation_dir(base: Path, models=None, n_values=None) -> Path:
    """Create a complete ablation directory structure with CSVs for all
    (model, N) combinations.  Returns the ablation_dir path."""
    if models is None:
        models = _ALL_MODELS
    if n_values is None:
        n_values = _ALL_N
    ablation_dir = base / "density_ablation"
    for n in n_values:
        n_dir = ablation_dir / f"N{n}"
        n_dir.mkdir(parents=True, exist_ok=True)
        for model_type in models:
            _write_accuracy_csv(n_dir, model_type)
    return ablation_dir


def _run_main(ablation_dir: Path) -> None:
    """Invoke plot_density_ablation.main() with the given ablation-dir."""
    with mock.patch.object(
        sys, "argv", ["plot_density_ablation.py", "--ablation-dir", str(ablation_dir)]
    ):
        pda.main()


class TestDiscoveryLogAllModels:
    """Requirement 7.4: per-model success log message fires for all five models."""

    def test_found_message_printed_for_all_five_models(self, tmp_path, capsys):
        ablation_dir = _create_full_ablation_dir(tmp_path)
        _run_main(ablation_dir)
        out = capsys.readouterr().out
        for model_type in _ALL_MODELS:
            assert f"Found all N-points for '{model_type}'" in out, (
                f"Expected discovery confirmation for '{model_type}' in stdout.\n"
                f"Actual stdout:\n{out}"
            )

    def test_found_message_not_printed_when_model_has_no_data(self, tmp_path, capsys):
        """If a model has zero CSV files (completely absent), no discovery message
        should appear for it — but since auto-discovery only returns models that
        have at least one CSV, this model won't be in model_types at all.  Just
        confirm the five present models still get their messages."""
        # Build full ablation dir for only 4 models (omit lodestar)
        ablation_dir = _create_full_ablation_dir(
            tmp_path, models=["rf-detr", "yolo12m", "yolo12n", "trackpy"]
        )
        _run_main(ablation_dir)
        out = capsys.readouterr().out
        for model_type in ["rf-detr", "yolo12m", "yolo12n", "trackpy"]:
            assert f"Found all N-points for '{model_type}'" in out
        # lodestar was never discovered — no message for it
        assert "Found all N-points for 'lodestar'" not in out


class TestWarningPathAndPngOutput:
    """Requirement 7.5: warning fired and PNG still written when one CSV is absent."""

    def test_warning_logged_when_one_csv_missing(self, tmp_path, capsys):
        ablation_dir = _create_full_ablation_dir(tmp_path)
        # Remove one CSV for rf-detr at N=600
        missing_csv = ablation_dir / "N600" / "accuracy_metrics_rf-detr.csv"
        missing_csv.unlink()

        _run_main(ablation_dir)

        out = capsys.readouterr().out
        assert "Warning:" in out, f"Expected 'Warning:' in stdout.\nActual:\n{out}"
        assert "rf-detr" in out
        assert "N=600" in out or "600" in out

    def test_png_still_written_when_one_csv_missing(self, tmp_path, capsys):
        ablation_dir = _create_full_ablation_dir(tmp_path)
        # Remove one CSV for yolo12n at N=200
        (ablation_dir / "N200" / "accuracy_metrics_yolo12n.csv").unlink()

        _run_main(ablation_dir)
        capsys.readouterr()

        png_path = ablation_dir / "density_ablation.png"
        assert png_path.exists(), "density_ablation.png must be written even when a CSV is absent"
        assert png_path.stat().st_size > 0

    def test_no_discovery_message_for_model_with_missing_n_point(self, tmp_path, capsys):
        """When a model is missing one N-point, it should NOT get a
        'Found all N-points' message."""
        ablation_dir = _create_full_ablation_dir(tmp_path)
        # Remove one CSV for trackpy at N=1446
        (ablation_dir / "N1446" / "accuracy_metrics_trackpy.csv").unlink()

        _run_main(ablation_dir)
        out = capsys.readouterr().out

        assert (
            "Found all N-points for 'trackpy'" not in out
        ), "Should not print success message for a model that is missing a data point."
        # The other four models should still get their messages
        for model_type in ["rf-detr", "yolo12m", "yolo12n", "lodestar"]:
            assert f"Found all N-points for '{model_type}'" in out

    def test_warning_and_png_with_multiple_missing_csvs(self, tmp_path, capsys):
        """Multiple missing CSVs across different models: still produces PNG, warns for each."""
        ablation_dir = _create_full_ablation_dir(tmp_path)
        (ablation_dir / "N200" / "accuracy_metrics_lodestar.csv").unlink()
        (ablation_dir / "N1000" / "accuracy_metrics_yolo12m.csv").unlink()

        _run_main(ablation_dir)
        out = capsys.readouterr().out

        # Both gaps produce warnings
        warning_count = out.count("Warning:")
        assert (
            warning_count >= 2
        ), f"Expected at least 2 Warning lines, got {warning_count}.\nStdout:\n{out}"
        # PNG is still produced
        assert (ablation_dir / "density_ablation.png").exists()
