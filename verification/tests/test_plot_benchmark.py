"""Tests for plot_benchmark.py -- the run-level 4-way comparison chart."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import plot_benchmark


def _write_accuracy_csv(path, rows):
    fieldnames = [
        "frame",
        "n_gt",
        "n_det",
        "n_tp",
        "n_fp",
        "n_fn",
        "precision",
        "recall",
        "f1",
        "mean_pos_error_px",
        "inference_time_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(frame, tp=8, fp=1, fn=1, err=1.5, inference_ms=10.0):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    return {
        "frame": frame,
        "n_gt": tp + fn,
        "n_det": tp + fp,
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_pos_error_px": err,
        "inference_time_ms": inference_ms,
    }


class TestAggregateFromCsv:
    def test_computes_mean_and_median_inference_time(self, tmp_path):
        csv_path = tmp_path / "accuracy_metrics_trackpy.csv"
        _write_accuracy_csv(
            csv_path,
            [_row(0, inference_ms=10.0), _row(1, inference_ms=20.0), _row(2, inference_ms=30.0)],
        )

        result = plot_benchmark._aggregate_from_csv(csv_path)

        assert result["mean_inference_ms"] == 20.0
        assert result["median_inference_ms"] == 20.0

    def test_missing_inference_time_column_returns_none(self, tmp_path):
        """Older CSVs (from before benchmark.py recorded per-frame timing)
        must not crash the aggregator -- both fields come back None."""
        csv_path = tmp_path / "accuracy_metrics_trackpy.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "n_gt",
                    "n_det",
                    "n_tp",
                    "n_fp",
                    "n_fn",
                    "precision",
                    "recall",
                    "f1",
                    "mean_pos_error_px",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "frame": 0,
                    "n_gt": 9,
                    "n_det": 9,
                    "n_tp": 8,
                    "n_fp": 1,
                    "n_fn": 1,
                    "precision": 0.8889,
                    "recall": 0.8889,
                    "f1": 0.8889,
                    "mean_pos_error_px": 1.5,
                }
            )

        result = plot_benchmark._aggregate_from_csv(csv_path)

        assert result["mean_inference_ms"] is None
        assert result["median_inference_ms"] is None


class TestMainEndToEnd:
    def test_generates_both_plots_for_single_model_no_tracking(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        _write_accuracy_csv(
            out_dir / "accuracy_metrics_trackpy.csv",
            [_row(i, inference_ms=10.0 + i) for i in range(5)],
        )

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        plot_benchmark.main()

        assert (out_dir / "benchmark_comparison.png").exists()
        assert (out_dir / "benchmark_summary.png").exists()
        captured = capsys.readouterr()
        assert "trackpy" in captured.out

    def test_generates_tracking_panels_when_tracking_csv_present(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        _write_accuracy_csv(
            out_dir / "accuracy_metrics_rf-detr.csv",
            [_row(i, inference_ms=50.0 + i) for i in range(5)],
        )
        with open(out_dir / "tracking_metrics_rf-detr.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["mota", "idf1", "num_fragmentations", "num_switches"]
            )
            writer.writeheader()
            writer.writerow({"mota": 0.5, "idf1": 0.6, "num_fragmentations": 12, "num_switches": 3})

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        plot_benchmark.main()

        captured = capsys.readouterr()
        assert "mota" in captured.out
        assert "id_switches" in captured.out
        assert (out_dir / "benchmark_summary.png").exists()

    def test_multiple_models_use_fixed_color_order(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        for model_type in ("rf-detr", "lodestar", "trackpy", "yolo"):
            _write_accuracy_csv(
                out_dir / f"accuracy_metrics_{model_type}.csv",
                [_row(i, inference_ms=10.0) for i in range(3)],
            )

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        plot_benchmark.main()

        assert (out_dir / "benchmark_summary.png").exists()
        # Every known model type keeps its own fixed color, never reassigned.
        seen = list(plot_benchmark._MODEL_COLORS.keys())
        for model_type in ("rf-detr", "lodestar", "trackpy", "yolo"):
            assert (
                plot_benchmark._color_for(model_type, seen)
                == plot_benchmark._MODEL_COLORS[model_type]
            )
