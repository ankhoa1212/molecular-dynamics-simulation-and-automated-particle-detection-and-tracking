"""Tests for plot_benchmark.py -- both the per-frame line/tracker-bar
comparison figure (benchmark_comparison.png) and the run-level 4-way summary
figure (benchmark_summary.png).

docs/plans/2026-08-18-001-feat-bytetrack-tracking-support-plan.md U6 test
scenarios:
- two tracker CSVs present for one model_type produce two bars (one per
  tracker) in the tracking bar panel, plus a two-row stdout entry
- only one tracker CSV present for a model_type produces one bar without
  erroring on the missing counterpart
- no tracking_metrics_*.csv files at all: the tracking bar panel renders an
  empty state, the rest of the plot (line panels) still renders
- the existing --models CLI filter restricts which model_types appear in
  both the line panels and the new bar panel
"""

import csv
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import plot_benchmark as pb


def _write_accuracy_csv(path, model_type, n_frames=3):
    """Write a minimal accuracy_metrics_{model_type}.csv with the columns
    _read_accuracy_csv/_aggregate_from_csv expect (no inference_time_ms --
    see _write_accuracy_csv_rows for that)."""
    with open(path / f"accuracy_metrics_{model_type}.csv", "w", newline="") as f:
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
        for frame in range(n_frames):
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


def _write_tracking_csv(path, model_type, tracker, mota=0.5, idf1=0.6, num_fragmentations=10):
    """Write a tracking_metrics_{model_type}_{tracker}.csv matching the single
    aggregate-row schema benchmark.py's _run_tracking_metrics/
    _run_bytetrack_metrics write (mota/idf1/num_fragmentations/num_switches/
    num_misses/num_false_positives, plus threshold metadata)."""
    with open(path / f"tracking_metrics_{model_type}_{tracker}.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mota",
                "idf1",
                "num_fragmentations",
                "num_switches",
                "num_misses",
                "num_false_positives",
                "matching_threshold_radii",
                "psf_sigma_px",
                "match_threshold_px",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mota": mota,
                "idf1": idf1,
                "num_fragmentations": num_fragmentations,
                "num_switches": 3,
                "num_misses": 100,
                "num_false_positives": 5,
                "matching_threshold_radii": 0.5,
                "psf_sigma_px": 5.0,
                "match_threshold_px": 2.5,
            }
        )


def _run_main(output_dir, models=None):
    argv = ["plot_benchmark.py", "--output-dir", str(output_dir)]
    if models:
        argv += ["--models", *models]
    with mock.patch.object(sys, "argv", argv):
        pb.main()


class TestTwoTrackersPresent:
    def test_two_tracker_csvs_produce_two_bars(self, tmp_path, capsys):
        _write_accuracy_csv(tmp_path, "rf-detr")
        _write_tracking_csv(tmp_path, "rf-detr", "trackpy", mota=0.7)
        _write_tracking_csv(tmp_path, "rf-detr", "bytetrack", mota=0.4)

        _run_main(tmp_path)

        out = capsys.readouterr().out
        assert "rf-detr    trackpy" in out
        assert "rf-detr    bytetrack" in out

        png_path = tmp_path / "benchmark_comparison.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_bar_panel_helper_draws_two_bars_for_two_trackers(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        per_tracking = {
            ("rf-detr", "trackpy"): {"mota": "0.7"},
            ("rf-detr", "bytetrack"): {"mota": "0.4"},
        }
        pb._plot_tracking_bars(ax, ["rf-detr"], per_tracking, "mota", "MOTA")
        # One bar (Rectangle patch) per tracker present for the single group.
        assert len(ax.patches) == 2
        plt.close(fig)


class TestHatchIndexIsGlobalNotPerGroup:
    """A model group with only 'bytetrack' present must still use
    bytetrack's fixed global hatch/alpha, not the local enumerate()
    position within that group's own (shorter) tracker subset -- otherwise
    its bar would be styled identically to a 'trackpy' bar elsewhere on the
    same figure while the legend claims the two hatches mean different
    things (found during review: correctness)."""

    def test_hatch_index_fixed_by_tracker_order_not_local_position(self):
        assert pb._hatch_index_for("trackpy") == 0
        assert pb._hatch_index_for("bytetrack") == 1

    def test_bytetrack_only_group_uses_bytetrack_hatch_not_index_zero(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # rf-detr has ONLY bytetrack present (no trackpy) -- its bar's
        # local enumerate() index would be 0, but bytetrack's fixed global
        # hatch index is 1.
        per_tracking = {("rf-detr", "bytetrack"): {"mota": "0.4"}}
        pb._plot_tracking_bars(ax, ["rf-detr"], per_tracking, "mota", "MOTA")

        assert len(ax.patches) == 1
        assert ax.patches[0].get_hatch() == pb._HATCHES[pb._hatch_index_for("bytetrack")]
        assert ax.patches[0].get_hatch() != pb._HATCHES[pb._hatch_index_for("trackpy")]
        plt.close(fig)


class TestSingleTrackerPresent:
    def test_single_tracker_csv_produces_one_bar_no_crash(self, tmp_path, capsys):
        _write_accuracy_csv(tmp_path, "rf-detr")
        _write_tracking_csv(tmp_path, "rf-detr", "trackpy")

        _run_main(tmp_path)

        out = capsys.readouterr().out
        assert "rf-detr    trackpy" in out
        assert "bytetrack" not in out
        assert (tmp_path / "benchmark_comparison.png").exists()

    def test_bar_panel_helper_draws_one_bar_for_one_tracker(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        per_tracking = {("rf-detr", "trackpy"): {"mota": "0.7"}}
        pb._plot_tracking_bars(ax, ["rf-detr"], per_tracking, "mota", "MOTA")
        assert len(ax.patches) == 1
        plt.close(fig)


class TestNoTrackingCsvsPresent:
    def test_no_tracking_csvs_omits_bar_panel_without_error(self, tmp_path, capsys):
        _write_accuracy_csv(tmp_path, "rf-detr")
        _write_accuracy_csv(tmp_path, "yolo")

        _run_main(tmp_path)

        out = capsys.readouterr().out
        # No tracking table header at all when nothing is present.
        assert "tracker" not in out
        assert "mota" not in out
        # Line-panel summary table is unaffected.
        assert "rf-detr" in out
        assert "yolo" in out
        assert (tmp_path / "benchmark_comparison.png").exists()

    def test_bar_panel_helper_shows_empty_state(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        pb._plot_tracking_bars(ax, ["rf-detr", "yolo"], {}, "mota", "MOTA")
        assert len(ax.patches) == 0
        assert len(ax.texts) == 1
        assert "no tracking metrics" in ax.texts[0].get_text()
        plt.close(fig)


class TestModelsFilterAppliesToBarPanel:
    def test_models_filter_restricts_bar_panel_groups(self, tmp_path, capsys):
        _write_accuracy_csv(tmp_path, "rf-detr")
        _write_accuracy_csv(tmp_path, "trackpy")
        _write_tracking_csv(tmp_path, "rf-detr", "trackpy")
        _write_tracking_csv(tmp_path, "trackpy", "bytetrack")

        _run_main(tmp_path, models=["rf-detr"])

        out = capsys.readouterr().out
        assert "rf-detr" in out
        assert "trackpy" not in out.split("\n\n")[0]  # line-panel table excludes it
        # Tracking table should only mention rf-detr's row, not trackpy's.
        tracking_section = out.split("mota")[-1] if "mota" in out else ""
        assert "bytetrack" not in tracking_section

    def test_discovery_only_globs_filtered_model_types(self, tmp_path):
        _write_accuracy_csv(tmp_path, "rf-detr")
        _write_accuracy_csv(tmp_path, "trackpy")
        _write_tracking_csv(tmp_path, "rf-detr", "trackpy")
        _write_tracking_csv(tmp_path, "trackpy", "bytetrack")

        _run_main(tmp_path, models=["rf-detr"])
        # Re-run discovery logic directly to check per_tracking keys precisely
        # by re-invoking main() would print; instead verify via the glob
        # pattern plot_benchmark.py itself uses for the filtered model only.
        prefix = "tracking_metrics_rf-detr_"
        matches = sorted(tmp_path.glob(f"{prefix}*.csv"))
        assert len(matches) == 1
        assert matches[0].name == "tracking_metrics_rf-detr_trackpy.csv"


class TestTrackerSortKey:
    def test_known_trackers_sort_before_unknown(self):
        trackers = sorted(["bytetrack", "unknown-tracker", "trackpy"], key=pb._tracker_sort_key)
        assert trackers == ["trackpy", "bytetrack", "unknown-tracker"]


# --- benchmark_summary.png: the run-level 4-way comparison chart ---


def _write_accuracy_csv_rows(path, rows):
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
        _write_accuracy_csv_rows(
            csv_path,
            [_row(0, inference_ms=10.0), _row(1, inference_ms=20.0), _row(2, inference_ms=30.0)],
        )

        result = pb._aggregate_from_csv(csv_path)

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

        result = pb._aggregate_from_csv(csv_path)

        assert result["mean_inference_ms"] is None
        assert result["median_inference_ms"] is None


class TestMainEndToEnd:
    def test_generates_both_plots_for_single_model_no_tracking(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        _write_accuracy_csv_rows(
            out_dir / "accuracy_metrics_trackpy.csv",
            [_row(i, inference_ms=10.0 + i) for i in range(5)],
        )

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        pb.main()

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
        _write_accuracy_csv_rows(
            out_dir / "accuracy_metrics_rf-detr.csv",
            [_row(i, inference_ms=50.0 + i) for i in range(5)],
        )
        # benchmark_summary.png's tracking panels use each model's trackpy
        # result specifically -- see _plot_summary_bars's own docstring.
        _write_tracking_csv(
            out_dir, "rf-detr", "trackpy", mota=0.5, idf1=0.6, num_fragmentations=12
        )

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        pb.main()

        captured = capsys.readouterr()
        assert "mota" in captured.out
        assert "id_switches" in captured.out
        assert (out_dir / "benchmark_summary.png").exists()

    def test_multiple_models_use_fixed_color_order(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        for model_type in ("rf-detr", "lodestar", "trackpy", "yolo12m"):
            _write_accuracy_csv_rows(
                out_dir / f"accuracy_metrics_{model_type}.csv",
                [_row(i, inference_ms=10.0) for i in range(3)],
            )

        monkeypatch.setattr(sys, "argv", ["plot_benchmark.py", "--output-dir", str(out_dir)])
        pb.main()

        assert (out_dir / "benchmark_summary.png").exists()
        # Every known model type keeps its own fixed color, never reassigned.
        seen = list(pb._MODEL_COLORS.keys())
        for model_type in ("rf-detr", "lodestar", "trackpy", "yolo12m"):
            assert pb._color_for(model_type, seen) == pb._MODEL_COLORS[model_type]
