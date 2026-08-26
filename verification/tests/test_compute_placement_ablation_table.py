"""Tests for compute_placement_ablation_table.py -- the placement-ablation
table (WACV paper Table 7) regenerated from physics/random RF-DETR
accuracy_metrics_*.csv files.
"""

import csv
import sys
from pathlib import Path

import pytest

_VERIFICATION_DIR = Path(__file__).parent.parent
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

from compute_placement_ablation_table import (  # noqa: E402
    TRACKPY_BASELINE,
    build_rows,
    _print_latex,
    _print_table,
)


def _write_accuracy_csv(path: Path, per_frame_rows: list[dict]) -> None:
    fieldnames = ["frame", "n_gt", "n_det", "n_tp", "n_fp", "n_fn", "precision", "recall", "f1"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        base_row = {"n_gt": 100, "n_det": 100, "n_tp": 90, "n_fp": 10, "n_fn": 10}
        for i, row in enumerate(per_frame_rows):
            writer.writerow({"frame": i, **base_row, **row})


class TestBuildRows:
    def test_physics_and_random_rows_are_per_frame_means(self, tmp_path):
        physics_path = tmp_path / "physics.csv"
        random_path = tmp_path / "random.csv"
        _write_accuracy_csv(
            physics_path,
            [
                {"precision": 0.80, "recall": 0.70, "f1": 0.75},
                {"precision": 0.90, "recall": 0.80, "f1": 0.85},
            ],
        )
        _write_accuracy_csv(
            random_path,
            [
                {"precision": 0.60, "recall": 0.50, "f1": 0.55},
                {"precision": 0.70, "recall": 0.60, "f1": 0.65},
            ],
        )

        rows = build_rows(physics_path, random_path)

        assert rows["physics"]["precision"] == pytest.approx(0.85)
        assert rows["physics"]["recall"] == pytest.approx(0.75)
        assert rows["physics"]["f1"] == pytest.approx(0.80)
        assert rows["random"]["precision"] == pytest.approx(0.65)
        assert rows["random"]["recall"] == pytest.approx(0.55)
        assert rows["random"]["f1"] == pytest.approx(0.60)

    def test_trackpy_row_is_the_documented_baseline_constant(self, tmp_path):
        physics_path = tmp_path / "physics.csv"
        random_path = tmp_path / "random.csv"
        _write_accuracy_csv(physics_path, [{"precision": 0.8, "recall": 0.7, "f1": 0.75}])
        _write_accuracy_csv(random_path, [{"precision": 0.6, "recall": 0.5, "f1": 0.55}])

        rows = build_rows(physics_path, random_path)

        assert rows["trackpy"] == TRACKPY_BASELINE
        # Must be a copy, not the same dict object, so callers can't mutate the module constant.
        assert rows["trackpy"] is not TRACKPY_BASELINE


class TestPrintTable:
    def test_plain_table_reports_all_three_rows_as_percentages(self, tmp_path, capsys):
        rows = {
            "physics": {"precision": 0.847, "recall": 0.733, "f1": 0.785},
            "random": {"precision": 0.792, "recall": 0.624, "f1": 0.698},
            "trackpy": dict(TRACKPY_BASELINE),
        }

        _print_table(rows)

        out = capsys.readouterr().out
        assert "84.7%" in out and "73.3%" in out and "78.5%" in out
        assert "79.2%" in out and "62.4%" in out and "69.8%" in out
        assert "55.8%" in out and "44.5%" in out and "49.5%" in out

    def test_latex_table_bolds_the_highest_f1_row(self, capsys):
        rows = {
            "physics": {"precision": 0.847, "recall": 0.733, "f1": 0.785},
            "random": {"precision": 0.792, "recall": 0.624, "f1": 0.698},
            "trackpy": dict(TRACKPY_BASELINE),
        }

        _print_latex(rows)

        out = capsys.readouterr().out
        assert r"\textbf{78.5\%}" in out
        assert r"\textbf{69.8\%}" not in out
        assert r"\textbf{49.5\%}" not in out
        assert out.count(r"\\") == 4  # header row + 3 data rows
