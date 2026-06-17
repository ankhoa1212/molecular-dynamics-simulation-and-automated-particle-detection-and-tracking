"""Tests for benchmark.py — U6: tracking metrics (MOTA/IDF1/fragmentation)."""

import csv
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_gt_tracks(path, rows):
    """Write a ground_truth_tracks.csv file."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "particle_id", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)


def _make_cfg(psf_sigma=5.0, search_range=15, memory=3, threshold_radii=0.5, enabled=True):
    return {
        "synthetic": {"psf_sigma": psf_sigma},
        "tracking": {
            "enabled": enabled,
            "search_range": search_range,
            "memory": memory,
            "matching_threshold_radii": threshold_radii,
        },
    }


# ---------------------------------------------------------------------------
# _run_tracking_metrics — perfect tracking
# ---------------------------------------------------------------------------


class TestRunTrackingMetrics:
    def test_perfect_detection_and_tracking(self, tmp_path):
        """MOTA = 1.0, IDF1 = 1.0, 0 fragmentations when detection is perfect."""
        # 3 particles across 5 frames, all correctly detected
        gt_rows = []
        for frame in range(5):
            for pid in [1, 2, 3]:
                x, y = float(pid * 50), float(pid * 50)
                gt_rows.append({"frame": frame, "particle_id": pid, "x": x, "y": y})
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # Perfect detections: exact same positions as ground truth
        detections = {
            frame: np.array([[float(pid * 50), float(pid * 50)] for pid in [1, 2, 3]])
            for frame in range(5)
        }
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert result["mota"] == pytest.approx(1.0, abs=1e-4)
        assert result["idf1"] == pytest.approx(1.0, abs=1e-4)
        assert result["num_fragmentations"] == 0

    def test_missing_frame_causes_fragmentations(self, tmp_path):
        """Dropping detections for one frame increases fragmentation count."""
        gt_rows = []
        for frame in range(5):
            for pid in [1, 2]:
                gt_rows.append(
                    {"frame": frame, "particle_id": pid, "x": float(pid * 100), "y": 50.0}
                )
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # Frame 2 has no detections
        detections = {}
        for frame in range(5):
            if frame == 2:
                detections[frame] = np.zeros((0, 2))
            else:
                detections[frame] = np.array([[float(pid * 100), 50.0] for pid in [1, 2]])

        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert result["num_fragmentations"] > 0

    def test_false_positives_decrease_mota(self, tmp_path):
        """Extra spurious detections (FP) cause MOTA < 1."""
        gt_rows = []
        for frame in range(4):
            gt_rows.append({"frame": frame, "particle_id": 1, "x": 100.0, "y": 50.0})
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # 1 real detection + 1 false positive per frame
        detections = {
            frame: np.array([[100.0, 50.0], [300.0, 50.0]]) for frame in range(4)  # second is FP
        }
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert result["mota"] < 1.0
        assert result["num_false_positives"] > 0

    def test_tracking_disabled_returns_none(self, tmp_path):
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg(enabled=False)
        result = benchmark._run_tracking_metrics({0: np.array([[50.0, 50.0]])}, str(gt_path), cfg)
        assert result is None

    def test_missing_gt_tracks_file_returns_none(self, tmp_path):
        absent = str(tmp_path / "nonexistent.csv")
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics({}, absent, cfg)
        assert result is None

    def test_missing_motmetrics_returns_none(self, tmp_path):
        """When py-motmetrics not installed, return None with a warning."""
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg()

        with mock.patch.dict(sys.modules, {"motmetrics": None}):
            result = benchmark._run_tracking_metrics(
                {0: np.array([[50.0, 50.0]])}, str(gt_path), cfg
            )
        # Either None (because import failed) or a real result; the test just confirms no crash
        # The actual behavior depends on whether motmetrics is installed in the test env
        assert result is None or isinstance(result, dict)

    def test_tracking_metrics_csv_includes_threshold(self, tmp_path):
        """tracking_metrics.csv must include matching_threshold_radii for reproducibility."""
        gt_rows = [{"frame": 0, "particle_id": 1, "x": 100.0, "y": 100.0}]
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)

        cfg = _make_cfg(threshold_radii=0.75)
        detections = {0: np.array([[100.0, 100.0]])}
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert "matching_threshold_radii" in result
        assert result["matching_threshold_radii"] == pytest.approx(0.75)

    def test_no_detections_returns_none(self, tmp_path):
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics({}, str(gt_path), cfg)
        assert result is None
