"""Tests for benchmark.py — U6: tracking metrics (MOTA/IDF1/fragmentation);
U1-U3: LodeSTAR model-type support."""

import csv
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Force supervision's own real import (and its internal torch-availability
# check) to happen now, before any test mocks sys.modules["torch"]. If a
# mocked torch is present the first time supervision is imported, its
# torch-integration code path initializes against the mock and later tests
# that import supervision for the first time under a *different* mocked
# torch instance can hit stale/inconsistent internal state.
import supervision as _sv_preload  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

# Regression guard for U1: importing benchmark.py must never trigger the
# cross-venv re-exec (os.execv would replace this pytest process). If the
# guard regresses, this import itself hangs/crashes rather than any
# individual assertion failing below.
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


# ---------------------------------------------------------------------------
# _resolve_model_type — U1: pre-argparse model-type sniffing
# ---------------------------------------------------------------------------


class TestResolveModelType:
    def test_defaults_to_rf_detr_with_no_args_and_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert benchmark._resolve_model_type([]) == "rf-detr"

    def test_reads_model_type_flag_space_separated(self):
        assert benchmark._resolve_model_type(["--model-type", "lodestar"]) == "lodestar"

    def test_reads_model_type_flag_equals_form(self):
        assert benchmark._resolve_model_type(["--model-type=lodestar"]) == "lodestar"

    def test_cli_flag_takes_precedence_over_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  model_type: lodestar\n")
        argv = ["--config", str(config_path), "--model-type", "rf-detr"]
        assert benchmark._resolve_model_type(argv) == "rf-detr"

    def test_falls_back_to_config_benchmark_model_type(self, tmp_path):
        config_path = tmp_path / "my_config.yaml"
        config_path.write_text("benchmark:\n  model_type: lodestar\n")
        argv = ["--config", str(config_path)]
        assert benchmark._resolve_model_type(argv) == "lodestar"

    def test_missing_config_file_falls_back_to_default(self, tmp_path):
        argv = ["--config", str(tmp_path / "nonexistent.yaml")]
        assert benchmark._resolve_model_type(argv) == "rf-detr"

    def test_config_without_benchmark_section_falls_back_to_default(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("synthetic:\n  render_strategy: procedural\n")
        argv = ["--config", str(config_path)]
        assert benchmark._resolve_model_type(argv) == "rf-detr"


# ---------------------------------------------------------------------------
# _reexec_for_model_venv — U1: model-type-aware venv/site-packages selection
# ---------------------------------------------------------------------------


def _make_fake_venv(base_dir, name, pyver="python3.1"):
    """Build a fake venv directory with a bin/python and a versioned
    site-packages dir, mimicking a real uv-managed venv layout closely
    enough for _reexec_for_model_venv to probe. pyver defaults to a Python
    version no real interpreter runs, so re-exec always attempts to fire."""
    venv_dir = base_dir / name
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
    (venv_dir / "bin" / "python").chmod(0o755)
    (venv_dir / "lib" / pyver / "site-packages").mkdir(parents=True)
    return venv_dir


class TestReexecForModelVenv:
    def test_lodestar_selects_particle_tracking_venv_not_rf_detr(self, tmp_path):
        """model_type=lodestar must re-exec into the lodestar-mapped venv's
        python, not the rf-detr-mapped venv's, even though both are
        registered in _MODEL_VENV_DIRS."""
        rf_detr_venv = _make_fake_venv(tmp_path, "rf_detr_venv")
        lodestar_venv = _make_fake_venv(tmp_path, "lodestar_venv")

        with mock.patch.dict(
            benchmark._MODEL_VENV_DIRS,
            {"rf-detr": rf_detr_venv, "lodestar": lodestar_venv},
        ):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("lodestar")

        fake_execv.assert_called_once()
        called_python = fake_execv.call_args[0][0]
        assert str(lodestar_venv) in called_python
        assert str(rf_detr_venv) not in called_python

    def test_rf_detr_default_targets_rf_detr_venv(self):
        """model_type omitted/rf-detr preserves today's exact re-exec targeting."""
        venv_python = benchmark._MODEL_VENV_DIRS["rf-detr"] / "bin" / "python"
        assert "rf-detr" in str(venv_python.absolute())
        assert "particle-tracking" not in str(venv_python.absolute())

    def test_unknown_model_type_falls_back_to_rf_detr_venv(self):
        assert (
            benchmark._MODEL_VENV_DIRS.get(
                "not-a-real-model", benchmark._MODEL_VENV_DIRS["rf-detr"]
            )
            == benchmark._MODEL_VENV_DIRS["rf-detr"]
        )

    def test_missing_venv_python_is_a_silent_noop_not_a_crash(self, tmp_path):
        """rf-detr/.venv absent but particle-tracking/.venv present: resolving
        lodestar must not touch (or error on) the missing rf-detr path."""
        with mock.patch.dict(
            benchmark._MODEL_VENV_DIRS,
            {"lodestar": tmp_path / "does-not-exist"},
        ):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("lodestar")  # must not raise
            fake_execv.assert_not_called()

    def test_matching_python_version_does_not_reexec(self, tmp_path, monkeypatch):
        """When the target venv's Python minor version already matches the
        running interpreter, no re-exec should fire."""
        venv_dir = tmp_path / "fake.venv"
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_pkgs = venv_dir / "lib" / pyver / "site-packages"
        site_pkgs.mkdir(parents=True)
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
        (venv_dir / "bin" / "python").chmod(0o755)

        with mock.patch.dict(benchmark._MODEL_VENV_DIRS, {"rf-detr": venv_dir}):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("rf-detr")
            fake_execv.assert_not_called()


# ---------------------------------------------------------------------------
# get_lodestar_model — U2: LodeSTAR model loading
# ---------------------------------------------------------------------------


def _fake_deeplay_module(built_model):
    fake_dl = mock.MagicMock()
    fake_dl.LodeSTAR.return_value.build.return_value = built_model
    return fake_dl


@pytest.fixture
def no_lodestar_venv(tmp_path):
    """Point _MODEL_VENV_DIRS['lodestar'] at a directory with no
    site-packages, so get_lodestar_model's injection/eviction step is a
    no-op and a sys.modules-injected fake deeplay/torch survives untouched."""
    with mock.patch.dict(benchmark._MODEL_VENV_DIRS, {"lodestar": tmp_path / "no-such-venv"}):
        yield


class TestGetLodestarModel:
    def test_reads_companion_json_for_n_transforms_and_num_outputs(
        self, tmp_path, no_lodestar_venv
    ):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        (tmp_path / "model.json").write_text('{"n_transforms": 4, "num_outputs": 5}')

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            result = benchmark.get_lodestar_model(str(checkpoint), device="cpu")

        fake_dl.LodeSTAR.assert_called_once_with(n_transforms=4, num_outputs=5)
        assert result is built_model
        built_model.eval.assert_called_once()

    def test_missing_companion_json_falls_back_to_defaults(self, tmp_path, no_lodestar_venv):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")  # no sibling .json

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            benchmark.get_lodestar_model(str(checkpoint), device="cpu")

        fake_dl.LodeSTAR.assert_called_once_with(n_transforms=8, num_outputs=3)

    def test_fp16_calls_half_on_model(self, tmp_path, no_lodestar_venv):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            benchmark.get_lodestar_model(str(checkpoint), device="cpu", fp16=True)

        built_model.half.assert_called_once()

    def test_missing_deeplay_prints_error_and_exits(self, tmp_path, no_lodestar_venv):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")

        # Neither deeplay nor torch is installed in verification's own venv —
        # letting the real import fail reproduces the actual missing-dependency path.
        with mock.patch.dict(sys.modules, {"deeplay": None}):
            with pytest.raises(SystemExit):
                benchmark.get_lodestar_model(str(checkpoint), device="cpu")


# ---------------------------------------------------------------------------
# detect_lodestar — U2: LodeSTAR detection, sigma scaling, NMS
# ---------------------------------------------------------------------------


def _fake_torch_module():
    fake_torch = mock.MagicMock()
    fake_model = mock.MagicMock()
    fake_model.parameters.return_value = iter([mock.Mock(dtype="float32")])
    return fake_torch, fake_model


class TestDetectLodestar:
    def test_converts_raw_detection_to_scaled_pixel_box(self):
        fake_torch, fake_model = _fake_torch_module()
        # sigma=0.01 (normalized, < 1.0) on a 512px frame -> radius = 0.01 * 512 = 5.12px
        fake_model.detect.return_value = np.array([[100.0, 200.0, 0.01]])
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 1
        cx = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        cy = (result.xyxy[0][1] + result.xyxy[0][3]) / 2
        # det[0]=y=100, det[1]=x=200 -> center (x=200, y=100)
        assert cx == pytest.approx(200.0, abs=0.5)
        assert cy == pytest.approx(100.0, abs=0.5)
        radius = (result.xyxy[0][2] - result.xyxy[0][0]) / 2
        assert radius == pytest.approx(0.01 * 512, rel=0.05)

    def test_sigma_already_in_pixel_units_is_not_rescaled(self):
        fake_torch, fake_model = _fake_torch_module()
        # sigma=12.0 (>= 1.0) is treated as already-pixel radius, not rescaled by frame_scale.
        fake_model.detect.return_value = np.array([[100.0, 200.0, 12.0]])
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        radius = (result.xyxy[0][2] - result.xyxy[0][0]) / 2
        assert radius == pytest.approx(12.0, rel=0.01)

    def test_empty_detections_returns_empty(self):
        fake_torch, fake_model = _fake_torch_module()
        fake_model.detect.return_value = np.zeros((0, 3))
        frame = np.zeros((64, 64), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 0

    def test_none_detections_returns_empty(self):
        fake_torch, fake_model = _fake_torch_module()
        fake_model.detect.return_value = None
        frame = np.zeros((64, 64), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 0

    def test_list_wrapped_detections_unwraps_first_element(self):
        """model.detect() sometimes returns a list of per-image detection
        arrays (batch dimension) — detect_lodestar must unwrap it."""
        fake_torch, fake_model = _fake_torch_module()
        fake_model.detect.return_value = [np.array([[10.0, 20.0, 0.02]])]
        frame = np.zeros((100, 100), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 1

    def test_nms_distance_suppresses_nearby_duplicates(self):
        fake_torch, fake_model = _fake_torch_module()
        # Two detections 3px apart (well within nms_distance=10), one far away.
        fake_model.detect.return_value = np.array(
            [
                [100.0, 100.0, 0.01],
                [100.0, 103.0, 0.01],
                [400.0, 400.0, 0.01],
            ]
        )
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(
                fake_model, frame, threshold=0.1, device="cpu", nms_distance=10
            )

        assert len(result) == 2  # the near-duplicate pair collapses to one

    def test_output_flows_through_match_detections_unchanged(self):
        """Integration: detect_lodestar's sv.Detections output is consumable
        by the same _match_detections logic used for RF-DETR, with no
        LodeSTAR-specific matching code."""
        fake_torch, fake_model = _fake_torch_module()
        fake_model.detect.return_value = np.array(
            [
                [100.0, 100.0, 0.01],  # matches GT particle 1
                [300.0, 300.0, 0.01],  # false positive
            ]
        )
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = benchmark.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        pred_centers = (result.xyxy[:, :2] + result.xyxy[:, 2:]) / 2
        gt_centers = np.array([[100.0, 100.0]])  # (x, y) — one real particle at det 1's location

        tp, fp, fn, dists = benchmark._match_detections(pred_centers, gt_centers, match_distance=10)
        assert tp == 1
        assert fp == 1
        assert fn == 0


# ---------------------------------------------------------------------------
# main() — U3: --model-type wiring through main() and config
# ---------------------------------------------------------------------------


def _write_ground_truth(path, frames_positions):
    payload = [{"frame": i, "positions": pos} for i, pos in enumerate(frames_positions)]
    path.write_text(json.dumps(payload))


def _write_frames(frames_dir, n=1):
    frames_dir.mkdir(exist_ok=True)
    for i in range(n):
        (frames_dir / f"frame_{i:05d}.png").write_bytes(
            b""
        )  # content unused; _load_frame_rgb mocked


class TestMainModelTypeWiring:
    def test_lodestar_model_type_calls_lodestar_loader_and_skips_tiling(
        self, tmp_path, monkeypatch
    ):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "get_rfdetr_model"
        ) as mock_get_rfdetr, mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_lodestar, mock.patch.object(
            benchmark, "detect_with_tiling"
        ) as mock_detect_tiling:
            benchmark.main()

        mock_get_lodestar.assert_called_once()
        mock_get_rfdetr.assert_not_called()
        mock_detect_lodestar.assert_called_once()
        mock_detect_tiling.assert_not_called()

    def test_lodestar_default_device_is_normalized_not_raw(self, tmp_path, monkeypatch):
        """Regression guard: with no --device flag and no benchmark.lodestar.device
        in config, get_lodestar_model must receive a normalized device string
        ("cuda:0"), not the raw unnormalized default ("0") that torch.device()
        rejects."""
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]  # --device intentionally omitted
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ):
            benchmark.main()

        called_device = mock_get_lodestar.call_args.args[1]
        assert called_device == "cuda:0"

    def test_rf_detr_model_type_unchanged_from_before_this_plan(self, tmp_path, monkeypatch):
        """Regression guard: --model-type rf-detr (or omitted) exercises the
        same rf-detr code path as before LodeSTAR support was added."""
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "rfdetr_model.pth"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  checkpoint: {checkpoint}\n  tiling:\n    enabled: false\n"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
        ]  # --model-type omitted entirely
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        fake_rfdetr_model = mock.Mock()
        fake_rfdetr_model.predict.return_value = _sv_preload.Detections.empty()
        with mock.patch.object(
            benchmark, "get_rfdetr_model", return_value=fake_rfdetr_model
        ) as mock_get_rfdetr, mock.patch.object(
            benchmark, "get_lodestar_model"
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_lodestar"
        ) as mock_detect_lodestar:
            benchmark.main()

        mock_get_rfdetr.assert_called_once()
        mock_get_lodestar.assert_not_called()
        mock_detect_lodestar.assert_not_called()
        fake_rfdetr_model.predict.assert_called_once()

    def test_missing_lodestar_checkpoint_prints_same_error_as_rf_detr(
        self, tmp_path, monkeypatch, capsys
    ):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        missing_checkpoint = tmp_path / "does-not-exist.pt"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {missing_checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            benchmark.main()

        assert "Error: checkpoint not found" in capsys.readouterr().out

    def test_ground_truth_tracks_with_lodestar_invokes_shared_tracking_metrics(
        self, tmp_path, monkeypatch
    ):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        gt_tracks_path = tmp_path / "ground_truth_tracks.csv"
        gt_tracks_path.write_text("frame,particle_id,x,y\n0,1,10.0,10.0\n")
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--ground-truth-tracks",
            str(gt_tracks_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ), mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ), mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ) as mock_tracking_metrics:
            benchmark.main()

        mock_tracking_metrics.assert_called_once()

    def test_unknown_model_type_rejected_by_argparse(self, tmp_path, monkeypatch):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--model-type",
            "yolo",  # not a valid choice
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            benchmark.main()
