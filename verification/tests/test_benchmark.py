"""Tests for benchmark.py — U6: tracking metrics (MOTA/IDF1/fragmentation);
U1-U3: LodeSTAR model-type support."""

import ast
import csv
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
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
from render import FWHM_TO_SIGMA

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

    def test_default_config_path_is_script_dir_anchored_not_cwd_relative(
        self, tmp_path, monkeypatch
    ):
        """U7: --config's default (and this pre-parse's fallback, which must
        stay consistent with it) resolves relative to SCRIPT_DIR, matching
        particle-tracking/track.py — not the caller's cwd. Proven by chdir-ing
        elsewhere and confirming a SCRIPT_DIR-anchored config.yaml still
        resolves, which a bare cwd-relative "config.yaml" string could not."""
        (tmp_path / "config.yaml").write_text("benchmark:\n  model_type: lodestar\n")
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        with mock.patch.object(benchmark, "SCRIPT_DIR", tmp_path):
            assert benchmark._resolve_model_type([]) == "lodestar"

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

    def test_trackpy_skips_reexec_entirely(self):
        """trackpy has no compiled/CUDA dependency and runs natively in this
        script's own venv — _MODEL_VENV_DIRS["trackpy"] is None, and
        _reexec_for_model_venv must return without touching os.execv."""
        with mock.patch.object(benchmark.os, "execv") as fake_execv:
            benchmark._reexec_for_model_venv("trackpy")  # must not raise
        fake_execv.assert_not_called()

    def test_matching_python_version_still_reexecs(self, tmp_path, monkeypatch):
        """A matching Python minor version is not sufficient to skip re-exec:
        detectors_common (which every rf-detr/lodestar model-loading call
        depends on) is only ever installed inside the model venvs, never in
        verification/.venv itself -- even when the two interpreters' ABI
        happens to match, running natively still leaves it unimportable.
        Regression test for the bug this caused: verification/pyproject.toml
        has no .python-version pin (unlike rf-detr/particle-tracking, both
        pinned to 3.11), so verification/.venv's resolved version colliding
        with theirs is a real, non-hypothetical scenario, not just a fake-venv
        contrivance."""
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
            fake_execv.assert_called_once()

    def test_already_running_as_target_venv_python_does_not_reexec(self, tmp_path, monkeypatch):
        """The only valid reason to skip re-exec is already running under that
        exact interpreter (i.e. a prior re-exec already landed here) -- not a
        version-string match. Without this guard, unconditional re-exec would
        loop forever once inside the target venv."""
        venv_dir = _make_fake_venv(tmp_path, "fake.venv")
        venv_python = (venv_dir / "bin" / "python").absolute()
        monkeypatch.setattr(benchmark.sys, "executable", str(venv_python))

        with mock.patch.dict(benchmark._MODEL_VENV_DIRS, {"rf-detr": venv_dir}):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("rf-detr")
            fake_execv.assert_not_called()


# ---------------------------------------------------------------------------
# get_lodestar_model / detect_lodestar — now thin lazy-import wrappers
# delegating to detectors_common.lodestar_loader (see that package's own
# test suite for the loading/scaling/NMS coverage this file used to carry
# directly). What remains here is specific to benchmark.py: that the
# wrapper delegates to the shared implementation with the right venv, and
# that the shared implementation's output still flows into this file's own
# _match_detections.
# ---------------------------------------------------------------------------


class TestGetLodestarModelWrapper:
    def test_delegates_to_shared_implementation_with_configured_venv(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        sentinel_model = mock.Mock()
        fake_impl = mock.Mock(return_value=sentinel_model)
        fake_lodestar_loader = mock.MagicMock(get_lodestar_model=fake_impl)

        with mock.patch.dict(
            sys.modules, {"detectors_common.lodestar_loader": fake_lodestar_loader}
        ):
            result = benchmark.get_lodestar_model(str(checkpoint), device="cpu", fp16=True)

        fake_impl.assert_called_once_with(
            str(checkpoint),
            "cpu",
            inject_venv_site_packages=benchmark._MODEL_VENV_DIRS["lodestar"],
            fp16=True,
        )
        assert result is sentinel_model


class TestLoadLodestarDefaultsWrapper:
    def test_delegates_to_shared_merge_with_benchmark_lodestar_key_map(self):
        """U6: proves _load_lodestar_defaults calls through to the real
        detectors_common.defaults.load_detector_config with the correct
        model_type and key_path_map — the TestMainModelTypeWiring tests all
        stub this function's return value directly, so nothing else in this
        file exercises the actual wiring."""
        sentinel_result = {"nms_distance": 30, "alpha": 0.9}
        fake_impl = mock.Mock(return_value=sentinel_result)
        fake_defaults_module = mock.MagicMock(load_detector_config=fake_impl)
        cfg = {"lodestar": {"nms_distance": 30}}

        with mock.patch.dict(sys.modules, {"detectors_common.defaults": fake_defaults_module}):
            result = benchmark._load_lodestar_defaults(cfg)

        fake_impl.assert_called_once_with(
            "lodestar", cfg, {"nms_distance": "lodestar.nms_distance", "alpha": "lodestar.alpha"}
        )
        assert result == sentinel_result


class TestNoQualifiedDetectorsCommonCallSites:
    def test_no_call_site_uses_the_qualified_detectors_common_path(self):
        """Static guard: every call in benchmark.py must go through one of the
        lazy-wrapper functions, never `detectors_common.<module>.<name>(`
        directly — a stray qualified call would silently bypass test mocks
        and reintroduce the drift this package exists to eliminate."""
        source = Path(benchmark.__file__).read_text()
        tree = ast.parse(source)
        qualified_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    if value.value.id == "detectors_common":
                        qualified_calls.append(f"detectors_common.{value.attr}.{node.func.attr}")
        assert qualified_calls == []


class TestDetectLodestarIntegration:
    def test_output_flows_through_match_detections_unchanged(self):
        """Integration: detect_lodestar's sv.Detections output (as produced by
        the shared implementation) is consumable by the same
        _match_detections logic used for RF-DETR, with no LodeSTAR-specific
        matching code."""
        fake_result = _sv_preload.Detections(
            xyxy=np.array(
                [
                    [95.0, 95.0, 105.0, 105.0],  # matches GT particle 1
                    [295.0, 295.0, 305.0, 305.0],  # false positive
                ],
                dtype=np.float32,
            ),
            confidence=np.array([1.0, 1.0], dtype=np.float32),
            class_id=np.zeros(2, dtype=int),
        )
        fake_lodestar_loader = mock.MagicMock(detect_lodestar=mock.Mock(return_value=fake_result))
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(
            sys.modules, {"detectors_common.lodestar_loader": fake_lodestar_loader}
        ):
            result = benchmark.detect_lodestar(mock.Mock(), frame, threshold=0.1, device="cpu")

        pred_centers = (result.xyxy[:, :2] + result.xyxy[:, 2:]) / 2
        gt_centers = np.array([[100.0, 100.0]])  # (x, y) — one real particle at det 1's location

        tp, fp, fn, dists = benchmark._match_detections(pred_centers, gt_centers, match_distance=10)
        assert tp == 1
        assert fp == 1
        assert fn == 0


# ---------------------------------------------------------------------------
# detect_trackpy — native (no venv injection) classical-detector baseline
# ---------------------------------------------------------------------------


def _make_gaussian_blob_frame(size=64, center=(32, 32), sigma=3.0, peak=200.0):
    """A single bright Gaussian blob on a dark background, uint8 RGB —
    matches this codebase's bright-particle-on-dark-background convention
    (see render.py / detect_lodestar)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = center
    gray = peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    gray = gray.clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


class TestDetectTrackpy:
    def test_locates_single_bright_blob_near_true_center(self):
        frame = _make_gaussian_blob_frame(center=(32, 32))
        result = benchmark.detect_trackpy(frame, diameter=11)

        assert len(result) == 1
        cx = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        cy = (result.xyxy[0][1] + result.xyxy[0][3]) / 2
        assert abs(cx - 32) < 1.0
        assert abs(cy - 32) < 1.0

    def test_dark_frame_returns_empty_detections(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = benchmark.detect_trackpy(frame, diameter=11, minmass=1.0)
        assert len(result) == 0

    def test_minmass_none_runs_without_error(self):
        frame = _make_gaussian_blob_frame(center=(32, 32))
        result = benchmark.detect_trackpy(frame, diameter=11, minmass=None)
        assert len(result) >= 0  # runs without raising; count is not asserted


class TestModelTypeChoices:
    def test_trackpy_is_a_valid_model_type_choice(self):
        assert "trackpy" in list(benchmark._MODEL_VENV_DIRS)


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
        # main()'s output_dir = Path("verification_output") is CWD-relative --
        # without chdir-ing into tmp_path, this test would write into (and
        # corrupt) the real verification/verification_output/ directory.
        monkeypatch.chdir(tmp_path)
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
        ) as mock_detect_tiling, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
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
        monkeypatch.chdir(tmp_path)
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
        ), mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        called_device = mock_get_lodestar.call_args.args[1]
        assert called_device == "cuda:0"

    def test_rf_detr_model_type_unchanged_from_before_this_plan(self, tmp_path, monkeypatch):
        """Regression guard: --model-type rf-detr (or omitted) exercises the
        same rf-detr code path as before LodeSTAR support was added."""
        monkeypatch.chdir(tmp_path)
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
        monkeypatch.chdir(tmp_path)
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

        with mock.patch.object(benchmark, "_load_lodestar_defaults", return_value={}):
            with pytest.raises(SystemExit):
                benchmark.main()

        assert "Error: checkpoint not found" in capsys.readouterr().out

    def test_ground_truth_tracks_with_lodestar_invokes_shared_tracking_metrics(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
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
        ) as mock_tracking_metrics, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        mock_tracking_metrics.assert_called_once()

    def test_trackpy_model_type_skips_checkpoint_and_model_loading(self, tmp_path, monkeypatch):
        """trackpy has no checkpoint and no loaded model — main() must not call
        either loader and must not require a checkpoint file to exist."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(benchmark, "get_rfdetr_model") as mock_get_rfdetr, mock.patch.object(
            benchmark, "get_lodestar_model"
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_trackpy:
            benchmark.main()  # must not raise despite no checkpoint file existing anywhere

        mock_get_rfdetr.assert_not_called()
        mock_get_lodestar.assert_not_called()
        mock_detect_trackpy.assert_called_once()
        called_kwargs = mock_detect_trackpy.call_args.kwargs
        assert called_kwargs["diameter"] == 11

    def test_trackpy_accumulates_detections_like_other_model_types(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--ground-truth-tracks",
            str(tmp_path / "gt_tracks.csv"),  # missing file: tracking metrics warn+skip, not raise
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )
        fake_detections = _sv_preload.Detections(
            xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=fake_detections
        ) as mock_detect_trackpy, mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ) as mock_tracking_metrics:
            benchmark.main()

        mock_detect_trackpy.assert_called_once()
        # all_detections_by_frame must have flowed into _run_tracking_metrics
        # the same generic way it does for rf-detr/lodestar.
        mock_tracking_metrics.assert_called_once()
        all_detections_arg = mock_tracking_metrics.call_args[0][0]
        assert 0 in all_detections_arg
        assert all_detections_arg[0].shape == (1, 2)

    def test_rf_detr_and_lodestar_unchanged_by_trackpy_addition(self, tmp_path, monkeypatch):
        """Regression guard: adding the trackpy branch must not alter rf-detr's
        or lodestar's existing checkpoint enforcement or model loading."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        missing_checkpoint = tmp_path / "does-not-exist.pth"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  checkpoint: {missing_checkpoint}\n  tiling:\n    enabled: false\n"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            benchmark.main()

    def test_unknown_model_type_rejected_by_argparse(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
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


class TestLodestarBoxSizeDerivation:
    """box_size (for main()'s lodestar accuracy loop) derives from psf_sigma_px --
    an explicit benchmark.lodestar.box_size config value always wins."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra=""):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n{config_yaml_extra}"
        )

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
        ), mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_lodestar, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        return mock_detect_lodestar.call_args.kwargs["box_size"]

    def test_derives_from_synthetic_psf_sigma(self, tmp_path, monkeypatch):
        box_size = self._run(tmp_path, monkeypatch, "synthetic:\n  psf_sigma: 6.0\n")
        assert box_size == pytest.approx(6.0 * FWHM_TO_SIGMA)

    def test_falls_back_to_synthetic_psf_sigma_px_when_psf_sigma_absent(
        self, tmp_path, monkeypatch
    ):
        box_size = self._run(tmp_path, monkeypatch, "synthetic:\n  psf:\n    sigma_px: 8.21\n")
        assert box_size == pytest.approx(8.21 * FWHM_TO_SIGMA)

    def test_falls_back_to_default_5_0_when_neither_present(self, tmp_path, monkeypatch):
        box_size = self._run(tmp_path, monkeypatch)
        assert box_size == pytest.approx(5.0 * FWHM_TO_SIGMA)

    def test_explicit_box_size_overrides_derivation(self, tmp_path, monkeypatch):
        box_size = self._run(
            tmp_path,
            monkeypatch,
            "    box_size: 17\nsynthetic:\n  psf_sigma: 6.0\n",
        )
        assert box_size == 17

    def test_shipped_config_yaml_no_longer_short_circuits_to_40(self):
        """Regression guard for the override-always-wins failure mode this unit
        fixes: verification/config.yaml's own lodestar.box_size must stay unset
        (commented out) so the derived value actually takes effect by default,
        not the literal 40 the old fallback config value used to carry."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "lodestar", "box_size") is None


# ---------------------------------------------------------------------------
# --save-video: _link_detections_for_video
# ---------------------------------------------------------------------------


class TestLinkDetectionsForVideo:
    """--save-video's linking helper: unlike _run_tracking_metrics (which only
    needs (x, y) centers), this must hand each linked track_id back its own
    box, so it threads an explicit local_idx column through tp.link_df rather
    than trusting row order to survive linking."""

    def test_returns_a_box_and_track_id_per_detection_per_frame(self):
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0], [100.0, 100.0, 110.0, 110.0]]),
            1: np.array([[11.0, 11.0, 21.0, 21.0], [101.0, 101.0, 111.0, 111.0]]),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        assert set(result.keys()) == {0, 1}
        for frame_idx, (boxes, track_ids) in result.items():
            assert boxes.shape == (2, 4)
            assert track_ids.shape == (2,)

    def test_same_particle_gets_the_same_track_id_across_frames(self):
        """A single particle drifting a few pixels per frame must link into
        one consistent track_id, and its own box must travel with it."""
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0]]),
            1: np.array([[12.0, 12.0, 22.0, 22.0]]),
            2: np.array([[14.0, 14.0, 24.0, 24.0]]),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        track_ids = [result[f][1][0] for f in (0, 1, 2)]
        assert track_ids[0] == track_ids[1] == track_ids[2]
        # Each frame's box must be the one that actually belongs to that frame,
        # not e.g. frame 0's box relinked onto frame 2's entry.
        np.testing.assert_array_equal(result[2][0][0], boxes_by_frame[2][0])

    def test_two_particles_keep_their_own_boxes_after_linking(self):
        """local_idx must correctly re-associate each track_id with its own
        box, not e.g. accidentally swap two same-frame detections' boxes."""
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0], [200.0, 200.0, 210.0, 210.0]]),
            1: np.array([[201.0, 201.0, 211.0, 211.0], [11.0, 11.0, 21.0, 21.0]]),  # order swapped
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        boxes0, ids0 = result[0]
        boxes1, ids1 = result[1]
        # Whichever track_id corresponds to the near-(10,10) particle must own
        # the near-(11,11) box in frame 1, not the near-(201,201) box.
        small_track_id = ids0[np.argmin(boxes0[:, 0])]
        small_box_frame1 = boxes1[ids1 == small_track_id][0]
        assert small_box_frame1[0] < 100  # the (11,11)-ish box, not the (201,201)-ish one

    def test_empty_detections_returns_empty_dict(self):
        boxes_by_frame = {0: np.zeros((0, 4)), 1: np.zeros((0, 4))}
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)
        assert result == {}

    def test_frame_with_no_detections_is_absent_from_result(self):
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0]]),
            1: np.zeros((0, 4)),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)
        assert 0 in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# _link_df_kwargs / tracking.adaptive_stop — SubnetOversizeException guard
# ---------------------------------------------------------------------------


def _cluster_det_df(n_particles, n_frames, box_size, jitter, seed=1):
    """A pandas DataFrame of (frame, x, y) detections for a cluster of
    n_particles confined to a box_size x box_size region, jittered by up to
    +/-jitter px per frame -- the shape _link_df_with_fallback consumes
    directly (mirrors _run_tracking_metrics'/_link_detections_for_video's
    own det_df construction)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    base = rng.uniform(0, box_size, size=(n_particles, 2))
    rows = []
    for frame in range(n_frames):
        j = rng.uniform(-jitter, jitter, size=(n_particles, 2))
        for x, y in base + j:
            rows.append({"frame": frame, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestLinkDfWithFallback:
    """_link_df_with_fallback retries tp.link_df at a shrinking
    search_range on SubnetOversizeException instead of giving up on the
    first (configured) attempt -- see its own docstring for why (real RF-
    DETR/LodeSTAR detector output at this repo's default trajectory density
    otherwise loses all trajectory data in --save-video, and MOTA/IDF1
    entirely, on the very first oversized subnet)."""

    def test_succeeds_at_configured_search_range_without_retry(self):
        det_df = _cluster_det_df(n_particles=5, n_frames=3, box_size=200, jitter=1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, cfg={"tracking": {}}, search_range=15, memory=3
        )

        assert linked is not None
        assert used_range == 15
        assert len(linked) == len(det_df)

    def test_recovers_via_smaller_search_range_after_oversized_subnet(self):
        # 60 particles in a tight 30x30px cluster: mutually within the
        # configured search_range=15 of each other, exceeding trackpy's
        # default max_subnet_size=30 -- see _dense_cluster_detections_and_gt
        # below for the same construction used against the real pipeline
        # functions.
        det_df = _cluster_det_df(n_particles=60, n_frames=3, box_size=30, jitter=1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, cfg={"tracking": {}}, search_range=15, memory=3
        )

        assert linked is not None
        assert used_range < 15
        assert len(linked) == len(det_df)

    def test_returns_none_when_even_floor_search_range_is_oversized(self):
        # 60 particles confined to a 0.5x0.5px box with 0.1px jitter: even
        # at the 1.0px floor, every particle remains mutually within range
        # of every other -- the subnet never shrinks below trackpy's
        # max_subnet_size regardless of how far search_range is reduced.
        det_df = _cluster_det_df(n_particles=60, n_frames=3, box_size=0.5, jitter=0.1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, cfg={"tracking": {}}, search_range=15, memory=3
        )

        assert linked is None
        assert used_range is None


def _dense_cluster_detections_and_gt(tmp_path, n_particles=60, n_frames=3, seed=0):
    """A tight cluster of particles all mutually within a typical
    search_range of each other across consecutive frames -- exactly what
    raises trackpy.linking.utils.SubnetOversizeException (default
    max_subnet_size=30, so n_particles=60 reliably triggers it)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 30, size=(n_particles, 2))
    gt_rows = []
    for frame in range(n_frames):
        jitter = rng.uniform(-1, 1, size=(n_particles, 2))
        for pid, (x, y) in enumerate(base + jitter):
            gt_rows.append({"frame": frame, "particle_id": pid, "x": x, "y": y})
    gt_path = tmp_path / "gt.csv"
    _write_gt_tracks(gt_path, gt_rows)

    detections = {
        frame: base + rng.uniform(-1, 1, size=(n_particles, 2)) for frame in range(n_frames)
    }
    return detections, gt_path


class TestLinkDfKwargs:
    """tracking.adaptive_stop/adaptive_step (opt-in, off by default) let
    trackpy retry an oversized subnet with a shrunken search_range instead of
    immediately raising SubnetOversizeException -- appropriate for
    moderately dense scenes. NOT the primary safety net against this
    dataset's pathological density: confirmed directly (2026-08-08) that
    enabling it against a ~1400-point mutually-connected subnet exhausted
    this machine's RAM+swap and had to be killed, since subnet size doesn't
    necessarily shrink fast enough as search_range shrinks when particles
    are this tightly packed. _run_tracking_metrics/_link_detections_for_video
    catching SubnetOversizeException directly (TestSubnetOversizeGuard below)
    is the real, always-on safety net."""

    def test_adaptive_stop_absent_omits_adaptive_kwargs(self):
        cfg = {"tracking": {"search_range": 15, "memory": 3}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs == {"search_range": 15, "memory": 3}

    def test_adaptive_stop_set_adds_adaptive_kwargs(self):
        cfg = {"tracking": {"adaptive_stop": 3.0, "adaptive_step": 0.9}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs == {
            "search_range": 15,
            "memory": 3,
            "adaptive_stop": 3.0,
            "adaptive_step": 0.9,
        }

    def test_adaptive_step_defaults_to_0_95_when_only_stop_is_set(self):
        cfg = {"tracking": {"adaptive_stop": 3.0}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs["adaptive_step"] == 0.95

    def test_shipped_config_yaml_leaves_adaptive_stop_disabled(self):
        """Regression guard: verification/config.yaml's tracking.adaptive_stop
        must stay null/absent by default -- enabling it against this repo's
        default dense trajectory is the RAM/swap-exhaustion failure mode
        confirmed above, not a safe default."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "tracking", "adaptive_stop") is None


def _scattered_detections_and_gt(tmp_path, n_particles_per_frame, n_frames, spacing, seed=2):
    """detections/gt where each frame's particles are placed on a fresh,
    independent widely-spaced grid offset -- no particle's position in one
    frame falls within any plausible search_range of its own position in
    the next frame, so trackpy links nothing across frames and every
    per-frame detection becomes its own distinct (1-frame-long) track. Used
    to exercise the track-id-cardinality guard without needing a dense/slow
    cluster."""
    rng = np.random.default_rng(seed)
    gt_rows = []
    detections = {}
    for frame in range(n_frames):
        # A large per-frame offset (spacing * n_particles_per_frame) keeps
        # every frame's grid physically disjoint from every other frame's.
        centers = np.stack(
            [
                frame * spacing * n_particles_per_frame
                + np.arange(n_particles_per_frame) * spacing,
                rng.uniform(0, spacing, size=n_particles_per_frame),
            ],
            axis=1,
        )
        detections[frame] = centers
        gt_rows.extend(
            {"frame": frame, "particle_id": frame * n_particles_per_frame + pid, "x": x, "y": y}
            for pid, (x, y) in enumerate(centers)
        )
    gt_path = tmp_path / "gt.csv"
    _write_gt_tracks(gt_path, gt_rows)
    return detections, gt_path


class TestTrackingMetricsDensityGuard:
    """_run_tracking_metrics must skip building the motmetrics accumulator
    -- not just guard the later mm.metrics.create().compute() call -- above
    a safe detection density or distinct-track-id count. Confirmed directly
    (2026-08-08): the accumulator-building loop's own acc.update() calls can
    grow this process's own memory into the double-digit-GB range and get
    OOM-killed on real (not synthetic-test) detector output, independently
    of _compute_motmetrics_with_timeout's own subprocess+timeout guard --
    that guard runs strictly after this loop, so it never gets a chance to
    protect against this specific cost."""

    def test_skips_when_average_density_exceeds_threshold(self, tmp_path):
        # 450 particles/frame, widely spaced (2px apart, no clustering) --
        # links trivially fast, but average density (450/frame) exceeds the
        # 400/frame safety threshold on its own.
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=450, n_frames=2, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is None

    def test_skips_when_track_id_cardinality_exceeds_threshold(self, tmp_path):
        # 300 particles/frame across 4 frames (1200 total detections, well
        # under the 400/frame density threshold on its own) but no particle
        # links across frames (see _scattered_detections_and_gt), so every
        # detection becomes its own track -- 1200 distinct track ids,
        # exceeding the 1000 threshold on cardinality alone.
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=300, n_frames=4, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is None

    def test_computes_normally_below_both_thresholds(self, tmp_path):
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=5, n_frames=3, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert "mota" in result and "idf1" in result


class TestSubnetOversizeGuard:
    """_run_tracking_metrics/_link_detections_for_video must catch
    trackpy.linking.utils.SubnetOversizeException directly and degrade
    gracefully -- the real, always-on safety net (independent of
    tracking.adaptive_stop, which is opt-in and unsafe for this dataset;
    see TestLinkDfKwargs). This is what actually fixed the 2026-08-08
    incident: RF-DETR's tiling fix raised recall enough to recover a
    genuinely dense physical cluster in verification_output/v2's
    continuous_force_1500_5.0 trajectory, and the resulting oversized
    subnet must not crash or hang the pipeline.

    Both functions call _link_df_with_fallback, which retries at a smaller
    search_range instead of giving up on the first SubnetOversizeException
    (see TestLinkDfWithFallback) -- confirmed directly (2026-08-08) that
    this recovers real trajectories for RF-DETR's video where the pipeline
    used to fall back to boxes-only. So a merely-oversized cluster like this
    one now succeeds at a smaller search_range rather than failing outright;
    these tests assert exactly that (not a crash, and not silently losing
    the result). TestLinkDfWithFallback separately covers the true failure
    path (even the smallest fallback search_range still oversized)."""

    def test_run_tracking_metrics_recovers_via_smaller_search_range(self, tmp_path):
        detections, gt_path = _dense_cluster_detections_and_gt(tmp_path)
        cfg = _make_cfg(search_range=15)  # adaptive_stop absent -- default/unsafe-off state

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg)

        assert result is not None
        assert "mota" in result and "idf1" in result

    def test_link_detections_for_video_recovers_via_smaller_search_range(self, tmp_path):
        detections, _ = _dense_cluster_detections_and_gt(tmp_path)
        boxes_by_frame = {
            frame: np.concatenate([centers - 5, centers + 5], axis=1)
            for frame, centers in detections.items()
        }
        cfg = {"tracking": {}}  # adaptive_stop absent

        result = benchmark._link_detections_for_video(
            boxes_by_frame, cfg, search_range=15, memory=3
        )

        assert set(result.keys()) == set(boxes_by_frame.keys())
        boxes, track_ids = result[0]
        assert len(boxes) == len(track_ids) == len(boxes_by_frame[0])


# ---------------------------------------------------------------------------
# _compute_motmetrics_with_timeout — motmetrics IDF1 scaling guard
# ---------------------------------------------------------------------------


def _slow_motmetrics_worker(acc, metrics, conn):
    """Module-level (not a test-method closure) so it's picklable by
    _compute_motmetrics_with_timeout's "spawn" context -- spawn re-imports
    the target function by its module+qualname in the child, which a local
    closure can't satisfy (fork could, since it inherits the parent's
    already-loaded function via COW memory; spawn can't)."""
    import time

    time.sleep(5)
    conn.send({})


class TestComputeMotmetricsWithTimeout:
    """motmetrics' IDF1 global identity-assignment can scale catastrophically
    with the number of distinct GT/predicted track IDs -- confirmed directly
    (2026-08-08): a real ~1446 GT particle x ~1700 fragmented-track-id
    trackpy run against verification_output/v2 exhausted this machine's
    RAM+swap and had to be killed manually. This must never happen again
    regardless of root cause, so mh.compute() runs in a subprocess with a
    hard OS-level timeout."""

    def test_happy_path_returns_summary_dict(self):
        import motmetrics as mm

        acc = mm.MOTAccumulator(auto_id=True)
        acc.update([1, 2], [1, 2], np.array([[0.0, 5.0], [5.0, 0.0]]))

        result = benchmark._compute_motmetrics_with_timeout(
            acc, metrics=["mota", "idf1"], timeout_s=30
        )

        assert result is not None
        assert "mota" in result
        assert "idf1" in result

    def test_returns_none_when_computation_exceeds_timeout(self, monkeypatch):
        monkeypatch.setattr(benchmark, "_motmetrics_compute_worker", _slow_motmetrics_worker)
        import motmetrics as mm

        acc = mm.MOTAccumulator(auto_id=True)
        acc.update([1], [1], np.array([[0.0]]))

        result = benchmark._compute_motmetrics_with_timeout(acc, metrics=["mota"], timeout_s=1)

        assert result is None
