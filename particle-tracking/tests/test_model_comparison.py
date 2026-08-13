import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Set non-interactive backend before pyplot is imported anywhere
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_comparison
from model_comparison import (
    ModelSpec,
    _build_rfdetr_script,
    _read_tuning,
    _write_model_config,
    build_arg_parser,
    build_comparison_figure,
    default_device,
    parse_crop,
    parse_model_spec,
    run_full_comparison,
)


class TestParseModelSpec:
    def test_valid_rfdetr(self):
        spec = parse_model_spec("rf-detr:checkpoints/best.pth")
        assert spec.model_type == "rf-detr"
        assert spec.checkpoint == Path("checkpoints/best.pth")

    def test_valid_yolo(self):
        spec = parse_model_spec("yolo:weights/best.pt")
        assert spec.model_type == "yolo"
        assert spec.checkpoint == Path("weights/best.pt")

    def test_valid_lodestar(self):
        spec = parse_model_spec("lodestar:models/lodestar.pth")
        assert spec.model_type == "lodestar"
        assert spec.checkpoint == Path("models/lodestar.pth")

    def test_unknown_type_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown model type"):
            parse_model_spec("fasterrcnn:weights/best.pt")

    def test_missing_separator_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid model spec"):
            parse_model_spec("rf-detr-only")

    def test_checkpoint_path_with_colon(self):
        # split(":", 1) keeps everything after first colon intact
        spec = parse_model_spec("yolo:C:\\weights\\best.pt")
        assert spec.checkpoint == Path("C:\\weights\\best.pt")


class TestDefaultDevice:
    def test_returns_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert default_device() == "cuda:0"

    def test_returns_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert default_device() == "cpu"

    def test_returns_cpu_when_torch_missing(self):
        original = sys.modules.pop("torch", None)
        try:
            with patch.dict(sys.modules, {"torch": None}):
                result = default_device()
        finally:
            if original is not None:
                sys.modules["torch"] = original
        assert result == "cpu"


class TestBuildComparisonFigure:
    def _empty_detections(self):
        d = MagicMock()
        d.xyxy = np.empty((0, 4), dtype=np.float32)
        d.__len__ = lambda self: 0
        return d

    def test_original_plus_two_models(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [
            ("rf-detr — 5 detections", self._empty_detections()),
            ("yolo — 7 detections", self._empty_detections()),
        ]
        fig = build_comparison_figure(frame, results)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_original_plus_one_model(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("rf-detr — 3 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_first_panel_title_is_original(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("rf-detr — 0 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert fig.axes[0].get_title() == "Original"
        plt.close(fig)

    def test_model_panel_title_is_set(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("yolo — 12 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert fig.axes[1].get_title() == "yolo — 12 detections"
        plt.close(fig)

    def test_draws_boxes_without_error(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_det = MagicMock()
        mock_det.xyxy = np.array([[10, 20, 50, 60], [80, 90, 120, 130]], dtype=np.float32)
        mock_det.confidence = np.array([0.9, 0.75], dtype=np.float32)
        mock_det.__len__ = lambda self: 2

        results = [("rf-detr — 2 detections", mock_det)]
        fig = build_comparison_figure(frame, results)
        # Just verify no exception was raised and boxes were added
        ax = fig.axes[1]
        assert len(ax.patches) == 2
        plt.close(fig)


class TestRunDetectionLodestarBoxSize:
    """run_detection's lodestar branch must thread box_size through to
    detect_lodestar -- otherwise --image mode silently ignores
    --lodestar-box-size and always draws detectors_common's hardcoded
    40px default, inconsistent with track.py's configured value."""

    def test_lodestar_box_size_passed_through(self):
        fake_helpers = MagicMock()
        fake_helpers.detect_lodestar.return_value = MagicMock()
        model = MagicMock()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(model_comparison, "_load_track_helpers", return_value=fake_helpers):
            model_comparison.run_detection(
                model, "lodestar", frame, 0.1, "cpu", lodestar_box_size=17
            )

        fake_helpers.detect_lodestar.assert_called_once_with(model, frame, 0.1, "cpu", box_size=17)

    def test_lodestar_box_size_defaults_to_40(self):
        fake_helpers = MagicMock()
        fake_helpers.detect_lodestar.return_value = MagicMock()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(model_comparison, "_load_track_helpers", return_value=fake_helpers):
            model_comparison.run_detection(MagicMock(), "lodestar", frame, 0.1, "cpu")

        assert fake_helpers.detect_lodestar.call_args.kwargs["box_size"] == 40


class TestBuildRfdetrScript:
    def test_build_rfdetr_script_includes_num_queries_when_set(self):
        script = _build_rfdetr_script("RFDETRLarge", Path("ckpt.pth"), "/tmp/frame.npy", 0.5, 6000)
        assert "num_queries=6000" in script

    def test_build_rfdetr_script_omits_num_queries_when_none(self):
        script = _build_rfdetr_script("RFDETRLarge", Path("ckpt.pth"), "/tmp/frame.npy", 0.5, None)
        assert "num_queries" not in script

    def test_build_rfdetr_script_uses_correct_class_and_checkpoint(self):
        script = _build_rfdetr_script(
            "RFDETRBase", Path("weights/best.pth"), "/tmp/f.npy", 0.25, None
        )
        assert "RFDETRBase(pretrain_weights='weights/best.pth')" in script


# ────────────────────────────────────────────────────────────
# Full-run comparison mode (--input)
# ────────────────────────────────────────────────────────────


def _fake_rfdetr_writer(
    name,
    input_path,
    output_dir,
    crop_w,
    crop_h,
    bridge_gap,
    script_dir,
    checkpoint=None,
    dataset_profile=None,
):
    """Stand-in for tracker_configs.write_rfdetr_config that avoids touching the real
    particle-tracking/run_configs/ directory and mirrors its real stub_filter/search_range
    defaults (90 / 25)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(output_dir) / f"{name}.yaml"
    checkpoint_line = f'\nmodel:\n  checkpoint: "{checkpoint}"' if checkpoint else ""
    profile_line = f'\ndataset_profile: "{dataset_profile}"' if dataset_profile else ""
    cfg_path.write_text(
        f'input: "{input_path}"\ntracking:\n  stub_filter: 90\n  search_range: 25\n'
        f"{checkpoint_line}{profile_line}"
    )
    return cfg_path


def _fake_lodestar_writer(
    name, input_path, output_dir, crop_w, crop_h, bridge_gap, script_dir, dataset_profile=None
):
    """Stand-in for tracker_configs.write_lodestar_config; mirrors its real
    stub_filter/search_range defaults (6 / 20)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(output_dir) / f"{name}.yaml"
    profile_line = f'\ndataset_profile: "{dataset_profile}"' if dataset_profile else ""
    cfg_path.write_text(
        f'input: "{input_path}"\ntracking:\n  stub_filter: 6\n  search_range: 20\n{profile_line}'
    )
    return cfg_path


def _parse_full_run_args(tmp_path, input_path, model_specs, output_dir=None, extra_argv=None):
    parser = build_arg_parser()
    output_dir = output_dir or (tmp_path / "cmp")
    argv = ["--input", str(input_path), "--models", *model_specs, "--output-dir", str(output_dir)]
    if extra_argv:
        argv += extra_argv
    return parser.parse_args(argv), parser


def _mock_popen(returncode, stderr="", pid=12345):
    """Stand-in for a subprocess.Popen instance, matching run_model_tracking's usage
    (proc.communicate(timeout=...) -> (stdout, stderr); proc.returncode; proc.pid)."""
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.communicate.return_value = (None, stderr)
    return proc


class TestParseCrop:
    def test_none_returns_none_none(self):
        parser = build_arg_parser()
        assert parse_crop(None, parser) == (None, None)

    def test_valid_crop_parsed(self):
        parser = build_arg_parser()
        assert parse_crop("1024x768", parser) == (1024, 768)

    def test_missing_x_errors(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parse_crop("1024768", parser)

    def test_non_positive_dimension_errors(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parse_crop("0x768", parser)


class TestReadTuning:
    def test_reads_stub_filter_and_search_range(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("tracking:\n  stub_filter: 42\n  search_range: 11\n")
        assert _read_tuning(cfg) == {"stub_filter": 42, "search_range": 11}

    def test_missing_tracking_section_returns_nones(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("input: 'x.tif'\n")
        assert _read_tuning(cfg) == {"stub_filter": None, "search_range": None}


class TestWriteModelConfig:
    def test_yolo_raises_clear_error(self):
        with pytest.raises(ValueError, match="No config writer available for model type 'yolo'"):
            _write_model_config("yolo", "name", "in.tif", "out", None, None, None)

    def test_rfdetr_dispatches_to_shared_writer(self, tmp_path):
        with patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer):
            cfg = _write_model_config(
                "rf-detr", "cmp_rf-detr", "in.tif", str(tmp_path / "out"), None, None, None
            )
        assert cfg.exists()
        assert _read_tuning(cfg)["stub_filter"] == 90

    def test_lodestar_dispatches_to_shared_writer(self, tmp_path):
        with patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer):
            cfg = _write_model_config(
                "lodestar", "cmp_lodestar", "in.tif", str(tmp_path / "out"), None, None, None
            )
        assert cfg.exists()
        assert _read_tuning(cfg)["stub_filter"] == 6


class TestBuildArgParserMutualExclusion:
    def test_image_and_input_together_is_a_cli_error(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--image", "a.png", "--input", "b.tif", "--models", "rf-detr:ckpt.pth"]
            )

    def test_neither_image_nor_input_is_a_cli_error(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--models", "rf-detr:ckpt.pth"])

    def test_image_alone_still_parses(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--image", "a.png", "--models", "rf-detr:ckpt.pth"])
        assert args.image == "a.png"
        assert args.input is None

    def test_input_alone_still_parses(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--input", "v.tif", "--models", "rf-detr:ckpt.pth"])
        assert args.input == "v.tif"
        assert args.image is None


class TestRunFullComparison:
    def test_all_three_models_run_distinct_subdirs_and_stats(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            [
                "rf-detr:../rf-detr/checkpoints/best.pth",
                "yolo:../yolov12/best.pt",
                "lodestar:../data-setup/models/lodestar_model_15",
            ],
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 5, "track_length_mean": 12.0}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is True  # yolo has no config writer -> counts as a real failure

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["models"]) == 3

        output_dirs = {m["output_dir"] for m in manifest["models"]}
        assert len(output_dirs) == 3  # per-model subdirectory convention prevents collisions

        by_type = {m["model_type"]: m for m in manifest["models"]}
        assert by_type["rf-detr"]["stats"] == {"n_tracks": 5, "track_length_mean": 12.0}
        assert by_type["rf-detr"]["exit_code"] == 0
        assert by_type["lodestar"]["stats"] == {"n_tracks": 5, "track_length_mean": 12.0}
        assert by_type["lodestar"]["exit_code"] == 0

        # yolo has no config writer yet (pre-existing gap) — recorded as a clear failure,
        # not a crash, and does not stop the other two models from completing.
        assert by_type["yolo"]["stats"] is None
        assert by_type["yolo"]["exit_code"] is None
        assert "error" in by_type["yolo"]
        assert "yolo" in by_type["yolo"]["error"]

        # rf-detr and lodestar both actually ran their subprocess.
        assert mock_popen_cls.call_count == 2

    def test_tuning_differs_true_for_rfdetr_vs_lodestar_defaults(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth", "lodestar:ckpt2.pth"],
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is False

        manifest = json.loads(manifest_path.read_text())
        # rf-detr's stub_filter=90 and lodestar's stub_filter=6 both appear in this run.
        assert manifest["tuning_differs"] is True

    def test_tuning_same_model_type_twice_not_flagged_differs(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        # Two rf-detr specs -> only rf-detr's own default tuning appears, so it should
        # not be flagged as differing (single-value set for stub_filter/search_range).
        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth", "rf-detr:ckpt2.pth"],
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is False

        manifest = json.loads(manifest_path.read_text())
        assert manifest["tuning_differs"] is False

    def test_one_model_subprocess_failure_does_not_stop_others(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth", "lodestar:ckpt2.pth"],
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            # rf-detr fails (non-zero exit), lodestar succeeds
            mock_popen_cls.side_effect = [
                _mock_popen(1, stderr="Traceback...CUDA OOM"),
                _mock_popen(0),
            ]
            mock_stats.return_value = {"n_tracks": 3}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is True

        manifest = json.loads(manifest_path.read_text())
        by_type = {m["model_type"]: m for m in manifest["models"]}

        assert by_type["rf-detr"]["exit_code"] == 1
        assert by_type["rf-detr"]["stats"] is None
        assert "error" in by_type["rf-detr"]
        assert "CUDA OOM" in by_type["rf-detr"]["stderr_tail"]

        # lodestar still ran to completion despite rf-detr's failure.
        assert by_type["lodestar"]["exit_code"] == 0
        assert by_type["lodestar"]["stats"] == {"n_tracks": 3}
        assert mock_popen_cls.call_count == 2

    def test_subprocess_invocation_exception_does_not_stop_others(self, tmp_path):
        """run_model_tracking itself raising (e.g. `uv` not found) is distinct from a
        non-zero exit code — both must be recorded without stopping the loop."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path, input_path, ["rf-detr:ckpt.pth", "lodestar:ckpt2.pth"]
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.side_effect = [FileNotFoundError("uv not found"), _mock_popen(0)]
            mock_stats.return_value = {"n_tracks": 2}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is True
        manifest = json.loads(manifest_path.read_text())
        by_type = {m["model_type"]: m for m in manifest["models"]}

        assert "error" in by_type["rf-detr"]
        assert by_type["rf-detr"]["exit_code"] is None

        # lodestar still ran to completion despite rf-detr's invocation failure.
        assert by_type["lodestar"]["exit_code"] == 0
        assert by_type["lodestar"]["stats"] == {"n_tracks": 2}

    def test_stats_failure_does_not_count_as_model_failure(self, tmp_path):
        """A stats-computation bug on an otherwise-successful run is recorded separately
        from a real run failure, and does not force a non-zero exit."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(tmp_path, input_path, ["rf-detr:ckpt.pth"])

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.side_effect = ValueError("corrupt tracks.csv")

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is False

        manifest = json.loads(manifest_path.read_text())
        entry = manifest["models"][0]
        assert entry["exit_code"] == 0
        assert "stats_error" in entry
        assert "error" not in entry

    def test_timeout_kills_process_group_and_does_not_stop_others(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path, input_path, ["rf-detr:ckpt.pth", "lodestar:ckpt2.pth"]
        )

        timed_out_proc = _mock_popen(None)
        timed_out_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="track.py", timeout=args.model_timeout
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("os.getpgid", return_value=999) as mock_getpgid,
            patch("os.killpg") as mock_killpg,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.side_effect = [timed_out_proc, _mock_popen(0)]
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is True
        mock_getpgid.assert_called_once_with(timed_out_proc.pid)
        mock_killpg.assert_called_once_with(999, signal.SIGKILL)

        manifest = json.loads(manifest_path.read_text())
        by_type = {m["model_type"]: m for m in manifest["models"]}
        assert by_type["rf-detr"]["exit_code"] is None
        assert "timed out" in by_type["rf-detr"]["error"]

        # lodestar still ran to completion despite rf-detr's timeout.
        assert by_type["lodestar"]["exit_code"] == 0

    def test_timeout_kill_survives_process_already_exited(self, tmp_path):
        """TOCTOU: the process can exit on its own between TimeoutExpired firing and
        the kill attempt running, in which case os.getpgid/os.killpg raise
        ProcessLookupError -- this must not propagate and crash the whole comparison."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(tmp_path, input_path, ["rf-detr:ckpt.pth"])

        timed_out_proc = _mock_popen(None)
        timed_out_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="track.py", timeout=args.model_timeout
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen", return_value=timed_out_proc),
            patch("os.getpgid", side_effect=ProcessLookupError),
            patch("os.killpg") as mock_killpg,
        ):
            manifest_path, any_model_failed = run_full_comparison(args, parser)

        assert any_model_failed is True
        mock_killpg.assert_not_called()  # never reached -- getpgid raised first
        manifest = json.loads(manifest_path.read_text())
        assert "timed out" in manifest["models"][0]["error"]

    def test_popen_invoked_with_correct_argv_and_process_group(self, tmp_path):
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("input: x\n")

        with patch("subprocess.Popen") as mock_popen_cls:
            mock_popen_cls.return_value = _mock_popen(0)
            model_comparison.run_model_tracking(config_path, timeout=30)

        mock_popen_cls.assert_called_once_with(
            ["uv", "run", "python", "-u", "track.py", "--config", str(config_path)],
            cwd=model_comparison.SCRIPT_DIR,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        mock_popen_cls.return_value.communicate.assert_called_once_with(timeout=30)

    def test_duplicate_model_type_gets_suffixed_output_dir(self, tmp_path):
        """Two ModelSpecs sharing model_type='rf-detr' (different checkpoints) must not
        collide on output dir/config name — this is path hygiene only, not checkpoint
        comparison (the config writers don't accept a checkpoint parameter today)."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path, input_path, ["rf-detr:ckptA.pth", "rf-detr:ckptB.pth"]
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        manifest = json.loads(manifest_path.read_text())
        output_dirs = [m["output_dir"] for m in manifest["models"]]
        assert len(set(output_dirs)) == 2
        assert output_dirs[0].endswith("rf-detr")  # first occurrence keeps unsuffixed naming
        assert output_dirs[1].endswith("rf-detr-2")

    def test_input_not_found_is_a_cli_error(self, tmp_path):
        args, parser = _parse_full_run_args(
            tmp_path, tmp_path / "does_not_exist.tif", ["rf-detr:ckpt.pth"]
        )
        with pytest.raises(SystemExit):
            run_full_comparison(args, parser)


class TestDatasetProfileFlag:
    """--dataset-profile threading through the full-run comparison, per R2/R3/R7."""

    @staticmethod
    def _write_profile(tmp_path, name="profile.yaml"):
        profile_path = tmp_path / name
        profile_path.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        return profile_path

    def test_reaches_generated_config_for_rfdetr(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")
        profile_path = self._write_profile(tmp_path)

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth"],
            extra_argv=["--dataset-profile", str(profile_path)],
        )

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        config_path = json.loads(manifest_path.read_text())["models"][0]["config"]
        parsed = yaml.safe_load(Path(config_path).read_text())
        assert parsed["dataset_profile"] == str(profile_path)

    def test_reaches_generated_config_for_lodestar_too(self, tmp_path):
        """Unlike checkpoint (rf-detr-only), dataset_profile must reach a lodestar
        spec's generated config as well -- it drives lodestar's own
        box_size/nms_distance/search_range."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")
        profile_path = self._write_profile(tmp_path)

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["lodestar:../data-setup/models/lodestar_model_15/model.pt"],
            extra_argv=["--dataset-profile", str(profile_path)],
        )

        with (
            patch("tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        config_path = json.loads(manifest_path.read_text())["models"][0]["config"]
        parsed = yaml.safe_load(Path(config_path).read_text())
        assert parsed["dataset_profile"] == str(profile_path)

    def test_omitted_by_default(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(tmp_path, input_path, ["rf-detr:ckpt.pth"])
        assert args.dataset_profile is None

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        config_path = json.loads(manifest_path.read_text())["models"][0]["config"]
        parsed = yaml.safe_load(Path(config_path).read_text())
        assert "dataset_profile" not in parsed

    def test_combines_with_crop_and_bridge_gap(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")
        profile_path = self._write_profile(tmp_path)

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth"],
            extra_argv=[
                "--dataset-profile",
                str(profile_path),
                "--crop",
                "512x512",
                "--bridge-gap",
                "10",
            ],
        )

        assert args.dataset_profile == str(profile_path)
        assert args.crop == "512x512"
        assert args.bridge_gap == 10

    def test_invalid_path_is_a_cli_error_before_any_config_is_written(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path,
            input_path,
            ["rf-detr:ckpt.pth"],
            extra_argv=["--dataset-profile", str(tmp_path / "does_not_exist.yaml")],
        )

        with patch(
            "tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer
        ) as mock_writer:
            with pytest.raises(SystemExit):
                run_full_comparison(args, parser)

        mock_writer.assert_not_called()


class TestCheckpointPassthroughRegression:
    """Direct regression coverage for the bug where write_rfdetr_config() hardcoded its
    checkpoint path and silently ignored --models' checkpoint argument entirely -- prior
    coverage only exercised this incidentally (e.g. via the fake writer's own default
    behavior), never asserting the checkpoint value actually reaches the writer call."""

    def test_rfdetr_checkpoint_reaches_the_writer_call(self, tmp_path):
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path, input_path, ["rf-detr:../rf-detr/checkpoints-a40/checkpoint_best_ema.pth"]
        )

        with (
            patch(
                "tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer
            ) as mock_writer,
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            run_full_comparison(args, parser)

        assert (
            mock_writer.call_args.kwargs["checkpoint"]
            == "../rf-detr/checkpoints-a40/checkpoint_best_ema.pth"
        )

    def test_lodestar_spec_does_not_forward_a_checkpoint_kwarg(self, tmp_path):
        """write_lodestar_config() never accepted a checkpoint parameter -- its hardcoded
        default already matches the canonical lodestar checkpoint. Confirms the rf-detr-only
        checkpoint threading in _write_model_config() doesn't spill over to lodestar specs."""
        input_path = tmp_path / "video.tif"
        input_path.write_bytes(b"fake")

        args, parser = _parse_full_run_args(
            tmp_path, input_path, ["lodestar:../data-setup/models/lodestar_model_15/model.pt"]
        )

        with (
            patch(
                "tracker_configs.write_lodestar_config", side_effect=_fake_lodestar_writer
            ) as mock_writer,
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            run_full_comparison(args, parser)

        assert "checkpoint" not in mock_writer.call_args.kwargs


class TestTracksCsvPathRegression:
    """Direct regression coverage for the bug where run_full_comparison() looked for
    tracks.csv directly at model_output_dir, but track.py always nests actual output under
    model_output_dir/<input's Path.stem>/ for batch-mode support -- prior coverage fully
    mocked compute_track_stats with no assertion on the path it was called with."""

    def test_directory_input_uses_nested_stem_path(self, tmp_path):
        input_dir = tmp_path / "synthetic_frames"
        input_dir.mkdir()
        (input_dir / "frame_00000.png").write_bytes(b"fake")

        args, parser = _parse_full_run_args(tmp_path, input_dir, ["rf-detr:ckpt.pth"])

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        model_output_dir = Path(json.loads(manifest_path.read_text())["models"][0]["output_dir"])
        called_path = Path(mock_stats.call_args.args[0])
        assert called_path == model_output_dir / "synthetic_frames" / "tracks.csv"

    def test_file_input_uses_nested_stem_path_not_full_filename(self, tmp_path):
        """Path("video.tif").stem == "video" -- the extension must be stripped, not just
        the directory-input case coincidentally working because a directory has no suffix."""
        input_file = tmp_path / "video.tif"
        input_file.write_bytes(b"fake")

        args, parser = _parse_full_run_args(tmp_path, input_file, ["rf-detr:ckpt.pth"])

        with (
            patch("tracker_configs.write_rfdetr_config", side_effect=_fake_rfdetr_writer),
            patch("subprocess.Popen") as mock_popen_cls,
            patch("analyze_tracks.compute_track_stats") as mock_stats,
        ):
            mock_popen_cls.return_value = _mock_popen(0)
            mock_stats.return_value = {"n_tracks": 1}

            manifest_path, _ = run_full_comparison(args, parser)

        model_output_dir = Path(json.loads(manifest_path.read_text())["models"][0]["output_dir"])
        called_path = Path(mock_stats.call_args.args[0])
        assert called_path == model_output_dir / "video" / "tracks.csv"


class TestExistingImageModeUnaffected:
    def test_image_mode_still_runs_end_to_end(self, tmp_path, monkeypatch):
        import model_comparison

        image_path = tmp_path / "frame.png"
        image_path.write_bytes(b"fake-image-bytes")
        output_path = tmp_path / "out.png"

        fake_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        fake_helpers = MagicMock()
        fake_helpers.load_frames.return_value = [fake_frame]

        mock_detections = MagicMock()
        mock_detections.xyxy = np.empty((0, 4), dtype=np.float32)
        mock_detections.confidence = None
        mock_detections.__len__ = lambda self: 0

        argv = [
            "model_comparison.py",
            "--image",
            str(image_path),
            "--models",
            "yolo:ckpt.pt",
            "--output",
            str(output_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with (
            patch.object(model_comparison, "_load_track_helpers", return_value=fake_helpers),
            patch.object(model_comparison, "_load_model", return_value=MagicMock()),
            patch.object(model_comparison, "run_detection", return_value=mock_detections),
        ):
            model_comparison.main()

        assert output_path.exists()
