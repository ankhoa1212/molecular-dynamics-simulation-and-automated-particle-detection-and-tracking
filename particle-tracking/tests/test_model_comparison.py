import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Set non-interactive backend before pyplot is imported anywhere
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_comparison import (
    ModelSpec,
    _build_rfdetr_script,
    build_comparison_figure,
    default_device,
    parse_model_spec,
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
