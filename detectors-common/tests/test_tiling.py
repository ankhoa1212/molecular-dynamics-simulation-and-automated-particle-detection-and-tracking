"""Tests for detectors_common.tiling — U4: detect_with_tiling's bounds-guarded
tile_starts (the negative-start-index bug verification's prior standalone
copy had, dormant only because its synthetic frames always equaled
tile_size in both dimensions)."""

from unittest import mock

import numpy as np
import supervision as sv

from detectors_common.tiling import detect_with_tiling


def _detections(xyxy):
    xyxy = np.array(xyxy, dtype=np.float32)
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
    )


class TestDetectWithTiling:
    def test_frame_within_tile_size_in_both_dimensions_bypasses_tiling(self):
        model = mock.Mock()
        model.predict.return_value = _detections([[10, 10, 20, 20]])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = detect_with_tiling(
            model, frame, threshold=0.3, tile_size=512, overlap=50, nms_threshold=0.3
        )

        model.predict.assert_called_once_with(frame, threshold=0.3)
        assert len(result) == 1

    def test_frame_smaller_than_tile_size_in_exactly_one_dimension_does_not_use_negative_start(
        self,
    ):
        """The bug case: H > tile_size (tiling-eligible overall) but W <= tile_size.
        Without the bounds guard, tile_starts(W) computes range(0, W - tile_size,
        stride) -> empty (W - tile_size is negative), then appends that negative
        value as the sole start index. When 0 < tile_size - W < W (as here:
        tile_size=150, W=100 -> -50, and -W=-100), numpy slicing doesn't clamp
        to 0 — it wraps: frame[:, -50:100] normalizes to frame[:, 50:100], a
        50-wide slice missing the frame's entire left half, instead of the
        guarded [0]-start's full frame[:, 0:100]. (A too-large tile_size like
        512 against W=100 would clamp to 0 regardless of the guard, masking
        the bug — these parameters are chosen specifically to not clamp.)
        """
        model = mock.Mock()
        model.predict.return_value = sv.Detections.empty()
        # H=300 (tiling-eligible at tile_size=150), W=100 (fits within tile_size).
        frame = np.zeros((300, 100, 3), dtype=np.uint8)

        detect_with_tiling(
            model, frame, threshold=0.3, tile_size=150, overlap=20, nms_threshold=0.3
        )

        # Every tile call's x-slice must start at 0 and cover the full 100px
        # width — never a wrapped slice that drops the left half of the frame.
        for call in model.predict.call_args_list:
            tile = call.args[0]
            assert tile.shape[1] == 100, (
                f"expected full 100px width from x0=0, got {tile.shape[1]}px "
                "— tile_starts likely used a negative, wrapping start index"
            )

    def test_frame_larger_than_tile_size_in_both_dimensions_merges_overlapping_tiles(self):
        model = mock.Mock()
        model.predict.return_value = _detections([[5, 5, 15, 15]])
        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        result = detect_with_tiling(
            model, frame, threshold=0.3, tile_size=512, overlap=50, nms_threshold=0.3
        )

        assert model.predict.call_count > 1  # actually tiled, not a single bypass call
        assert len(result) > 0

    def test_no_detections_in_any_tile_returns_empty(self):
        model = mock.Mock()
        model.predict.return_value = sv.Detections.empty()
        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        result = detect_with_tiling(
            model, frame, threshold=0.3, tile_size=512, overlap=50, nms_threshold=0.3
        )

        assert len(result) == 0
