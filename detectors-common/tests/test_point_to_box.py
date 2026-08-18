"""Tests for detectors_common.point_to_box -- U1: shared centroid-to-xyxy
box synthesis extracted from detect_lodestar and verification/benchmark.py's
detect_trackpy (R3)."""

import numpy as np

from detectors_common.point_to_box import points_to_xyxy


class TestPointsToXyxy:
    def test_single_center_produces_correct_xyxy(self):
        result = points_to_xyxy([(10, 10)], box_size=4)

        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0], [8, 8, 12, 12])

    def test_multiple_centers_produce_one_row_each_in_input_order(self):
        centers = [(10, 10), (0, 0), (5, 20)]

        result = points_to_xyxy(centers, box_size=2)

        assert result.shape == (3, 4)
        np.testing.assert_allclose(result[0], [9, 9, 11, 11])
        np.testing.assert_allclose(result[1], [-1, -1, 1, 1])
        np.testing.assert_allclose(result[2], [4, 19, 6, 21])

    def test_zero_box_size_produces_zero_area_box_without_raising(self):
        result = points_to_xyxy([(3, 4)], box_size=0)

        assert result.shape == (1, 4)
        x1, y1, x2, y2 = result[0]
        assert x1 == x2
        assert y1 == y2

    def test_empty_input_returns_empty_array_without_raising(self):
        result = points_to_xyxy([], box_size=4)

        assert result.shape == (0, 4)

    def test_empty_ndarray_input_returns_empty_array_without_raising(self):
        result = points_to_xyxy(np.empty((0, 2)), box_size=4)

        assert result.shape == (0, 4)
