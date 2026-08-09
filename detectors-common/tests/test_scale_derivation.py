"""Tests for detectors_common.scale_derivation -- U3: detection-side
parameter derivation (box_size, nms_distance, tile_size) from a dataset
profile, with the three-tier explicit > profile-derived > hardcoded-default
precedence chain (R6, R7)."""

from detectors_common.scale_derivation import (
    DEFAULT_BOX_SIZE,
    DEFAULT_NMS_DISTANCE,
    DEFAULT_TILE_SIZE,
    FWHM_TO_SIGMA,
    TARGET_PARTICLES_PER_TILE_SIDE,
    TILE_SIZE_FLOOR_PX,
    resolve_box_size,
    resolve_nms_distance,
    resolve_tile_size,
)


def _profile(size_px, spacing_px):
    return {"size_px": size_px, "spacing_px": spacing_px}


class TestResolveBoxSize:
    def test_happy_path_derives_from_profile(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_box_size(None, profile)

        assert result == 8.0 * FWHM_TO_SIGMA

    def test_explicit_value_wins_over_derived(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_box_size(99, profile)

        assert result == 99

    def test_no_profile_falls_back_to_hardcoded_default(self):
        result = resolve_box_size(None, None)

        assert result == DEFAULT_BOX_SIZE
        assert result == 40

    def test_no_profile_uses_caller_supplied_hardcoded_default(self):
        result = resolve_box_size(None, None, hardcoded_default=512)

        assert result == 512


class TestResolveNmsDistance:
    def test_happy_path_derives_from_profile(self):
        # AE1: size_px=8, spacing_px=12 -> min(8*1.0, 12*0.5) = 6.0
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_nms_distance(None, profile)

        assert result == 6.0

    def test_explicit_value_wins_over_derived(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_nms_distance(7, profile)

        assert result == 7

    def test_no_profile_falls_back_to_hardcoded_default(self):
        result = resolve_nms_distance(None, None)

        assert result == DEFAULT_NMS_DISTANCE
        assert result == 30

    def test_no_profile_uses_caller_supplied_hardcoded_default(self):
        result = resolve_nms_distance(None, None, hardcoded_default=5)

        assert result == 5

    def test_spacing_based_cap_engages_for_large_size_px(self):
        # size_px * 1.0 = 20.0 would exceed spacing_px * 0.5 = 5.0 -- capped.
        profile = _profile(size_px=20.0, spacing_px=10.0)

        result = resolve_nms_distance(None, profile)

        assert result == 5.0

    def test_size_based_value_used_when_below_cap(self):
        # size_px * 1.0 = 3.0 stays below spacing_px * 0.5 = 6.0 -- uncapped.
        profile = _profile(size_px=3.0, spacing_px=12.0)

        result = resolve_nms_distance(None, profile)

        assert result == 3.0


class TestResolveTileSize:
    def test_happy_path_derives_from_profile(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_tile_size(None, profile, frame_width=2048, frame_height=2048)

        expected = 12.0 * TARGET_PARTICLES_PER_TILE_SIDE
        assert result == expected

    def test_explicit_value_wins_over_derived(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_tile_size(777, profile, frame_width=2048, frame_height=2048)

        assert result == 777

    def test_no_profile_falls_back_to_hardcoded_default(self):
        result = resolve_tile_size(None, None, frame_width=2048, frame_height=2048)

        assert result == DEFAULT_TILE_SIZE

    def test_no_profile_uses_caller_supplied_hardcoded_default(self):
        # e.g. particle-tracking's own existing 1024 or verification's own 160.
        result = resolve_tile_size(
            None, None, frame_width=2048, frame_height=2048, hardcoded_default=1024
        )

        assert result == 1024

        result = resolve_tile_size(
            None, None, frame_width=2048, frame_height=2048, hardcoded_default=160
        )

        assert result == 160

    def test_floor_clamp_engages_for_very_dense_profile(self):
        # spacing_px small enough that spacing_px * TARGET_PARTICLES_PER_TILE_SIDE
        # would fall below the floor.
        profile = _profile(size_px=1.0, spacing_px=2.0)
        raw = 2.0 * TARGET_PARTICLES_PER_TILE_SIDE
        assert raw < TILE_SIZE_FLOOR_PX  # sanity check on the fixture itself

        result = resolve_tile_size(None, profile, frame_width=2048, frame_height=2048)

        assert result == TILE_SIZE_FLOOR_PX

    def test_frame_dimensions_ceiling_clamp_engages_for_sparse_profile(self):
        # spacing_px large enough that spacing_px * TARGET_PARTICLES_PER_TILE_SIDE
        # would exceed the frame's own (smaller) dimension.
        profile = _profile(size_px=10.0, spacing_px=50.0)
        raw = 50.0 * TARGET_PARTICLES_PER_TILE_SIDE
        assert raw > 300  # sanity check on the fixture itself

        result = resolve_tile_size(None, profile, frame_width=300, frame_height=400)

        assert result == 300

    def test_ceiling_uses_smaller_of_width_and_height(self):
        profile = _profile(size_px=10.0, spacing_px=50.0)

        result = resolve_tile_size(None, profile, frame_width=400, frame_height=250)

        assert result == 250
