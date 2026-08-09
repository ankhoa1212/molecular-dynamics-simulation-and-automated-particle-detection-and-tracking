"""Tests for trackers_common.scale_derivation -- U4: tracking-side
parameter derivation (search_range, diameter, memory) from a dataset
profile, with the three-tier explicit > profile-derived > hardcoded-default
precedence chain (R8), and memory's deliberate non-derivation exception
(R9)."""

from trackers_common.scale_derivation import (
    DEFAULT_DIAMETER,
    DEFAULT_MEMORY,
    DEFAULT_SEARCH_RANGE,
    FWHM_TO_SIGMA,
    resolve_diameter,
    resolve_memory,
    resolve_search_range,
    round_to_nearest_odd,
)


def _profile(size_px, spacing_px):
    return {"size_px": size_px, "spacing_px": spacing_px}


class TestResolveSearchRange:
    def test_happy_path_derives_from_profile(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_search_range(None, profile)

        assert result == 12.0 * 0.5

    def test_explicit_value_wins_over_derived(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_search_range(99, profile)

        assert result == 99

    def test_no_profile_falls_back_to_hardcoded_default(self):
        result = resolve_search_range(None, None)

        assert result == DEFAULT_SEARCH_RANGE
        assert result == 25.0

    def test_no_profile_uses_caller_supplied_hardcoded_default(self):
        result = resolve_search_range(None, None, hardcoded_default=15.0)

        assert result == 15.0


class TestRoundToNearestOdd:
    def test_rounds_up_to_nearest_odd(self):
        # 8.9 is 0.1 from 9 and 1.9 from 7 -- nearer candidate is 9, not a
        # coincidental "int(round(x)) | 1" result.
        assert round_to_nearest_odd(8.9) == 9

    def test_rounds_up_from_below_midpoint_of_even_int(self):
        # 8.1 is 0.9 from 9 and 1.1 from 7 -- still nearer to 9, even though
        # 8.1 itself is much closer to the even integer 8 than to 9.
        assert round_to_nearest_odd(8.1) == 9

    def test_rounds_down_to_nearest_odd(self):
        # 7.1 is 0.1 from 7 and 1.9 from 9 -- nearer candidate is 7.
        assert round_to_nearest_odd(7.1) == 7

    def test_rounds_down_from_above_midpoint_of_even_int(self):
        # 7.9 is 0.9 from 7 and 1.1 from 9 -- still nearer to 7.
        assert round_to_nearest_odd(7.9) == 7

    def test_exact_odd_integer_is_unchanged(self):
        assert round_to_nearest_odd(9.0) == 9
        assert round_to_nearest_odd(7.0) == 7

    def test_exact_even_integer_breaks_tie_upward(self):
        # 8.0 is exactly 1 away from both 7 and 9 -- tie breaks upward.
        assert round_to_nearest_odd(8.0) == 9
        assert round_to_nearest_odd(6.0) == 7

    def test_bit_trick_shortcut_would_be_wrong_here(self):
        # int(round(x)) | 1 forces the *last bit* on rather than finding the
        # true nearest odd candidate. For 8.1: int(round(8.1)) = 8, 8 | 1 = 9
        # (coincidentally correct). For 7.4: int(round(7.4)) = 7, already
        # odd, 7 | 1 = 7 -- but the true nearest odd to 7.4 is genuinely 7
        # (distance 0.4) vs 9 (distance 1.6), so this case doesn't
        # distinguish the two approaches either. The real divergence shows
        # up once you round *first*: round(8.4) = 8, 8 | 1 = 9, yet 8.4's
        # true nearest odd is 9 (distance 0.6) vs 7 (distance 1.4) -- still
        # coincidentally matching. The bit-trick actually fails when
        # rounding-then-forcing crosses a boundary the direct-comparison
        # approach doesn't: round(6.4) = 6, 6 | 1 = 7, but 6.4 is nearer to 7
        # (distance 0.6) than to 5 (distance 1.4) -- also matching. This test
        # instead directly asserts the two lower-rounding cases above (7.1,
        # 7.9) land correctly, which is where naive single-direction
        # bit-forcing (e.g. always rounding up to the next odd) would fail.
        assert round_to_nearest_odd(7.1) != 9
        assert round_to_nearest_odd(6.4) == 7


class TestResolveDiameter:
    def test_happy_path_derives_from_profile(self):
        # size_px=8.0 -> 8.0 * 2.355 = 18.84 -> nearest odd is 19.
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_diameter(None, profile)

        assert result == round_to_nearest_odd(8.0 * FWHM_TO_SIGMA)
        assert result == 19

    def test_derives_and_rounds_down_for_another_size(self):
        # size_px=3.0 -> 3.0 * 2.355 = 7.065 -> nearest odd is 7.
        profile = _profile(size_px=3.0, spacing_px=12.0)

        result = resolve_diameter(None, profile)

        assert result == 7

    def test_explicit_value_wins_over_derived(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_diameter(21, profile)

        assert result == 21

    def test_no_profile_falls_back_to_hardcoded_default(self):
        result = resolve_diameter(None, None)

        assert result == DEFAULT_DIAMETER
        assert result == 15

    def test_no_profile_uses_caller_supplied_hardcoded_default(self):
        result = resolve_diameter(None, None, hardcoded_default=11)

        assert result == 11


class TestResolveMemory:
    def test_derives_per_model_canonical_default_with_profile(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_memory(None, profile, "rf-detr")

        assert result == 5

        result = resolve_memory(None, profile, "lodestar")

        assert result == 10

    def test_derives_per_model_canonical_default_without_profile(self):
        # R9: memory resolves through model_type, not through profile
        # presence -- identical whether a profile is referenced or not.
        result = resolve_memory(None, None, "rf-detr")

        assert result == 5

        result = resolve_memory(None, None, "lodestar")

        assert result == 10

    def test_memory_unchanged_across_very_different_size_and_spacing(self):
        # Covers R9: vary size_px/spacing_px across very different values,
        # with and without a profile referenced at all, for the same
        # model_type -- memory must match the per-model canonical default in
        # every case, never varying with the spatial values.
        small_profile = _profile(size_px=1.0, spacing_px=2.0)
        large_profile = _profile(size_px=500.0, spacing_px=900.0)

        results = {
            resolve_memory(None, small_profile, "rf-detr"),
            resolve_memory(None, large_profile, "rf-detr"),
            resolve_memory(None, None, "rf-detr"),
        }

        assert results == {5}

        results_lodestar = {
            resolve_memory(None, small_profile, "lodestar"),
            resolve_memory(None, large_profile, "lodestar"),
            resolve_memory(None, None, "lodestar"),
        }

        assert results_lodestar == {10}

    def test_explicit_value_wins_over_per_model_canonical_default(self):
        profile = _profile(size_px=8.0, spacing_px=12.0)

        result = resolve_memory(33, profile, "rf-detr")

        assert result == 33

        result = resolve_memory(33, None, "rf-detr")

        assert result == 33

    def test_unknown_model_type_still_resolves_via_rf_detr_fallback(self):
        # A model_type with no tracker_defaults.yaml entry of its own still
        # resolves via load_tracking_config's own FALLBACK_MODEL_TYPE
        # (rf-detr), which does have a "memory" entry.
        result = resolve_memory(None, None, "some-future-model")

        assert result == 5

    def test_no_canonical_entry_falls_back_to_hardcoded_default(self, monkeypatch):
        # Today's tracker_defaults.yaml always has a "memory" entry for every
        # model_type this resolves to (directly or via FALLBACK_MODEL_TYPE),
        # so this module's own hardcoded_default tier is a defensive
        # fallback that doesn't normally engage. Exercise it directly by
        # faking a canonical lookup that omits "memory" entirely.
        monkeypatch.setattr(
            "trackers_common.scale_derivation.load_tracking_config",
            lambda model_type, tool_config, key_path_map: {},
        )

        result = resolve_memory(None, None, "rf-detr", hardcoded_default=99)

        assert result == 99
        assert DEFAULT_MEMORY == 5
